import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from ..modeling_utils import ConfigMixin, ModelMixin, register_to_config
from ..base import TransformerBlock, dtype_str_to_torch
from typing import Tuple
from pathlib import Path
from .utils import (
    get_index_and_grouping,
    get_index_and_grouping_linear,
    get_index_and_grouping_recursive_half,
    get_index_and_grouping_recursive_quarter,
    get_io_interface_mask,
    reorder_and_group_token_batch,
    remove_pads_from_attn_mask,
    get_attn_mask,
    reset_to_ori_order,
    get_sigma_reorder_and_grouping,
    get_geo_reorder_and_grouping,
    get_vanilla_reorder_and_grouping,
)
from einops import repeat, rearrange, reduce, einsum
from jaxtyping import Float, Int, Bool


class EasyTransformer(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        num_layers_input: int,
        num_layers_output: int,
        image_len: int,
        d_model: int,
        d_mlp: int,
        num_heads: int,
        dtype: str,
        inference_grouping_type: str,
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dtype_torch = dtype_str_to_torch[dtype]

        self.register_to_config(mask_token_id=vocab_size - 1)

        self.input_pos_embed = nn.Embedding(max_seq_len, d_model)
        self.output_pos_embed = nn.Embedding(max_seq_len, d_model)

        self.tok_embed = nn.Embedding(vocab_size, d_model)

        self.input_blocks = nn.ModuleList(
            [
                TransformerBlock(d_model=d_model, d_mlp=d_mlp, num_heads=num_heads)
                for _ in range(num_layers_input)
            ]
        )
        self.output_blocks = nn.ModuleList(
            [
                TransformerBlock(d_model=d_model, d_mlp=d_mlp, num_heads=num_heads)
                for _ in range(num_layers_output)
            ]
        )
        self.unembed = nn.Linear(d_model, vocab_size)

        # Input-output interface
        self.input_k_transform = nn.Linear(d_model, d_model)

        self.oi_pos_ln = nn.LayerNorm(d_model)

        self.register_to_config(**kwargs)


    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True

    def _forward(
        self,
        input_ids_reordered: Int[Tensor, "batch seq"],
        reorder_idx_seq: Int[Tensor, "batch seq"],
        attn_mask: Bool[Tensor, "batch seq seq"],
        io_interface_mask: Bool[Tensor, "batch seq seq"],
        inference_groups: Int[Tensor, "batch seq"],
    ) -> Float[Tensor, "batch seq vocab"]:
        input_ids_reordered = input_ids_reordered.contiguous()
        batch_size, seq_len = input_ids_reordered.shape

        with torch.autocast(dtype=self.dtype_torch, device_type=input_ids_reordered.device.type):
            resid = self.tok_embed(input_ids_reordered) + self.input_pos_embed(
                reorder_idx_seq
            )

            output_pos_embed = self.output_pos_embed(reorder_idx_seq)

            # Add sum of output position embeddings to each input position
            # generating a prediction for them
            oi_interface_mask = rearrange(io_interface_mask, "batch input output -> batch output input").float()
            resid += self.oi_pos_ln(einsum(
                oi_interface_mask,
                output_pos_embed,
                "batch seq seq_agg, batch seq_agg d_model -> batch seq d_model"
            ))

            for input_block in self.input_blocks:
                resid = input_block(resid, attn_mask=attn_mask)

            # Now that we have the final residual stream for the input block at each
            # position, we need to transfer the residual stream information from
            # each inference group of input blocks to the next inference group of
            # output blocks via attention. We do this by initializing the full seq x
            # seq attention mask, which is a wasteful but temporary solution.
            output_resid = output_pos_embed
            assert output_resid.shape == (batch_size, seq_len, self.config.d_model)
            assert resid.shape == output_resid.shape

            # Compute full attention weight matrix with initial output residual
            # streams as Q and final input residual streams (transformed) as K.
            attn_weights = einsum(
                output_resid, # Q
                self.input_k_transform(resid), # K
                "batch seq_q d_model, batch seq_k d_model -> batch seq_q seq_k"
            ) / seq_len ** 0.5 # scaling factor

            # Apply our attention mask
            attn_weights = torch.where(
                # Leave attention weights in rows without any query positions in
                # attention mask unchanged so we don't get NaNs in softmax. This
                # is okay because we also set the attention weights for mask=0 to
                # zero after the softmax.
                io_interface_mask | ~reduce(io_interface_mask, "batch seq_q seq_k -> batch seq_q 1", "sum").bool(),
                attn_weights,
                float("-inf"),
            )

            attn_weights = torch.softmax(attn_weights, dim=-1)

            attn_weights = torch.where(
                io_interface_mask,
                attn_weights,
                0.0,
            )

            assert torch.all(torch.isfinite(attn_weights))

            attn_result = einsum(
                attn_weights, # QK
                resid, # V
                "batch seq_q seq_k, batch seq_k d_model -> batch seq_q d_model"
            )

            assert attn_result.shape == (batch_size, seq_len, self.config.d_model)

            resid = output_resid + attn_result

            # Now continue with remaining blocks like a typical transformer
            for output_block in self.output_blocks:
                resid = output_block(resid, attn_mask=attn_mask)

            logits = self.unembed(resid)

        return logits
    
    def get_img_reorder_idx_and_inference_groups(
        self,
        batch_size: int,
        device: torch.device,
        img_reorder_idx: Int[Tensor, "batch img_seq"] = None,
        img_inference_groups: Int[Tensor, "batch img_seq"] = None,
    ) -> Tuple[Int[Tensor, "batch img_seq"], Int[Tensor, "batch img_seq"]]:

        image_len = self.config.image_len
        image_dim = int(image_len ** 0.5)
        assert image_dim * image_dim == image_len

        if img_reorder_idx is not None:
            img_reorder_idx = img_reorder_idx.to(device)
        if img_inference_groups is not None:
            img_inference_groups = img_inference_groups.to(device)


        # Construct image reorder idx and inference groups if not provided
        if img_reorder_idx is None or img_inference_groups is None:

            root_only = False
            if self.config.inference_grouping_type == "recursive":
                root_only = True
                img_reorder_idx_root, img_inference_groups_root = get_index_and_grouping(image_dim)
            elif self.config.inference_grouping_type == "linear":
                root_only = True
                img_reorder_idx_root, img_inference_groups_root = get_index_and_grouping_linear(
                    self.config.inference_grouping_num,
                    image_dim,
                )
            elif self.config.inference_grouping_type == "recursive_half":
                root_only = True
                img_reorder_idx_root, img_inference_groups_root = get_index_and_grouping_recursive_half(image_dim) 
            elif self.config.inference_grouping_type == "recursive_quarter":
                root_only = True
                img_reorder_idx_root, img_inference_groups_root = get_index_and_grouping_recursive_quarter(image_dim) 
            elif self.config.inference_grouping_type == "sigma":
                img_reorder_idx, img_inference_groups = get_sigma_reorder_and_grouping(
                    batch_size,
                    image_dim,
                    device=device,
                )
            elif self.config.inference_grouping_type == "geo":
                img_reorder_idx, img_inference_groups = get_geo_reorder_and_grouping(
                    batch_size,
                    image_dim,
                    self.config.inference_grouping_prob,
                    device=device,
                )
            elif self.config.inference_grouping_type == "vanilla":
                img_reorder_idx, img_inference_groups = get_vanilla_reorder_and_grouping(
                    batch_size,
                    image_dim,
                    device=device,
                )
            else:
                raise ValueError(f"Unknown inference grouping type: {self.config.inference_grouping_type}")


            if root_only:
                if img_reorder_idx is None:
                    img_reorder_idx = repeat(
                        img_reorder_idx_root,
                        "img_seq -> batch img_seq",
                        batch=batch_size,
                    ).clone()
                if img_inference_groups is None:
                    img_inference_groups = repeat(
                        img_inference_groups_root,
                        "img_seq -> batch img_seq",
                        batch=batch_size,
                    ).clone()

        assert img_reorder_idx.shape == (batch_size, self.config.image_len)
        assert img_inference_groups.shape == (batch_size, self.config.image_len)
        for i in range(batch_size):
            assert torch.equal(img_inference_groups[i], torch.sort(img_inference_groups[i])[0])
            assert img_reorder_idx[i].unique().numel() == image_len

        return img_reorder_idx, img_inference_groups

    def forward(
        self,
        input_ids: Int[Tensor, "batch seq"],
        batch_size_t2i: int,
        batch_size_lm: int,
        batch_size_mmu: int,
        pad_id: int,
        soi_id: int,
        eoi_id: int,
        ignore_id: int,
        labels: Int[Tensor, "batch seq"] = None,
        global_step: int = None,
        label_smoothing: float = 0.0,
        keep_prediction_order: bool = False,
        img_reorder_idx: Int[Tensor, "batch img_seq"] = None,
        img_inference_groups: Int[Tensor, "batch img_seq"] = None,
        **kwargs,
    ):
        with torch.no_grad():
            device = input_ids.device
            batch_size = input_ids.shape[0]
            assert batch_size == batch_size_t2i + batch_size_lm + batch_size_mmu
            seq_len = input_ids.shape[1]
            assert seq_len <= self.config.max_seq_len

            img_reorder_idx, img_inference_groups = self.get_img_reorder_idx_and_inference_groups(
                batch_size_t2i + batch_size_mmu,
                device,
                img_reorder_idx,
                img_inference_groups,
            )

            # Construct inference groups and reorder idxs for each task type then combine
            # t2i and mmu have images so we need to reorder
            (reorder_idx_t2i_batch, reorder_idx_t2i_seq), inference_groups_t2i = (
                reorder_and_group_token_batch(
                    input_ids[:batch_size_t2i],
                    soi_id,
                    eoi_id,
                    self.config.image_len,
                    img_reorder_idx[:batch_size_t2i],
                    img_inference_groups[:batch_size_t2i],
                )
            )
            # TODO: combine these into single call to reorder and group
            (reorder_idx_mmu_batch, reorder_idx_mmu_seq), inference_groups_mmu = (
                reorder_and_group_token_batch(
                    input_ids[batch_size_t2i + batch_size_lm :],
                    soi_id,
                    eoi_id,
                    self.config.image_len,
                    img_reorder_idx[batch_size_t2i:],
                    img_inference_groups[batch_size_t2i:],
                )
            )
            reorder_idx_lm_seq = repeat(
                torch.arange(seq_len, device=device),
                "seq -> batch seq",
                batch=batch_size_lm,
            ).clone()
            reorder_idx_lm_batch = torch.arange(
                batch_size_lm, device=device
            ).unsqueeze(1)
            inference_groups_lm = reorder_idx_lm_seq.clone()

            # Combine reorder idx
            reorder_idx_batch = torch.cat(
                [
                    reorder_idx_t2i_batch,
                    reorder_idx_lm_batch + batch_size_t2i,
                    reorder_idx_mmu_batch + batch_size_t2i + batch_size_lm,
                ]
            )
            reorder_idx_seq = torch.cat(
                [reorder_idx_t2i_seq, reorder_idx_lm_seq, reorder_idx_mmu_seq], dim=0
            )
            assert reorder_idx_batch.shape == (batch_size, 1)
            assert reorder_idx_seq.shape == (batch_size, seq_len)
            assert torch.unique(reorder_idx_batch).shape == (batch_size,)
            assert reorder_idx_batch[0][0] == 0

            # Combine inference groups
            inference_groups = torch.cat(
                [inference_groups_t2i, inference_groups_lm, inference_groups_mmu], dim=0
            )
            assert inference_groups.shape == (batch_size, seq_len)
            assert torch.all(inference_groups[:, 0] == 0)
            # Currently, we shouldn't have any first inference groups larger
            # than one token.
            assert torch.all((inference_groups == 0).sum(dim=1) == 1)

            # Reorder input ids
            input_ids_reordered = input_ids[reorder_idx_batch, reorder_idx_seq]

            # Construct attention mask from inference groups
            attn_mask = get_attn_mask(inference_groups).to(device)
            # Note that we don't zero out the pad tokens, because it causes
            # MultiHeadAttention to produces NaNs
            assert attn_mask.shape == (batch_size, seq_len, seq_len)
            assert attn_mask.dtype == torch.bool

            # Get input output interface mask, which is a sparse attention mask for
            # passing residuals from last input block to first output block
            io_interface_mask = get_io_interface_mask(inference_groups)
            assert io_interface_mask.shape == (batch_size, seq_len, seq_len)

            # Construct mask designating where inferences start in each batch.
            # This will exclude the initial padding tokens and the inference
            # group immediately after them.
            # Get first token after optional left padding.
            nonpad_first = (input_ids_reordered != pad_id).int().argmax(dim=1)
            # Get inference groups for each first token.
            first_inference_groups = inference_groups[torch.arange(batch_size, device=device), nonpad_first]

        logits = self._forward(
            input_ids_reordered, reorder_idx_seq, attn_mask, io_interface_mask, inference_groups,
        )

        # Assert no nans
        assert torch.all(torch.isfinite(logits)), "Logits contain NaNs or Infs"

        result = [
            logits if keep_prediction_order else reset_to_ori_order(logits, reorder_idx_batch, reorder_idx_seq)
        ]

        if labels is not None:
            assert labels.shape == input_ids.shape
            # Only padding should be ignored
            assert torch.all((labels != ignore_id) | (input_ids == pad_id))

            # Apply same reordering to labels
            labels_reordered = labels[reorder_idx_batch, reorder_idx_seq]

            # Each output is at the same position as its label, but the first
            # inference group should not be predicted, so we can replace those
            # with ignore and then shove the whole thing into cross entropy
            # without any shifts.
            labels_reordered = torch.where(
                inference_groups == first_inference_groups.unsqueeze(1), ignore_id, labels_reordered
            )
            # Reorder indexes of logits for cross entropy
            logits_rearranged = rearrange(logits, "batch seq vocab -> batch vocab seq")

            loss_t2i = F.cross_entropy(
                input=logits_rearranged[:batch_size_t2i],
                target=labels_reordered[:batch_size_t2i],
                ignore_index=ignore_id,
                label_smoothing=label_smoothing,
            )

            loss_lm = F.cross_entropy(
                input=logits_rearranged[
                    batch_size_t2i : batch_size_t2i + batch_size_lm
                ],
                target=labels_reordered[
                    batch_size_t2i : batch_size_t2i + batch_size_lm
                ],
                ignore_index=ignore_id,
                label_smoothing=label_smoothing,
            )

            # Assert that first ignore is at least num image tokens + num
            # special tokens in (excluding ignore from first inference group)
            is_ignore = labels_reordered[batch_size_t2i + batch_size_lm :, 1:] == ignore_id
            if not torch.all((torch.argmax(is_ignore.int(), dim=1) > (self.config.image_len + 2)) | ~torch.any(is_ignore, dim=1)):
                torch.save(labels_reordered, "labels_reordered.pt")
                raise Exception("Invalid mmu data")


            loss_mmu = F.cross_entropy(
                input=logits_rearranged[batch_size_t2i + batch_size_lm :],
                target=labels_reordered[batch_size_t2i + batch_size_lm :],
                ignore_index=ignore_id,
                label_smoothing=label_smoothing,
            )

            result += [loss_t2i, loss_lm, loss_mmu]

        return tuple(result)

    def sample_t2i(
        self,
        input_ids: Int[Tensor, "batch seq"],
        pad_id: int,
        soi_id: int,
        eoi_id: int,
        img_vocab_start_idx: int,
        img_reorder_idx: Int[Tensor, "batch img_seq"] = None,
        img_inference_groups: Int[Tensor, "batch img_seq"] = None,
        **kwargs,
    ):
        with torch.no_grad():
            device = input_ids.device
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            assert seq_len <= self.config.max_seq_len

            img_reorder_idx, img_inference_groups = self.get_img_reorder_idx_and_inference_groups(
                batch_size,
                device,
                img_reorder_idx,
                img_inference_groups,
            )

            # Construct inference groups and reorder idxs for each task type then combine
            (reorder_idx_batch, reorder_idx_seq), inference_groups = (
                reorder_and_group_token_batch(
                    input_ids,
                    soi_id,
                    eoi_id,
                    self.config.image_len,
                    img_reorder_idx,
                    img_inference_groups,
                )
            )

            assert reorder_idx_batch.shape == (batch_size, 1)
            assert reorder_idx_seq.shape == (batch_size, seq_len)
            assert torch.unique(reorder_idx_batch).shape == (batch_size,)
            assert reorder_idx_batch[0][0] == 0
            assert inference_groups.shape == (batch_size, seq_len)
            assert torch.all(inference_groups[:, 0] == 0)
            # Currently, we shouldn't have any first inference groups larger
            # than one token.
            assert torch.all((inference_groups == 0).sum(dim=1) == 1)

            # Reorder input ids
            input_ids_reordered = input_ids[reorder_idx_batch, reorder_idx_seq]

            # Construct attention mask from inference groups
            attn_mask = get_attn_mask(inference_groups).to(device)
            # Note that we don't zero out the pad tokens, because it causes
            # MultiHeadAttention to produces NaNs
            assert attn_mask.shape == (batch_size, seq_len, seq_len)
            assert attn_mask.dtype == torch.bool

            # Get input output interface mask, which is a sparse attention mask for
            # passing residuals from last input block to first output block
            io_interface_mask = get_io_interface_mask(inference_groups)
            assert io_interface_mask.shape == (batch_size, seq_len, seq_len)
        
        # Repeatedly generate logits on the entire batch, and fill in tokens
        # corresponding to the current inference group, until we have gone
        # through all inference groups in all rows.

        # The initial inference group for each row will correspond to the first
        # image token.
        image_pos = torch.argmax((input_ids_reordered == soi_id).int(), dim=1) + 1
        current_inference_group = inference_groups[
            torch.arange(batch_size, device=device), image_pos
        ]
        # The final inference group for each row will correspond to the
        # inference group of last image token
        last_image_pos = torch.argmax((input_ids_reordered == eoi_id).int(), dim=1) - 1
        final_inference_group = inference_groups[
            torch.arange(batch_size, device=device), last_image_pos
        ]
        while torch.any(current_inference_group <= final_inference_group):
            logits = self._forward(
                input_ids_reordered, reorder_idx_seq, attn_mask, io_interface_mask, inference_groups,
            )

            assert torch.all(torch.isfinite(logits)), "Logits contain NaNs or Infs"

            # For all tokens in each row that correspond to that row's current
            # inference group, fill in that token by sampling from the generated
            # logits.
            for i in range(batch_size):
                if current_inference_group[i] > final_inference_group[i]:
                    continue
                mask = (inference_groups[i] == current_inference_group[i])
                if torch.any(mask):
                    num_to_generate = mask.sum().item()
                    logits_selected_img = logits[i, mask, img_vocab_start_idx:]
                    probs = F.softmax(logits_selected_img, dim=-1)
                    sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    assert sampled_tokens.shape == (num_to_generate,)
                    input_ids_reordered[i, mask] = sampled_tokens + img_vocab_start_idx
            
            # Move to next inference group
            current_inference_group += 1


        return reset_to_ori_order(input_ids_reordered, reorder_idx_batch, reorder_idx_seq) 