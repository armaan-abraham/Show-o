import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from ..modeling_utils import ConfigMixin, ModelMixin, register_to_config
from ..base import TransformerBlock, dtype_str_to_torch
from pathlib import Path
from .utils import (
    get_index_and_grouping,
    get_io_interface_mask,
    reorder_and_group_token_batch,
    remove_pads_from_attn_mask,
    get_attn_mask,
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
        accelerator,
    ):
        super().__init__()
        self.accelerator = accelerator
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

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True

    def _forward(
        self,
        input_ids_reordered: Int[Tensor, "batch seq"],
        reorder_idx_seq: Int[Tensor, "batch seq"],
        attn_mask: Bool[Tensor, "batch seq seq"],
        io_interface_mask: Bool[Tensor, "batch seq seq"],
        inference_groups: Int[Tensor, "batch seq"],
        first_inference_groups: Int[Tensor, "batch"],
    ) -> Float[Tensor, "batch seq vocab"]:
        input_ids_reordered = input_ids_reordered.clone()

        batch_size, seq_len = input_ids_reordered.shape

        with torch.autocast(dtype=torch.float32, device_type=input_ids_reordered.device.type):

            resid = self.tok_embed(input_ids_reordered) + self.input_pos_embed(
                reorder_idx_seq
            )

            for input_block in self.input_blocks:
                resid = input_block(resid, attn_mask=attn_mask)

            # Now that we have the final residual stream for the input block at each
            # position, we need to transfer the residual stream information from
            # each inference group of input blocks to the next inference group of
            # output blocks via attention. We do this by initializing the full seq x
            # seq attention mask, which is a wasteful but temporary solution.
            output_resid = self.output_pos_embed(reorder_idx_seq)
            assert output_resid.shape == (batch_size, seq_len, self.config.d_model)
            assert resid.shape == output_resid.shape

            # For Q rows in attn mask that are all false, the residuals will be
            # nan. For this operation, we use a modified version of the
            # residuals that are zeroed out at those positions.
            resid_noq_zero = torch.where(
                reduce(attn_mask, "batch seq_q seq_k -> batch seq_q 1", "sum") > 0,
                resid,
                0.0,
            )
            assert torch.all(torch.isfinite(resid_noq_zero)), resid_noq_zero

            # Compute full attention weight matrix with initial output residual
            # streams as Q and final input residual streams (transformed) as K.
            attn_weights = einsum(
                output_resid, # Q
                self.input_k_transform(resid_noq_zero), # K
                "batch seq_q d_model, batch seq_k d_model -> batch seq_q seq_k"
            ) / seq_len ** 0.5 # scaling factor

            # Apply our attention mask
            attn_weights.masked_fill_(~io_interface_mask, 1e-5)

            # Only compute softmax over rows that have at least one entry in Q
            valid = repeat(reduce(io_interface_mask, "batch seq_q seq_k -> batch seq_q", "sum") > 0, "batch seq_q -> batch seq_q seq_k", seq_k=seq_len)
            attn_probs = torch.zeros_like(attn_weights, dtype=attn_weights.dtype)
            attn_probs[valid] = torch.softmax(attn_weights[valid], dim=-1).to(attn_weights.dtype)
            attn_probs[~valid] = 0.0
            attn_probs = torch.where(
                io_interface_mask,
                attn_probs,
                0.0,
            )

            assert torch.all(torch.isfinite(attn_probs))

            attn_result = einsum(
                attn_probs, # QK
                resid_noq_zero, # V
                "batch seq_q seq_k, batch seq_k d_model -> batch seq_q d_model"
            )

            assert attn_result.shape == (batch_size, seq_len, self.config.d_model)

            # The first inference group has no corresponding input for its
            # output, so we just set the initial output residual stream to the
            # output positional embedding.
            resid = output_resid + attn_result

            # Now continue with remaining blocks like a typical transformer
            for output_block in self.output_blocks:
                resid = output_block(resid, attn_mask=attn_mask)

            resid_noq_zero = torch.where(
                reduce(attn_mask, "batch seq_q seq_k -> batch seq_q 1", "sum") > 0,
                resid,
                0.0,
            )

            logits = self.unembed(resid_noq_zero)

        return logits

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
        **kwargs,
    ):
        torch.autograd.set_detect_anomaly(True)
        with torch.no_grad():
            device = input_ids.device
            batch_size = input_ids.shape[0]
            assert batch_size == batch_size_t2i + batch_size_lm + batch_size_mmu
            seq_len = input_ids.shape[1]
            assert seq_len <= self.config.max_seq_len

            # Construct inference groups and reorder idxs for each task type then combine
            # t2i and mmu have images so we need to reorder
            (reorder_idx_t2i_batch, reorder_idx_t2i_seq), inference_groups_t2i = (
                reorder_and_group_token_batch(
                    input_ids[:batch_size_t2i],
                    soi_id,
                    eoi_id,
                    self.config.image_len,
                )
            )
            (reorder_idx_mmu_batch, reorder_idx_mmu_seq), inference_groups_mmu = (
                reorder_and_group_token_batch(
                    input_ids[batch_size_t2i + batch_size_lm :],
                    soi_id,
                    eoi_id,
                    self.config.image_len,
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
            print(f"Reorder idx batch (batch 0): {reorder_idx_batch[0]}")
            print(f"First 20 reorder idx seq (batch 0): {reorder_idx_seq[0, :20]}")

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
            # attn_mask = remove_pads_from_attn_mask(attn_mask, input_ids_reordered, pad_id)
            assert attn_mask.shape == (batch_size, seq_len, seq_len)
            assert attn_mask.dtype == torch.bool

            # Get input output interface mask, which is a sparse attention mask for
            # passing residuals from last input block to first output block
            io_interface_mask = get_io_interface_mask(inference_groups)
            # io_interface_mask = remove_pads_from_attn_mask(io_interface_mask, input_ids_reordered, pad_id)
            assert io_interface_mask.shape == (batch_size, seq_len, seq_len)

            # Construct mask designating where inferences start in each batch.
            # This will exclude the initial padding tokens and the inference
            # group immediately after them.
            # Get first token after optional left padding.
            nonpad_first = (input_ids_reordered != pad_id).int().argmax(dim=1)
            # Get inference groups for each first token.
            first_inference_groups = inference_groups[torch.arange(batch_size, device=device), nonpad_first]

            # Assert that no token attends to pad token
            # assert torch.all(~((reduce(attn_mask, "batch seq_q seq_k -> batch seq_k", "sum") > 0) & (input_ids_reordered == pad_id)))

            # Assert that the only query rows in the attention mask with no entries are pad tokens
            # assert torch.all((reduce(attn_mask, "batch seq_q seq_k -> batch seq_q", "sum") > 0) | (input_ids_reordered == pad_id))

        logits = self._forward(
            input_ids_reordered, reorder_idx_seq, attn_mask, io_interface_mask, inference_groups, first_inference_groups
        )

        # Assert no nans
        assert torch.all(torch.isfinite(logits)), "Logits contain NaNs or Infs"

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
                inference_groups == first_inference_groups.unsqueeze(1), labels_reordered, ignore_id
            )
            # Reorder indexes of logits for cross entropy
            logits_rearranged = rearrange(logits, "batch seq vocab -> batch vocab seq")

            loss_t2i = F.cross_entropy(
                input=logits_rearranged[:batch_size_t2i],
                target=labels_reordered[:batch_size_t2i],
                ignore_index=ignore_id,
                label_smoothing=label_smoothing,
            )
            assert(not torch.isnan(loss_t2i) and not torch.isinf(loss_t2i))

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
            assert(not torch.isnan(loss_lm) and not torch.isinf(loss_lm))

            loss_mmu = F.cross_entropy(
                input=logits_rearranged[batch_size_t2i + batch_size_lm :],
                target=labels_reordered[batch_size_t2i + batch_size_lm :],
                ignore_index=ignore_id,
                label_smoothing=label_smoothing,
            )
            assert(not torch.isnan(loss_mmu) and not torch.isinf(loss_mmu))

            return logits, loss_t2i, loss_lm, loss_mmu

        return logits
