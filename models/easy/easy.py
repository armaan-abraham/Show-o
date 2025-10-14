import torch
from torch import Tensor
import torch.nn.functional as F
from ..modeling_utils import ConfigMixin, ModelMixin, register_to_config
from ..base import TransformerBlock
from pathlib import Path
from .utils import get_index_and_grouping, get_input_output_interface_mask, reorder_and_group_token_batch, remove_pads_from_attn_mask
from einops import repeat
from jaxtyping import Float, Int


class EasyTransformer(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            vocab_size: int,
            max_seq_len: int,
            num_layers_input: int,
            num_layers_output: int,
            d_model: int,
            d_mlp: int,
            num_heads: int,
            dtype: str,
            accelerator,
    ):
        super().__init__()
        self.accelerator = accelerator
        self.num_heads = num_heads

        self.input_pos_embed = nn.Embedding(max_seq_len, d_model)
        self.output_pos_embed = nn.Embedding(max_seq_len, d_model)

        self.tok_embed = nn.Embedding(vocab_size, d_model)

        self.input_blocks = nn.ModuleList([TransformerBlock(d_model=d_model, d_mlp=d_mlp, num_heads=num_heads) for _ in range(num_layers_input)])
        self.output_blocks = nn.ModuleList([TransformerBlock(d_model=d_model, d_mlp=d_mlp, num_heads=num_heads) for _ in range(num_layers_output)])
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
    ) -> Float[Tensor, "batch seq vocab"]:
        resid = self.tok_embed(input_ids_reordered) + self.input_pos_embed(reorder_idx_seq)
        attn_mask_expand = repeat(attn_mask, "batch seq seq -> batch num_heads seq seq", num_heads=self.num_heads)

        for input_block in self.input_blocks:
            resid = input_block(resid, attn_mask_expand)
        
        # Now that we have the final residual stream for the input block at each
        # position, we need to transfer the residual stream information from
        # each inference group of input blocks to the next inference group of
        # output blocks via attention. We do this by initializing the full seq x
        # seq attention mask, which is a wasteful but temporary solution.

        output_embed = self.output_pos_embed(reorder_idx_seq)
        resid = output_embed + F.scaled_dot_product_attention(
            output_embed, # Q
            self.input_k_transform(resid), # K
            resid, # V
            attn_mask=io_interface_mask,
        )
         
        # Now continue with remaining blocks like a typical transformer
        for output_block in self.output_blocks:
            resid = output_block(resid, attn_mask_expand)
        
        return self.unembed(resid)

    def forward(
        self,
        input_ids: Tensor[Int, "batch seq"],
        batch_size_t2i: int,
        batch_size_lm: int,
        batch_size_mmu: int,
        pad_id: int,
        soi_id: int,
        eoi_id: int,
        ignore_id: int,
        labels: Tensor[Int, "batch seq"] = None,
        global_step: int = None,
    ): 
        with torch.no_grad():
            batch_size = input_ids.shape[0]
            assert(batch_size == batch_size_t2i + batch_size_lm + batch_size_mmu)
            seq_len = input_ids.shape[1]
            assert seq_len <= self.config.max_seq_len

            # Construct inference groups and reorder idxs for each task type then combine
            # t2i and mmu have images so we need to reorder
            (reorder_idx_t2i_batch, reorder_idx_t2i_seq), inference_groups_t2i = reorder_and_group_token_batch(input_ids[:batch_size_t2i], soi_id, eoi_id, config.num_image_tokens)
            (reorder_idx_mmu_batch, reorder_idx_t2i_seq), inference_groups_mmu = reorder_and_group_token_batch(input_ids[batch_size_t2i + batch_size_lm:], soi_id, eoi_id, config.num_image_tokens)
            reorder_idx_lm_seq = repeat(torch.arange(seq_len, device=input_ids.device), "seq -> batch seq", batch=batch_size_lm).clone()
            reorder_idx_lm_batch = torch.arange(batch_size_lm, device=input_ids.device).unsqueeze(1)
            inference_groups_lm = reorder_idx_lm_seq.clone()

            # Combine reorder idx
            reorder_idx_batch = torch.cat([reorder_idx_t2i_batch, reorder_idx_lm_batch + batch_size_t2i, reorder_idx_mmu_batch + batch_size_t2i + batch_size_lm])
            reorder_idx_seq = torch.cat([reorder_idx_t2i_seq, reorder_idx_lm_seq, reorder_idx_t2i_seq], dim=0)
            reorder_idx_batch = torch.cat([reorder_idx_t2i_batch, reorder_idx_lm_batch + batch_size_t2i, reorder_idx_mmu_batch + batch_size_t2i + batch_size_lm])
            assert reorder_idx_batch.shape == (batch_size, 1)
            assert reorder_idx_seq.shape == (batch_size, seq_len)
            assert torch.unique(reorder_idx_batch).shape == (batch_size,)
            assert reorder_idx_batch[0][0] == 0

            # Combine inference groups
            inference_groups = torch.cat([inference_groups_t2i, inference_groups_lm, inference_groups_mmu], dim=0)
            assert inference_groups.shape == (batch_size, seq_len)
            assert torch.all(inference_groups[:, 0] == 0)
            # Currently, we shouldn't have any first inference groups larger
            # than one token.
            assert torch.all((inference_groups == 0).sum(dim=1) == 1)

            # Reorder input ids
            input_ids_reordered = input_ids[reorder_idx_batch, reorder_idx_seq]

            # Construct attention mask from inference groups
            attn_mask = get_attn_mask(inference_groups)
            attn_mask = remove_pads_from_attn_mask(attn_mask, input_ids, pad_id)
            assert attn_mask.shape == (batch_size, seq_len, seq_len)
            assert attn_mask.dtype == torch.bool

            # Get input output interface mask, which is a sparse attention mask for
            # passing residuals from last input block to first output block
            io_interface_mask = get_input_output_interface_mask(inference_groups)
            assert io_interface_mask.shape == (batch_size, seq_len, seq_len)
        
        logits = self._forward(input_ids_reordered, reorder_idx_seq, attn_mask, io_interface_mask)

        if labels is not None:
            assert labels.shape == input_ids.shape
            # There should be no ignore token at the beginning of any sequence
            assert torch.all(labels[:, 0] != ignore_id)

            # Apply same reordering to labels
            labels_reordered = labels[reorder_idx_batch, reorder_idx_seq]

            # Each output is at the same position as its label, but the first
            # inference group should not be predicted, so we can replace those
            # with ignore and then shove the whole thing into cross entropy
            # without any shifts.
            labels_reordered = torch.where(inference_groups == 0, labels_reordered, ignore_id)
            # Reorder indexes of logits for cross entropy
            logits_rearranged = rearrange(logits, "batch seq vocab -> batch vocab seq")

            loss_t2i = F.cross_entropy(
                input=logits_rearranged[: batch_size_t2i], 
                target=labels_reordered[: batch_size_t2i],
                ignore_index=ignore_id,
            )

            loss_lm = F.cross_entropy(
                input=logits_rearranged[batch_size_t2i : batch_size_t2i + batch_size_lm], 
                target=labels_reordered[batch_size_t2i : batch_size_t2i + batch_size_lm],
                ignore_index=ignore_id,
            )

            loss_mmu = F.cross_entropy(
                input=logits_rearranged[batch_size_t2i + batch_size_lm :], 
                target=labels_reordered[batch_size_t2i + batch_size_lm :],
                ignore_index=ignore_id,
            )

            return logits, loss_t2i, loss_lm, loss_mmu

        return logits
