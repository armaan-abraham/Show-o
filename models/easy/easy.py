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

        # Parameters
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
        batch_size = input_ids.shape[0]
        assert(batch_size == batch_size_t2i + batch_size_lm + batch_size_mmu)
        seq_len = input_ids.shape[1]
        assert seq_len <= self.config.max_seq_len

        # t2i and mmu have images so we need to reorder
        reorder_idx_t2i, inference_groups_t2i = reorder_and_group_token_batch(input_ids[:batch_size_t2i], soi_id, eoi_id, config.num_image_tokens)
        reorder_idx_mmu, inference_groups_mmu = reorder_and_group_token_batch(input_ids[batch_size_t2i + batch_size_lm:], soi_id, eoi_id, config.num_image_tokens)
        reorder_idx_lm = repeat(torch.arange(seq_len), "seq -> batch seq", batch=batch_size_lm).clone()
        inference_groups_lm = reorder_idx_lm.clone()

        inference_groups = torch.cat([inference_groups_t2i, inference_groups_lm, inference_groups_mmu], dim=0)
        assert inference_groups.shape == (batch_size, seq_len)

        reorder_idx = torch.cat([reorder_idx_t2i, reorder_idx_lm, reorder_idx_mmu], dim=0)
        assert reorder_idx.shape == (batch_size, seq_len)

        attn_mask = get_attn_mask(inference_groups)
        assert attn_mask.shape == (batch_size, seq_len, seq_len)
        assert attn_mask.dtype == torch.bool

        attn_mask = remove_pads_from_attn_mask(attn_mask, input_ids, pad_id)

        if labels is not None:
            assert torch.all(input_ids == labels)





            return logits, loss_t2i, loss_lm, loss_mmu, None

        return logits
