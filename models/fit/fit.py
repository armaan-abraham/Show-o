import torch
import torch.nn.functional as F
from ..modeling_utils import ConfigMixin, ModelMixin, register_to_config
from ..base import TransformerBlock
from pathlib import Path
from .utils import get_index_and_grouping, get_input_output_interface_mask
from einops import repeat


# Flexible Inference Transformer (FIT)
class FIT(ModelMixin, ConfigMixin):
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
    

    def reorder_and_group(self, input_ids, soi_id, eoi_id):
        # Assume single image per seq
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # Compute number of image tokens from first row by counting tokens between soi and eoi
        first_row = input_ids[0]
        soi_idx = (first_row == soi_id).nonzero(as_tuple=True)[0][0].item()
        eoi_idx = (first_row == eoi_id).nonzero(as_tuple=True)[0][0].item()
        num_image_tokens = eoi_idx - soi_idx - 1  # tokens between soi and eoi (exclusive)
        assert(num_image_tokens == 256)

        soi_idxs = (input_ids == soi_id).long()
        assert torch.all(einops.reduce(soi_idxs, "batch seq -> batch", "sum") == 1), "More than one soi token in a sequence"
        soi_idxs = soi_idxs.argmax(dim=1) + 1 # [B]

        img_reorder_idx, img_inference_groups = get_index_and_grouping(16)
        img_reorder_idx = repeat(img_reorder_idx, "seq -> batch seq", batch=batch_size)

        # Reorder image tokens using image token reorder indexes
        img_reorder_idx_src = img_reorder_idx + soi_idxs.unsqueeze(1)
        img_reorder_idx_dest = torch.arange(num_image_tokens, device=input_ids.device).unsqueeze(0) + soi_idxs.unsqueeze(1)

        batch_indices = torch.arange(batch_size, device=input_ids.device).unsqueeze(1)
        input_ids[batch_indices, img_reorder_idx_dest] = input_ids[batch_indices, img_reorder_idx_src]

        # Create full versions of reorder indexes and insert image portion we just used
        reorder_idx = repeat(torch.arange(seq_len, device=input_ids.device), "seq -> batch seq", batch=batch_size)
        reorder_idx[torch.arange(batch_size, device=input_ids.device), soi_idxs:soi_idxs + num_image_tokens] = img_reorder_idx_src

        # Create inference groups for all tokens by inserting the image portion in each sequence
        inference_groups = repeat(torch.arange(seq_len, device=input_ids.device), "seq -> batch seq", batch=batch_size)
        num_inference_groups_per_img = img_inference_groups[-1]
        img_inference_groups = repeat(img_inference_groups, "seq -> batch seq", batch=batch_size) + soi_idxs.unsqueeze(1)
        inference_groups[torch.arange(batch_size, device=input_ids.device), soi_idxs:soi_idxs + num_image_tokens] = img_inference_groups
        # Update the inference groups after the image to start after last image inference group
        inference_groups[torch.arange(batch_size, device=input_ids.device), soi_idxs + num_image_tokens:] -= (num_image_tokens - num_inference_groups_per_img)

        return reorder_idx, inference_groups



    def forward(
            self,
            input_ids,
            labels=None,
            batch_size_t2i=0,
            batch_size_lm=0,
            batch_size_mmu=0,
            max_seq_length=128,
            global_step=None,
            pad_id=None,
            soi_id=None,
            eoi_id=None,
            config=None,
            **kwargs,
    ):


        if labels is not None:
            assert torch.all(input_ids == labels)





            return logits, loss_t2i, loss_lm, loss_mmu, None

        return logits
