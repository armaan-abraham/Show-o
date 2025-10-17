import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum, rearrange, repeat
from jaxtyping import Float, Int, Bool

dtype_str_to_torch = {
    "bf16": torch.bfloat16,
}

class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_mlp: int,
            num_heads: int,
    ):
        super().__init__()
        self.num_heads = num_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )

    def forward(
        self,
        x: Float[Tensor, "batch seq d_model"],
        attn_mask: Bool[Tensor, "batch seq seq"],
    ) -> Float[Tensor, "batch seq d_model"]:
        """
        True in the attention mask means attend, False means don't attend.
        """
        x_ln1 = self.ln1(x)
        # Multi head attention expects true to mean don't attend, false to mean
        # attend, so we invert the mask here.
        x = x + self.attn(x_ln1, x_ln1, x_ln1, attn_mask=rearrange(repeat(~attn_mask, "batch seq_q seq_k -> batch head seq_q seq_k", head=self.num_heads), "batch head seq_q seq_k -> (batch head) seq_q seq_k"), need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(
            self, 
            vocab_size: int,
            max_seq_len: int,
            num_layers: int,
            d_model: int,
            d_mlp: int,
            num_heads: int,
            dtype: str,
        ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.dtype = dtype_str_to_torch[dtype]
        
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model=d_model, d_mlp=d_mlp, num_heads=num_heads) for _ in range(num_layers)])
        self.unembed = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: Int[Tensor, "batch seq"], attn_mask: Bool[Tensor, "batch seq seq"]=None) -> Float[Tensor, "batch seq vocab"]:
        """
        True in the attention mask means attend, False means don't attend.
        """
        with torch.autocast(device_type=input_ids.device.type, dtype=self.dtype):
            batch_size, seq_len = input_ids.shape

            # Init residual stream with embeddings
            resid = repeat(self.pos_embed(torch.arange(seq_len, dtype=torch.long, device=input_ids.device)), "seq d_model -> batch seq d_model", batch=batch_size)
            resid = resid + self.tok_embed(input_ids)

            assert resid.shape == (batch_size, seq_len, self.d_model)

            # Flow through transformer blocks
            for block in self.blocks:
                resid = block(resid, attn_mask)

            assert resid.shape == (batch_size, seq_len, self.d_model)

            logits = self.unembed(resid)

            assert logits.shape == (batch_size, seq_len, self.vocab_size)

            return logits


class TransformerForShowo(ModelMixin, ConfigMixin):
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
    ):
        super().__init__()
        num_layers = num_layers_input + num_layers_output
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_mlp,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, self.num_layers)
        self.output = nn.Linear(d_model, vocab_size)

    def generate_causal_mask(self, sz):
        # Create causal mask (upper triangular = True means masked)
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def _forward(self, x):
        seq_len = x.size(1)
        
        # Embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(positions)
        
        # Causal mask
        mask = self.generate_causal_mask(seq_len).to(x.device)
        
        # Transformer
        x = self.transformer(x, mask=mask, is_causal=True)
        
        # Output projection
        logits = self.output(x)
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
        keep_prediction_order: bool = False,
        **kwargs,
    ):
        logits = self._forward(input_ids)

        result = [logits]

        if labels is not None:
            loss_t2i = F.cross_entropy(
                input=logits[:batch_size_t2i, :-1],
                target=labels[:batch_size_t2i, 1:],
                ignore_index=ignore_id,
                label_smoothing=label_smoothing,
            )

            loss_lm = F.cross_entropy(
                input=logits_rearranged[
                    batch_size_t2i : batch_size_t2i + batch_size_lm, :-1
                ],
                target=labels_reordered[
                    batch_size_t2i : batch_size_t2i + batch_size_lm, 1:
                ],
                ignore_index=ignore_id,
                label_smoothing=label_smoothing,
            )


            loss_mmu = F.cross_entropy(
                input=logits_rearranged[batch_size_t2i + batch_size_lm :, :-1],
                target=labels_reordered[batch_size_t2i + batch_size_lm :, 1:],
                ignore_index=ignore_id,
                label_smoothing=label_smoothing,
            ) 
            
            result += [loss_t2i, loss_lm, loss_mmu]
        
        return tuple(result)