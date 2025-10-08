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
        attention_mask: Bool[Tensor, "batch head seq seq"],
    ) -> Float[Tensor, "batch seq d_model"]:
        x_ln1 = self.ln1(x)
        x = x + self.attn(x_ln1, x_ln1, x_ln1, attn_mask=rearrange(repeat(attention_mask, "batch 1 seq_q seq_k -> batch head seq_q seq_k", head=self.num_heads), "batch head seq_q seq_k -> (batch head) seq_q seq_k"), need_weights=False)[0]
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

    def forward(self, input_ids: Int[Tensor, "batch seq"], attention_mask: Bool[Tensor, "batch head seq seq"]=None) -> Float[Tensor, "batch seq vocab"]:
        with torch.autocast(device_type=input_ids.device.type, dtype=self.dtype):
            batch_size, seq_len = input_ids.shape

            # Init residual stream with embeddings
            resid = repeat(self.pos_embed(torch.arange(seq_len, dtype=torch.long, device=input_ids.device)), "seq d_model -> batch seq d_model", batch=batch_size)
            resid = resid + self.tok_embed(input_ids)

            assert resid.shape == (batch_size, seq_len, self.d_model)

            # Flow through transformer blocks
            for block in self.blocks:
                resid = block(resid, attention_mask)

            assert resid.shape == (batch_size, seq_len, self.d_model)

            logits = self.unembed(resid)

            assert logits.shape == (batch_size, seq_len, self.vocab_size)

            return logits
        