import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum, rearrange
from jaxtyping import Float, Int


class Transformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

        self.W = nn.Parameter(torch.randn(vocab_size, dtype=torch.bfloat16))

    def forward(self, input_ids: Int[Tensor, "batch seq"], attention_mask=None):
        logits = einsum(input_ids, self.W, "batch seq, vocab -> batch seq vocab")
        return {'logits': logits}