import torch
from transformers import AutoTokenizer
from einops import repeat

"""
 {'<|soi|>': tensor([50296]), '<|eoi|>': tensor([50297]), '<|sov|>':
 tensor([50298]), '<|eov|>': tensor([50299]), '<|t2i|>': tensor([50300]),
 '<|mmu|>': tensor([50301]), '<|t2v|>': tensor([50302]), '<|v2v|>':
 tensor([50303]), '<|lvg|>': tensor([50304]), '<|sot|>': tensor([50256]),
 '<|eot|>': tensor([50256]), '<|pad|>': tensor([50295])}
 """

logits = torch.load("logits.pt", map_location='cpu')
labels = torch.load("labels.pt", map_location='cpu')
input_ids = torch.load("input_ids.pt", map_location='cpu')
attn_mask = torch.load("attn_mask.pt", map_location='cpu')

# Load the tokenizer from the config (microsoft/phi-1_5)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", padding_side="right")

print(input_ids[:10, :10])

soi_id = 50296
image_len = 256
image_start_idx = (torch.argmax((input_ids == soi_id).int(), dim=1) + 1).unsqueeze(1)
image_tokens_in_input = torch.arange(image_len)
image_tokens_in_input = torch.arange(input_ids.shape[1]).unsqueeze(0)
image_tokens_in_input = (image_tokens_in_input >= image_start_idx) & (image_tokens_in_input < image_start_idx + 256)

def get_image_pos(ids, soi_id, image_len):
    """ get idxs for images in batch of tokens """
    batch_size = ids.shape[0]
    seq_idx = repeat(torch.arange(image_len, device=ids.device), "seq -> batch seq", batch=batch_size).clone()
    seq_idx += (torch.argmax((ids == soi_id).int(), dim=1) + 1).unsqueeze(1)
    assert seq_idx.shape == (batch_size, image_len)
    batch_idx = repeat(torch.arange(batch_size, device=ids.device), "batch -> batch seq", seq=image_len)
    return batch_idx, seq_idx

result = get_image_pos(input_ids, soi_id, image_len)

print(result[0], result[1])

print(input_ids[result][:10, :10])

exit()
print(labels[:10, :10])
print((labels[:10] != -100).int()[:, :10])
print(torch.argmax((labels[:10] != -100).int(), dim=1))
print(labels[:10][:, torch.argmax((labels[:10] != -100).int(), dim=1)])

for i in range(input_ids.shape[0]):
    print(input_ids[i, :10])
    print(labels[i, :10])
    print(tokenizer.decode(input_ids[i, :10]))
    print()