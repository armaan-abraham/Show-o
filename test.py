import torch

input_ids = torch.load("input_ids.pt", map_location='cpu')
reorder_idx_seq = torch.load("reorder_idx_seq.pt", map_location='cpu')
attn_mask = torch.load("attn_mask.pt", map_location='cpu')
io_interface_mask = torch.load("io_interface_mask.pt", map_location='cpu')
inference_groups = torch.load("inference_groups.pt", map_location='cpu')
first_inference_groups = torch.load("first_inference_groups.pt", map_location='cpu')
resid_before_attn = torch.load("resid_before_attn.pt", map_location='cpu')

for tok in resid_before_attn[0]:
    print(tok)