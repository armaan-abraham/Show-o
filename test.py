import torch

input_ids = torch.load("input_ids.pt", map_location='cpu')
input_ids_reordered = torch.load("input_ids_reordered.pt", map_location='cpu')
labels = torch.load("labels.pt", map_location='cpu')
labels_reordered = torch.load("labels_reordered.pt", map_location='cpu')
reorder_idx_seq = torch.load("reorder_idx_seq.pt", map_location='cpu')
io_interface_mask = torch.load("io_interface_mask.pt", map_location='cpu')
inference_groups = torch.load("inference_groups.pt", map_location='cpu')
attn_mask = torch.load("attn_mask.pt", map_location='cpu')

print("=== t2i ===")
num_toks = 30

print("inputs")
print(input_ids[0, :num_toks])

print("labels")
print(labels[0, :num_toks])

print("inputs reordered")
print(input_ids_reordered[0, :num_toks])

print("labels reordered")
print(labels_reordered[0, :num_toks])

print("reorder idx")
print(reorder_idx_seq[0, :num_toks])

print("attn mask")
print(attn_mask[0, :num_toks, :num_toks].int())

print("io interface")
print(io_interface_mask[0, :num_toks, :num_toks].int())

print("inference_groups")
print(inference_groups[0, :num_toks])