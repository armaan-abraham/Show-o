import torch

def compare_tensors(file1, file2, name):
    """Compare two tensor files and report differences."""
    tensor1 = torch.load(file1)
    tensor2 = torch.load(file2)

    if tensor1.shape != tensor2.shape:
        print(f"{name}: Shapes differ - {tensor1.shape} vs {tensor2.shape}")
        return

    equal = torch.equal(tensor1, tensor2)

    if equal:
        print(f"{name}: All elements are equal")
    else:
        unequal_count = (tensor1 != tensor2).sum().item()
        total_elements = tensor1.numel()
        print(f"{name}: {unequal_count} / {total_elements} elements are unequal")

# Compare input_ids files
compare_tensors("input_ids1.pt", "models/input_ids.pt", "input_ids")

# Compare attention_mask files
compare_tensors("attention_mask1.pt", "models/attention_mask.pt", "attention_mask")
