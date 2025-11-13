from typing import Tuple, List, Union
import torch
from torch import Tensor
from jaxtyping import Int, Bool, Float
import math
from einops import reduce, repeat, rearrange
from torch.distributions import Geometric

Coord = Tuple[int, int]

def get_tri_quadrant_subroots(root: Coord, dim: int) -> List[Coord]:
    # Assumes that root is the top left corner of a square and that the image
    # grows down and to the right, and gets the 3 other corners of the square

    assert dim % 2 == 0, "Only dims that are powers of 2 are supported"

    return [
        (root[0], root[1] + int(dim / 2)),
        (root[0] + int(dim / 2), root[1]),
        (root[0] + int(dim / 2), root[1] + int(dim / 2)),
    ]


def get_coords(root: Coord, dim: int) -> List[List[Coord]]:
    # Returns each level of recursion as separate list

    if dim == 1:
        return []
    
    subroots = get_tri_quadrant_subroots(root, dim)

    subquadrants_coords = [
        get_coords(subroot, int(dim / 2))
        for subroot in [root] + subroots
    ]

    # Merge results at each recursion level across recursive subcalls
    assert(len(set(len(results) for results in subquadrants_coords)) == 1)

    results = [subroots] + [
        sum(
            [subquadrant_coords[level] for subquadrant_coords in subquadrants_coords],
            []
        ) for level in range(len(subquadrants_coords[0]))
    ]
    return results


def create_image_token_ordering(dim: int):
    # Assumes square image
    # Returns 2d indexes

    root = (0, 0)

    coords_by_level = get_coords(root, dim)
    return [[root]] + coords_by_level

def convert_coord_to_index(coord: Coord, dim: int) -> int:
    return coord[0] * dim + coord[1]

def get_index_and_grouping(dim: int) -> Tuple[Int[Tensor, "token"], Int[Tensor, "token"]]:
    # Get indexes for reordering and indexes for inference groups for an image of size dim x dim
    indexes = []
    grouping = []

    for group_id, level_results in enumerate(create_image_token_ordering(dim)):
        for coord in level_results:
            index = convert_coord_to_index(coord, dim)
            indexes.append(index)
            grouping.append(group_id)
            
    return torch.tensor(indexes), torch.tensor(grouping)

def get_index_and_grouping_recursive_half(dim: int) -> Tuple[Int[Tensor, "token"], Int[Tensor, "token"]]:
    indexes = []
    grouping = []

    group_id = 0

    for level_results in create_image_token_ordering(dim):
        if len(level_results) == 1:
            results_for_segments = [level_results]
        else:
            # Split in half
            results_for_segments = [
                level_results[1::2],
                level_results[::2],
            ]
        
        for results_for_segment in results_for_segments:
        
            for coord in results_for_segment:
                index = convert_coord_to_index(coord, dim)
                indexes.append(index)
                grouping.append(group_id)

            group_id += 1
            
    return torch.tensor(indexes), torch.tensor(grouping) 

def get_index_and_grouping_recursive_quarter(dim: int) -> Tuple[Int[Tensor, "token"], Int[Tensor, "token"]]:
    indexes = []
    grouping = []

    group_id = 0

    for level_results in create_image_token_ordering(dim):
        if len(level_results) < 4:
            results_for_segments = [
                [element] for element in level_results
            ]
        else:
            results_for_segments = [
                level_results[3::4],
                level_results[2::4],
                level_results[1::4],
                level_results[::4],
            ]
        
        for results_for_segment in results_for_segments:
        
            for coord in results_for_segment:
                index = convert_coord_to_index(coord, dim)
                indexes.append(index)
                grouping.append(group_id)

            group_id += 1
            
    return torch.tensor(indexes), torch.tensor(grouping) 

def get_index_and_grouping_linear(num_groups: int, dim: int) -> Tuple[Int[Tensor, "token"], Int[Tensor, "token"]]:
    num_tokens = dim ** 2
    tokens_per_group = int(num_tokens / num_groups)
    assert num_tokens % num_groups == 0

    reorder_idx = rearrange(
        torch.arange(num_groups).unsqueeze(0) + torch.arange(tokens_per_group).unsqueeze(1) * num_groups,
        "group token -> (token group)"
    )
    
    
    inference_groups = repeat(
        torch.arange(num_groups),
        "group -> (group token)",
        token=tokens_per_group
    )
    return reorder_idx, inference_groups


def get_io_interface_mask(inference_groups: Int[Tensor, "batch seq"]) -> Bool[Tensor, "batch seq seq"]:
    groups_i = inference_groups.unsqueeze(2)
    groups_j = inference_groups.unsqueeze(1)
    return (groups_i == groups_j + 1)

def get_attn_mask(inference_groups: Int[Tensor, "batch seq"]) -> Bool[Tensor, "batch seq seq"]:
    query_groups = inference_groups.unsqueeze(2)
    key_groups = inference_groups.unsqueeze(1)
    mask = key_groups <= query_groups
    return mask
    
def reorder_and_group_token_batch(input_ids: Int[Tensor, "batch seq"], soi_id: int, eoi_id: int, num_image_tokens: int, img_reorder_idx: Int[Tensor, "batch seq"], img_inference_groups: Int[Tensor, "batch seq"]) -> Tuple[Tuple[Int[Tensor, "batch 1"], Int[Tensor, "batch seq"]], Int[Tensor, "batch seq"]]:
    assert input_ids.dim() == 2
    # Assume single image per seq
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    device = input_ids.device

    # Compute number of image tokens from first row by counting tokens between soi and eoi
    first_row = input_ids[0]
    soi_idx = (first_row == soi_id).nonzero(as_tuple=True)[0][0].item()
    eoi_idx = (first_row == eoi_id).nonzero(as_tuple=True)[0][0].item()
    assert(eoi_idx - soi_idx - 1 == num_image_tokens)

    soi_idxs = (input_ids == soi_id).long()
    assert torch.all(reduce(soi_idxs, "batch seq -> batch", "sum") == 1), "More than one soi token in a sequence"
    soi_idxs = soi_idxs.argmax(dim=1) + 1 # [B]

    # Img reorder idxs describe the image indexes starting at index 0, need to offset by soi idxs for each row
    img_reorder_idx = img_reorder_idx.to(device) + soi_idxs.unsqueeze(1)
    img_idx_ori = torch.arange(num_image_tokens, device=device).unsqueeze(0) + soi_idxs.unsqueeze(1)
    # Create full versions of reorder indexes and insert image portion we just created
    reorder_idx_seq = repeat(torch.arange(seq_len, device=device), "seq -> batch seq", batch=batch_size).clone()
    reorder_idx_seq[torch.arange(batch_size, device=device).unsqueeze(1), img_idx_ori] = img_reorder_idx
    reorder_idx_batch = torch.arange(batch_size, device=device).unsqueeze(1)

    # Create inference groups for all tokens by inserting the image portion in each sequence
    img_inference_groups = img_inference_groups.to(device)
    # Get number of inference groups before adding offset
    num_inference_groups_per_img = img_inference_groups[:, -1] + 1
    # Add offset
    img_inference_groups = img_inference_groups + soi_idxs.unsqueeze(1)
    inference_groups = repeat(torch.arange(seq_len, device=device), "seq -> batch seq", batch=batch_size).clone()
    inference_groups[torch.arange(batch_size, device=device).unsqueeze(1), img_idx_ori] = img_inference_groups

    # Update the inference groups after the image to start after last image inference group
    post_img_sizes = seq_len - (soi_idxs + num_image_tokens)
    max_size = post_img_sizes.max()
    shared_seq = torch.arange(max_size, device=device)
    grid = repeat(shared_seq, "len -> batch len", batch=batch_size).clone() + soi_idxs.unsqueeze(1) + num_image_tokens
    # Create a boolean mask to select the number of tokens after the image for
    # each row
    mask = shared_seq.unsqueeze(0) < post_img_sizes.unsqueeze(1)   # [num_rows, max_len] boolean
    post_img_idxs_by_row = grid[mask]

    batch_idx = torch.repeat_interleave(torch.arange(batch_size, device=device), post_img_sizes)
    inference_groups[batch_idx, post_img_idxs_by_row] -= (
        num_image_tokens - torch.repeat_interleave(num_inference_groups_per_img, post_img_sizes)
    )

    return (reorder_idx_batch, reorder_idx_seq), inference_groups

def get_sigma_reorder_and_grouping(batch_size: int, dim: int, device: torch.device) -> Tuple[Int[Tensor, "batch seq"], Int[Tensor, "batch seq"]]:
    # Random permutations of image tokens with inference groups of size 1
    image_len = dim ** 2
    # Separate random permutation for each row
    random_vals = torch.rand(batch_size, image_len, device=device)
    reorder_idx = torch.argsort(random_vals, dim=1)
    inference_groups = repeat(
        torch.arange(image_len, device=device),
        "seq -> batch seq",
        batch=batch_size
    ).clone()
    return reorder_idx, inference_groups

def get_geo_reorder_and_grouping(batch_size: int, dim: int, prob: float, device: torch.device) -> Tuple[Int[Tensor, "batch seq"], Int[Tensor, "batch seq"]]:
    # Random permutations of image tokens with inference groups drawn from geometric distribution
    image_len = dim ** 2
    # Separate random permutation for each row
    random_vals = torch.rand(batch_size, image_len, device=device)
    reorder_idx = torch.argsort(random_vals, dim=1)

    geo_dist = Geometric(prob)
    inference_groups = torch.zeros(batch_size, image_len, dtype=torch.long, device=device)
    for i in range(batch_size):
        inference_group = 0
        token_count = 0
        while token_count < image_len:
            inference_group_size = geo_dist.sample().long().clamp(min=1)
            for _ in range(inference_group_size.item()):
                if token_count >= image_len:
                    break
                inference_groups[i, token_count] = inference_group
                token_count += 1
            inference_group += 1
            if token_count >= image_len:
                break

    return reorder_idx, inference_groups

def get_vanilla_reorder_and_grouping(batch_size: int, dim: int, device: torch.device) -> Tuple[Int[Tensor, "batch seq"], Int[Tensor, "batch seq"]]:
    # No reordering, size-1 inference groups
    image_len = dim ** 2
    reorder_idx = repeat(
        torch.arange(image_len, device=device),
        "seq -> batch seq",
        batch=batch_size
    )
    inference_groups = reorder_idx.clone()
    return reorder_idx, inference_groups

def remove_pads_from_attn_mask(attn_mask: Bool[Tensor, "batch seq seq"], input_ids: Int[Tensor, "batch seq"], pad_id: int) -> Bool[Tensor, "batch seq seq"]:
    # attn_mask: [B, S, S]
    # input_ids: [B, S]
    pad_mask = (input_ids != pad_id)  # [B, S]
    attn_mask = attn_mask & pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2)
    return attn_mask


def reset_to_ori_order(predictions: Union[Float[Tensor, "batch seq vocab"], Int[Tensor, "batch seq"]], reorder_idx_batch: Int[Tensor, "batch 1"], reorder_idx_seq: Int[Tensor, "batch seq"]) -> Union[Float[Tensor, "batch seq vocab"], Int[Tensor, "batch seq"]]:
    # Operates on both logits and token ids
    result = torch.zeros_like(predictions)
    result[reorder_idx_batch, reorder_idx_seq] = predictions
    return result

if __name__ == "__main__":
    print("Starting")

    results = create_image_token_ordering(4)

    print("Results")
    for level_results in results:
        print("Level")
        print(level_results)

    indexes, grouping = get_index_and_grouping(8)
    print("== Recursive ==")
    print("Indexes")
    print(len(indexes))
    print(indexes)
    print("Grouping")
    print(grouping)
    print(grouping.unique())
    print("Input-output interface mask")
    print(get_io_interface_mask(grouping.unsqueeze(0)).int())
    print("Attention mask")
    print(get_attn_mask(grouping.unsqueeze(0)).int())

    indexes, grouping = get_index_and_grouping_recursive_quarter(4)
    print("== Recursive quarter ==")
    print("Indexes")
    print(len(indexes))
    print(indexes)
    print("Grouping")
    print(grouping)
    print(grouping.unique())
    print("Input-output interface mask")
    print(get_io_interface_mask(grouping.unsqueeze(0)).int())
    print("Attention mask")
    print(get_attn_mask(grouping.unsqueeze(0)).int())


    for val in grouping.unique():
        mask = grouping == val
        print(len(indexes[mask]))

    # indexes, grouping = get_index_and_grouping_linear(8, 4)
    # print("== Linear ==")
    # print("Indexes")
    # print(len(indexes))
    # print(indexes)
    # print("Grouping")
    # print(grouping)
    # print(grouping.unique())
    # print("Input-output interface mask")
    # print(get_io_interface_mask(grouping.unsqueeze(0)).int())
    # print("Attention mask")
    # print(get_attn_mask(grouping.unsqueeze(0)).int())


    # Simple test input for reorder_and_group with image size 4
    # Image size 4 means 4x4 = 16 image tokens
    soi_id = 9999
    eoi_id = 9998
    image_size = 4
    num_image_tokens = image_size * image_size  # 16 tokens

    # Create input: [text_tokens, soi, image_tokens, eoi, text_tokens]
    # Image tokens numbered 100-115 for easy visualization
    batch_size = 2
    test_input = torch.tensor([
        [
            0, 0, 3,
            soi_id,  # start of image
            100, 101, 102, 103,  # first row of image
            104, 105, 106, 107,  # second row
            108, 109, 110, 111,  # third row
            112, 113, 114, 115,  # fourth row
            eoi_id,  # end of image
            4, 5, 6 
        ],
        [
            1, 2,
            soi_id,  # start of image
            100, 101, 102, 103,  # first row of image
            104, 105, 106, 107,  # second row
            108, 109, 110, 111,  # third row
            112, 113, 114, 115,  # fourth row
            eoi_id,  # end of image
            3, 0, 0, 0
        ]
    ])

    print("\nTest input tensor:")
    print(test_input)
    print(f"Shape: {test_input.shape}")
    reorder_idx_seq, inference_groups = get_index_and_grouping_recursive_quarter(4)
    reorder_idx_seq = repeat(reorder_idx_seq, "seq -> batch seq", batch=batch_size)
    inference_groups = repeat(inference_groups, "seq -> batch seq", batch=batch_size)
    print(reorder_idx_seq.shape)
    print(inference_groups.shape)
    (reorder_idx_batch, reorder_idx_seq), inference_groups = reorder_and_group_token_batch(test_input, soi_id, eoi_id, num_image_tokens, reorder_idx_seq, inference_groups)
    test_input_reordered = test_input[reorder_idx_batch, reorder_idx_seq]

    print("\nReordered input tensor:")
    print(test_input_reordered)
    test_input_ori_order = reset_to_ori_order(test_input_reordered, reorder_idx_batch, reorder_idx_seq)

    print("\nOriginal order")
    print(test_input_ori_order)
    print(f"Shape: {test_input_reordered.shape}")

    print("\nReorder indexes:")
    print(reorder_idx_batch)
    print(reorder_idx_seq)

    print("\ninference groups:")
    print(inference_groups)

    print("\nAttention mask:")
    attn_mask = get_attn_mask(inference_groups).int()
    pad_id = 0
    attn_mask = remove_pads_from_attn_mask(attn_mask, test_input_reordered, pad_id=pad_id).int()
    for mat in attn_mask:
        print(mat)

    print("Io interface mask")
    io_interface_mask = get_io_interface_mask(inference_groups).int()
    for mat in io_interface_mask:
        print(mat)

    nonpad_first = (test_input_reordered != pad_id).int().argmax(dim=1)
    # Get inference groups for each first token.
    first_inference_groups = inference_groups[torch.arange(batch_size), nonpad_first]
    print(first_inference_groups)


    sigma_reorder_idx, sigma_inference_groups = get_sigma_reorder_and_grouping(batch_size, image_size, device=test_input.device)
    print("Sigma reorder idx")
    print(sigma_reorder_idx.shape)
    print(sigma_reorder_idx)
    print("Sigma inference groups")
    print(sigma_inference_groups.shape)
    print(sigma_inference_groups)


    geo_reorder_idx, geo_inference_groups = get_geo_reorder_and_grouping(batch_size, image_size, 0.4, device=test_input.device)
    print("Geo reorder idx")
    print(geo_reorder_idx.shape)
    print(geo_reorder_idx)
    print("Geo inference groups")
    print(geo_inference_groups.shape)
    print(geo_inference_groups)