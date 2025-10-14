from typing import Tuple, List
import torch
from torch import Tensor
from jaxtyping import Int, Bool
import math
from einops import reduce, repeat

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

def get_input_output_interface_mask(inference_groups: Int[Tensor, "token"]) -> Bool[Tensor, "token token"]:
    groups_i = inference_groups.unsqueeze(1)
    groups_j = inference_groups.unsqueeze(0)
    return (groups_i == groups_j + 1)

def get_attn_mask(inference_groups: Int[Tensor, "batch seq"]) -> Bool[Tensor, "batch seq seq"]:
    query_groups = inference_groups.unsqueeze(2)
    key_groups = inference_groups.unsqueeze(1)
    mask = key_groups <= query_groups
    return mask
    
def reorder_and_group_token_batch(input_ids: Int[Tensor, "batch seq"], soi_id: int, eoi_id: int, num_image_tokens: int) -> Tuple[Int[Tensor, "batch seq"], Int[Tensor, "batch seq"]]:
    # Assume single image per seq
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]

    # Compute number of image tokens from first row by counting tokens between soi and eoi
    first_row = input_ids[0]
    soi_idx = (first_row == soi_id).nonzero(as_tuple=True)[0][0].item()
    eoi_idx = (first_row == eoi_id).nonzero(as_tuple=True)[0][0].item()
    assert(eoi_idx - soi_idx - 1 == num_image_tokens)

    soi_idxs = (input_ids == soi_id).long()
    assert torch.all(reduce(soi_idxs, "batch seq -> batch", "sum") == 1), "More than one soi token in a sequence"
    soi_idxs = soi_idxs.argmax(dim=1) + 1 # [B]

    img_reorder_idx, img_inference_groups = get_index_and_grouping(int(math.sqrt(num_image_tokens)))

    # Reorder image tokens using image token reorder indexes
    img_reorder_idx = repeat(img_reorder_idx, "seq -> batch seq", batch=batch_size).clone()
    img_reorder_idx_src = img_reorder_idx + soi_idxs.unsqueeze(1)
    img_idx_ori = torch.arange(num_image_tokens, device=input_ids.device).unsqueeze(0) + soi_idxs.unsqueeze(1)

    batch_indices = torch.arange(batch_size, device=input_ids.device).unsqueeze(1)
    input_ids[batch_indices, img_idx_ori] = input_ids[batch_indices, img_reorder_idx_src]

    # Create full versions of reorder indexes and insert image portion we just used
    reorder_idx = repeat(torch.arange(seq_len, device=input_ids.device), "seq -> batch seq", batch=batch_size).clone()
    reorder_idx[torch.arange(batch_size, device=input_ids.device).unsqueeze(1), img_idx_ori] = img_reorder_idx_src

    # Create inference groups for all tokens by inserting the image portion in each sequence
    inference_groups = repeat(torch.arange(seq_len, device=input_ids.device), "seq -> batch seq", batch=batch_size).clone()
    num_inference_groups_per_img = img_inference_groups[-1]
    img_inference_groups = repeat(img_inference_groups, "seq -> batch seq", batch=batch_size).clone() + soi_idxs.unsqueeze(1)
    inference_groups[torch.arange(batch_size, device=input_ids.device).unsqueeze(1), img_idx_ori] = img_inference_groups

    # Update the inference groups after the image to start after last image inference group
    post_img_sizes = seq_len - (soi_idxs + num_image_tokens)
    max_size = post_img_sizes.max()
    # Make a grid [0..max_len-1] per row, then mask by each row's length
    shared_seq = torch.arange(max_size, device=input_ids.device)
    grid = repeat(shared_seq, "len -> batch len", batch=batch_size).clone() + soi_idxs.unsqueeze(1) + num_image_tokens
    mask = shared_seq.unsqueeze(0) < post_img_sizes.unsqueeze(1)   # [num_rows, max_len] boolean
    post_img_idxs_by_row = grid[mask]

    batch_idx = torch.repeat_interleave(torch.arange(batch_size, device=input_ids.device), post_img_sizes)
    inference_groups[batch_idx, post_img_idxs_by_row] -= (num_image_tokens - num_inference_groups_per_img)

    return reorder_idx, inference_groups

def remove_pads_from_attn_mask(attn_mask: Bool[Tensor, "batch seq seq"], input_ids: Int[Tensor, "batch seq"], pad_id: int) -> Bool[Tensor, "batch seq seq"]:
    # attn_mask: [B, S, S]
    # input_ids: [B, S]
    pad_mask = (input_ids != pad_id)  # [B, S]
    attn_mask = attn_mask & pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2)
    return attn_mask

if __name__ == "__main__":
    print("Starting")

    results = create_image_token_ordering(4)

    print("Results")
    for level_results in results:
        print("Level")
        print(level_results)

    indexes, grouping = get_index_and_grouping(4)
    print("Indexes")
    print(len(indexes))
    print(indexes)
    print("Grouping")
    print(grouping)
    # print(create_image_token_ordering(16))
    print("Input-output interface mask")
    print(get_input_output_interface_mask(grouping).int())
    print("Attention mask")
    print(get_attn_mask(grouping.unsqueeze(0)).int())

    # Simple test input for reorder_and_group with image size 4
    # Image size 4 means 4x4 = 16 image tokens
    soi_id = 9999
    eoi_id = 9998
    image_size = 4
    num_image_tokens = image_size * image_size  # 16 tokens

    # Create input: [text_tokens, soi, image_tokens, eoi, text_tokens]
    # Image tokens numbered 100-115 for easy visualization
    test_input = torch.tensor([
        [
            0, 0, 3,  # prefix text tokens
            soi_id,  # start of image
            100, 101, 102, 103,  # first row of image
            104, 105, 106, 107,  # second row
            108, 109, 110, 111,  # third row
            112, 113, 114, 115,  # fourth row
            eoi_id,  # end of image
            4, 5, 6  # suffix text tokens
        ],
        [
            1, 2,  # prefix text tokens
            soi_id,  # start of image
            100, 101, 102, 103,  # first row of image
            104, 105, 106, 107,  # second row
            108, 109, 110, 111,  # third row
            112, 113, 114, 115,  # fourth row
            eoi_id,  # end of image
            3, 0, 0, 0  # suffix text tokens
        ]
    ])

    print("\nTest input tensor:")
    print(test_input)
    print(f"Shape: {test_input.shape}")
    reorder_idx, inference_groups = reorder_and_group_token_batch(test_input, soi_id, eoi_id, num_image_tokens)
    print("\nReordered input tensor:")
    print(test_input)
    print(f"Shape: {test_input.shape}")
    print("\nReorder indexes:")
    print(reorder_idx)
    print("\ninference groups:")
    print(inference_groups)
    print("\nAttention mask:")
    attn_mask = get_attn_mask(inference_groups).int()
    attn_mask = remove_pads_from_attn_mask(attn_mask, test_input, pad_id=0).int()
    for mat in attn_mask:
        print(mat)




