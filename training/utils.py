import math
import random
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Any, List, Tuple, Union


##################################################
#              config utils
##################################################
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


def flatten_omega_conf(cfg: Any, resolve: bool = False) -> List[Tuple[str, Any]]:
    ret = []

    def handle_dict(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{k1}", v1) for k1, v1 in flatten_omega_conf(value, resolve=resolve)]

    def handle_list(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{idx}", v1) for idx, v1 in flatten_omega_conf(value, resolve=resolve)]

    if isinstance(cfg, DictConfig):
        for k, v in cfg.items_ex(resolve=resolve):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(k, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(k, v, resolve=resolve))
            else:
                ret.append((str(k), v))
    elif isinstance(cfg, ListConfig):
        for idx, v in enumerate(cfg._iter_ex(resolve=resolve)):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(idx, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(idx, v, resolve=resolve))
            else:
                ret.append((str(idx), v))
    else:
        assert False

    return ret


##################################################
#              training utils
##################################################
def soft_target_cross_entropy(logits, targets, soft_targets):
    # ignore the first token from logits and targets (class id token)
    logits = logits[:, 1:]
    targets = targets[:, 1:]

    logits = logits[..., : soft_targets.shape[-1]]

    log_probs = F.log_softmax(logits, dim=-1)
    padding_mask = targets.eq(-100)

    loss = torch.sum(-soft_targets * log_probs, dim=-1)
    loss.masked_fill_(padding_mask, 0.0)

    # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
    num_active_elements = padding_mask.numel() - padding_mask.long().sum()
    loss = loss.sum() / num_active_elements
    return loss


def get_loss_weight(t, mask, min_val=0.3):
    return 1 - (1 - mask) * ((1 - t) * (1 - min_val))[:, None]


def mask_or_random_replace_tokens(unified_input_ids, unified_labels, mask_id, soi_id, eoi_id, config, mask_schedule, is_train=True, ignore_id=-100):
    """
    Apply masking to unified token sequence (after unified prompting).

    Args:
        unified_input_ids: Token sequence after unified prompting [B, L]
        unified_labels: Label sequence after unified prompting [B, L]
        mask_id: ID to use for masked tokens
        soi_id: Start of image token ID
        eoi_id: End of image token ID
        config: Training configuration
        mask_schedule: Masking schedule function
        is_train: Whether in training mode
        ignore_id: ID to use for ignored positions in labels (default: -100)

    Returns:
        masked_input_ids: Input tokens with masks applied
        masked_labels: Label tokens with masks applied
        loss_weight: Loss weights (None in this implementation)
        mask_prob: Masking probabilities used
    """
    batch_size, total_seq_len = unified_input_ids.shape
    device = unified_input_ids.device

    # Compute number of image tokens from first row by counting tokens between soi and eoi
    first_row = unified_input_ids[0]
    soi_idx = (first_row == soi_id).nonzero(as_tuple=True)[0][0].item()
    eoi_idx = (first_row == eoi_id).nonzero(as_tuple=True)[0][0].item()
    num_image_tokens = eoi_idx - soi_idx - 1  # tokens between soi and eoi (exclusive)

    # Find starting index of images in each row (vectorized)
    # Find soi token positions and increment by 1 to get first image token position
    soi_positions = (unified_input_ids == soi_id).long().argmax(dim=1)
    image_start_indices = soi_positions + 1  # [B]

    # Sample masking probabilities
    if not is_train and config.training.get("eval_mask_ratios", None):
        mask_prob = random.choices(config.training.eval_mask_ratios, k=batch_size)
        mask_prob = torch.tensor(mask_prob, device=device)
    else:
        # Sample a random timestep for each image
        timesteps = torch.rand(batch_size, device=device)
        # Sample a random mask probability for each image using timestep and cosine schedule
        mask_prob = mask_schedule(timesteps)
        mask_prob = mask_prob.clip(config.training.min_masking_rate)

    # Compute number of tokens to mask per sample
    num_token_masked = (num_image_tokens * mask_prob).round().clamp(min=1).to(torch.long)  # [B]

    assert "mask_contiguous_region_prob" not in config.training

    # Create random permutation for image tokens only
    perm = torch.rand(batch_size, num_image_tokens, device=device).argsort(dim=-1)  # [B, S]

    # Select positions where permutation value < num_token_masked (consistent with original implementation)
    mask = perm < num_token_masked.unsqueeze(-1)  # [B, S]

    # Get indices where mask is True
    row_idx, col_idx = torch.where(mask)

    # Convert image-space indices to unified-space indices by adding start offset
    unified_col_idx = col_idx + image_start_indices[row_idx]

    # Clone input_ids and labels
    masked_input_ids = unified_input_ids.clone()
    masked_labels = unified_labels.clone()

    # Apply masking using index_put_
    assert config.training["noise_type"] == "mask"
    masked_input_ids.index_put_((row_idx, unified_col_idx), torch.full_like(row_idx, int(mask_id)))

    # Create mask for all image token positions (True = ignore, False = predict) - vectorized
    positions = torch.arange(total_seq_len, device=device).unsqueeze(0)  # [1, L]
    start_indices = image_start_indices.unsqueeze(1)  # [B, 1]
    end_indices = start_indices + num_image_tokens  # [B, 1]
    image_region_mask = (positions >= start_indices) & (positions < end_indices)  # [B, L]

    # Set masked positions to False (we want to predict these)
    image_region_mask.index_put_((row_idx, unified_col_idx), torch.zeros_like(row_idx, dtype=torch.bool))

    # Apply mask: set all True positions to ignore_id
    masked_labels = torch.where(image_region_mask, ignore_id, unified_labels)

    loss_weight = None

    return masked_input_ids, masked_labels, loss_weight, mask_prob


##################################################
#              misc
##################################################
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

from torchvision import transforms
def image_transform(image, resolution=256, normalize=True):
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    if normalize:
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
    return image