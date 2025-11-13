import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# Disable CUDA to force CPU-only execution
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from lightning.pytorch.utilities import CombinedLoader

from models import MAGVITv2
from models.easy import EasyTransformer
from training.prompting_utils import UniversalPrompting
from training.data import Text2ImageDataset
from training.imagenet_dataset import ImageNetDataset
from training.c4_dataset import C4Dataset
from torch.utils.data import DataLoader

def main():
    # Load config
    config = OmegaConf.load("runs/easy/config.yaml")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.tokenize.text_tokenizer, padding_side="left")

    # Initialize universal prompting
    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=(
            "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
            "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
        ),
        ignore_id=-100
    )

    # Load VQ model
    vq_model = MAGVITv2.from_pretrained(config.model.vq_model.vq_model_name)
    vq_model.eval()
    vq_model.requires_grad_(False)

    # Load the model
    model = EasyTransformer(
        vocab_size=config.model.tokenize.vocab_size,
        **config.model.core
    )

    # Load checkpoint to CPU
    checkpoint_path = "runs/easy/checkpoint-20000/unwrapped_model/pytorch_model.bin"
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Set up dataloaders
    num_dataloader_workers = 2

    # T2I dataset
    dataset_imagenet = ImageNetDataset(
        config.dataset.params.train_t2i_shards_path_or_url,
        image_size=config.dataset.preprocessing.resolution,
    )
    train_dataloader_t2i = DataLoader(
        dataset_imagenet,
        batch_size=config.training.batch_size_t2i,
        collate_fn=dataset_imagenet.collate_fn,
        shuffle=False,
        num_workers=num_dataloader_workers,
        prefetch_factor=2
    )

    # MMU dataset
    dataset_mmu = Text2ImageDataset(
        train_shards_path_or_url=config.dataset.params.train_mmu_shards_path_or_url,
        tokenizer=None,
        max_seq_length=config.dataset.preprocessing.max_seq_length,
        per_gpu_batch_size=config.training.batch_size_mmu,
        num_workers=num_dataloader_workers,
        seed=config.training.seed,
        resolution=config.dataset.preprocessing.resolution,
        shuffle_buffer_size=config.dataset.params.shuffle_buffer_size,
        pin_memory=False,
        persistent_workers=False,
        is_captioning=True,
        add_caption_prompt=config.dataset.params.add_caption_prompt,
    )
    train_dataloader_mmu = dataset_mmu.train_dataloader

    # LM dataset
    dataset_lm = C4Dataset(config.dataset.params.train_lm_shards_path_or_url)
    train_dataloader_lm = DataLoader(
        dataset_lm,
        batch_size=config.training.batch_size_lm,
        collate_fn=dataset_lm.collate_fn,
        num_workers=num_dataloader_workers
    )

    # Combine dataloaders
    iterables = {
        "t2i_flow": train_dataloader_t2i,
        "lm_flow": train_dataloader_lm,
        "mmu_flow": train_dataloader_mmu,
    }
    combined_dataloader = CombinedLoader(iterables, mode=config.dataset.combined_loader_mode)

    # Get a single batch
    batch, batch_idx, dataloader_idx = next(iter(combined_dataloader))

    # Get batch sizes
    batch_size_t2i = batch["t2i_flow"]["images"].shape[0]
    batch_size_lm = len(batch["lm_flow"]["input_ids"])
    batch_size_mmu = batch["mmu_flow"]["images"].shape[0]

    # Get special token IDs
    pad_id = int(uni_prompting.sptids_dict['<|pad|>'])
    soi_id = int(uni_prompting.sptids_dict['<|soi|>'])
    eoi_id = int(uni_prompting.sptids_dict['<|eoi|>'])

    with torch.no_grad():
        # Process T2I data
        pixel_values, texts = batch["t2i_flow"]["images"], batch["t2i_flow"]["input_ids"]
        image_tokens = vq_model.get_code(pixel_values)
        image_tokens = image_tokens + len(uni_prompting.text_tokenizer)
        input_ids, masks, labels = uni_prompting(
            (texts, image_tokens, image_tokens),
            't2i',
            ignore_prefix_tokens=False
        )

        # Process LM data
        texts_lm = batch["lm_flow"]["input_ids"]
        input_ids_lm, _, labels_lm = uni_prompting(
            (texts_lm, input_ids.shape[-1]),
            'lm',
            ignore_prefix_tokens=False
        )
        input_ids = torch.cat((input_ids, input_ids_lm.to(input_ids.device)), dim=0)
        labels = torch.cat((labels, labels_lm.to(input_ids.device)), dim=0)

        # Process MMU data
        pixel_values_mmu, texts_mmu = batch["mmu_flow"]["images"], batch["mmu_flow"]["input_ids"]
        image_tokens_mmu = vq_model.get_code(pixel_values_mmu)
        image_tokens_mmu = image_tokens_mmu + len(uni_prompting.text_tokenizer)
        input_ids_mmu, _, labels_mmu = uni_prompting(
            (image_tokens_mmu, texts_mmu),
            'mmu',
            ignore_prefix_tokens=False
        )
        input_ids = torch.cat((input_ids, input_ids_mmu.to(input_ids.device)), dim=0)
        labels = torch.cat((labels, labels_mmu.to(input_ids.device)), dim=0)

        # Run forward pass
        logits, loss_t2i, loss_lm, loss_mmu = model(
            input_ids=input_ids,
            labels=labels,
            label_smoothing=config.training.label_smoothing,
            batch_size_t2i=batch_size_t2i,
            batch_size_lm=batch_size_lm,
            batch_size_mmu=batch_size_mmu,
            max_seq_length=config.dataset.preprocessing.max_seq_length,
            global_step=0,
            pad_id=pad_id,
            soi_id=soi_id,
            eoi_id=eoi_id,
            mask_schedule=None,
            ignore_id=uni_prompting.ignore_id,
            config=config,
        )

        # Compute overall loss
        loss = (config.training.t2i_coeff * loss_t2i +
                config.training.lm_coeff * loss_lm +
                config.training.mmu_coeff * loss_mmu)

        print(f"Overall Loss: {loss.item()}")
        print(f"  T2I Loss: {loss_t2i.item()}")
        print(f"  LM Loss: {loss_lm.item()}")
        print(f"  MMU Loss: {loss_mmu.item()}")

if __name__ == "__main__":
    main()
