import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# Disable CUDA to force CPU-only execution (remove these lines if you want to use GPU)
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
import numpy as np
from PIL import Image
from pathlib import Path

from models import MAGVITv2
from models.easy import EasyTransformer
from training.prompting_utils import UniversalPrompting

# Configuration
OUTPUT_DIR = "generated_images"  # Path where generated images will be saved
NUM_SAMPLES_PER_PROMPT = 4  # Number of images to generate per prompt

# List of text prompts to generate images for
PROMPTS = [
    "cat",
    "sunset",
    "sports car",
    "golden retriever",
]

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load config
    config = OmegaConf.load("runs/easy/config.yaml")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
    print("Loading VQ model...")
    vq_model = MAGVITv2.from_pretrained(config.model.vq_model.vq_model_name)
    vq_model.to(device)
    vq_model.eval()
    vq_model.requires_grad_(False)

    # Load the transformer model
    print("Loading transformer model...")
    model = EasyTransformer(
        vocab_size=config.model.tokenize.vocab_size,
        **config.model.core
    )

    # Load checkpoint
    checkpoint_path = "runs/easy/checkpoint-70000/unwrapped_model/pytorch_model.bin"
    print(f"Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # Get special token IDs
    pad_id = int(uni_prompting.sptids_dict['<|pad|>'])
    soi_id = int(uni_prompting.sptids_dict['<|soi|>'])
    eoi_id = int(uni_prompting.sptids_dict['<|eoi|>'])

    # Get image length from config
    image_len = config.model.core.image_len

    # Prepare prompts with repetitions
    all_prompts = []
    prompt_labels = []
    for i, prompt in enumerate(PROMPTS):
        for j in range(NUM_SAMPLES_PER_PROMPT):
            all_prompts.append(prompt)
            prompt_labels.append(f"prompt_{i}_sample_{j}")

    print(f"\nGenerating {len(all_prompts)} images ({NUM_SAMPLES_PER_PROMPT} samples for each of {len(PROMPTS)} prompts)...")

    # Process in batches (to avoid memory issues)
    batch_size = config.training.batch_size_t2i
    num_batches = (len(all_prompts) + batch_size - 1) // batch_size

    all_generated_images = []
    all_labels = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_prompts))
        batch_prompts = all_prompts[start_idx:end_idx]
        batch_labels = prompt_labels[start_idx:end_idx]

        print(f"\nProcessing batch {batch_idx + 1}/{num_batches} ({len(batch_prompts)} images)...")

        # Tokenize text prompts
        text_ids = tokenizer(batch_prompts)['input_ids']
        print("Text ids:", text_ids)

        # Create input sequences using t2i_gen_prompt
        input_ids = uni_prompting.t2i_gen_prompt(
            text_ids,
            image_len=image_len,
            device=device,
        )
        print("Input ids shape:", input_ids.shape)
        print("Input ids:", input_ids[:, :10])

        input_ids = input_ids.to(device)

        # Generate images using sample_t2i
        print("Generating image tokens...")
        with torch.no_grad():
            generated_ids = model.sample_t2i(
                input_ids=input_ids,
                pad_id=pad_id,
                soi_id=soi_id,
                eoi_id=eoi_id,
                img_vocab_start_idx=config.model.tokenize.llm_vocab_size + config.model.tokenize.num_new_special_tokens,
            )

        # Extract image tokens from generated sequence
        # Find the position of soi_id and extract image_len tokens after it
        for i in range(generated_ids.shape[0]):
            # Find SOI token position
            soi_pos = (generated_ids[i] == soi_id).nonzero(as_tuple=True)[0].item()

            # Extract image tokens (between soi and eoi)
            image_tokens = generated_ids[i, soi_pos + 1 : soi_pos + 1 + image_len]

            # Subtract text tokenizer vocab size to get VQ codes
            image_tokens = image_tokens - len(uni_prompting.text_tokenizer)

            # Decode image tokens to pixel values
            print(f"Decoding image {i + 1}/{len(batch_prompts)}...")
            decoded_image = vq_model.decode_code(image_tokens.unsqueeze(0))

            # Normalize and convert to numpy
            decoded_image = torch.clamp((decoded_image + 1.0) / 2.0, min=0.0, max=1.0)
            decoded_image = decoded_image * 255.0
            decoded_image = decoded_image.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

            all_generated_images.append(decoded_image)
            all_labels.append(batch_labels[i])

    # Save all generated images
    print(f"\nSaving {len(all_generated_images)} images to {OUTPUT_DIR}...")
    for i, (image_np, label) in enumerate(zip(all_generated_images, all_labels)):
        # Create PIL image
        pil_image = Image.fromarray(image_np)

        # Save image with descriptive filename
        prompt_idx = i // NUM_SAMPLES_PER_PROMPT
        sample_idx = i % NUM_SAMPLES_PER_PROMPT
        prompt_text = PROMPTS[prompt_idx].replace(" ", "_")[:50]  # Truncate long prompts
        filename = f"{label}_{prompt_text}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)

        pil_image.save(filepath)
        print(f"Saved: {filename}")

    print(f"\nDone! Generated {len(all_generated_images)} images in {OUTPUT_DIR}/")

    # Print summary
    print("\nSummary:")
    for i, prompt in enumerate(PROMPTS):
        print(f"  Prompt {i}: '{prompt}' - {NUM_SAMPLES_PER_PROMPT} samples")

if __name__ == "__main__":
    main()
