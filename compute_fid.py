#!/usr/bin/env python3
"""
Compute FID (Fréchet Inception Distance) between generated images and ImageNet training set.

This script compares a folder of PNG files against the ImageNet training dataset
to evaluate image generation quality.
"""

import os
import sys
import torch
from torch_fidelity import calculate_metrics
from PIL import Image
import tempfile
import shutil
from pathlib import Path

# ============================================================================
# CONFIGURATION - Modify these constants as needed
# ============================================================================

# Path to ImageNet training dataset (same as in configs/train_easy_geo.yaml)
IMAGENET_TRAIN_PATH = "/iris/u/armaana/datasets/ILSVRC2012_img_train"
THIS_DIR = Path(__file__).parent

# Number of images to sample from each dataset for FID computation
# Set to None to use all available images
NUM_GENERATED_IMAGES = 4
NUM_IMAGENET_IMAGES = 4

# ============================================================================


def collect_images_from_folder(folder_path, num_images=None, extensions=('.png',)):
    """
    Collect image paths from a folder.

    Args:
        folder_path: Path to folder containing images
        num_images: Maximum number of images to collect (None for all)
        extensions: Tuple of valid image extensions

    Returns:
        List of image file paths
    """
    image_paths = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(extensions):
                image_paths.append(os.path.join(root, file))

    if num_images is not None and len(image_paths) > num_images:
        # Sample random subset
        import random
        random.seed(42)  # For reproducibility
        image_paths = random.sample(image_paths, num_images)

    return image_paths


def collect_imagenet_images(imagenet_path, num_images=None):
    """
    Collect image paths from ImageNet training dataset.

    ImageNet training data is organized as: imagenet_path/class_folder/image.JPEG

    Args:
        imagenet_path: Path to ImageNet training dataset
        num_images: Maximum number of images to collect (None for all)

    Returns:
        List of image file paths
    """
    extensions = ('.jpg', '.jpeg', '.JPEG', '.JPG')
    return collect_images_from_folder(imagenet_path, num_images, extensions)


def create_temp_dataset(image_paths, temp_dir):
    """
    Create a temporary flat directory with images for torch-fidelity.

    torch-fidelity expects images in a flat directory or organized by class.
    We create a flat directory with symlinks to avoid copying large files.

    Args:
        image_paths: List of source image paths
        temp_dir: Temporary directory to create dataset in

    Returns:
        Path to temporary dataset directory
    """
    dataset_dir = os.path.join(temp_dir, 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)

    for idx, img_path in enumerate(image_paths):
        ext = os.path.splitext(img_path)[1]
        link_path = os.path.join(dataset_dir, f'image_{idx:06d}{ext}')

        # Use symlink to avoid copying files
        try:
            os.symlink(img_path, link_path)
        except OSError:
            # If symlink fails (e.g., on some filesystems), copy instead
            shutil.copy2(img_path, link_path)

    return dataset_dir


def compute_fid(generated_folder, imagenet_folder,
                num_generated=None, num_imagenet=None):
    """
    Compute FID between generated images and ImageNet dataset.

    Args:
        generated_folder: Path to folder with generated PNG images
        imagenet_folder: Path to ImageNet training dataset
        num_generated: Number of generated images to use (None for all)
        num_imagenet: Number of ImageNet images to use (None for all)

    Returns:
        Dictionary with FID metric and other statistics
    """
    print("Collecting generated images...")
    generated_paths = collect_images_from_folder(
        generated_folder,
        num_generated,
        extensions=('.png', '.PNG')
    )
    print(f"Found {len(generated_paths)} generated images")

    print("\nCollecting ImageNet images...")
    imagenet_paths = collect_imagenet_images(
        imagenet_folder,
        num_imagenet
    )
    print(f"Found {len(imagenet_paths)} ImageNet images")

    if len(generated_paths) == 0:
        raise ValueError(f"No images found in {generated_folder}")
    if len(imagenet_paths) == 0:
        raise ValueError(f"No images found in {imagenet_folder}")

    # Create temporary directories for torch-fidelity
    print("\nPreparing datasets for FID computation...")
    with tempfile.TemporaryDirectory(dir=THIS_DIR) as temp_dir:
        gen_dataset_dir = create_temp_dataset(generated_paths,
                                               os.path.join(temp_dir, 'gen'))
        imagenet_dataset_dir = create_temp_dataset(imagenet_paths,
                                                    os.path.join(temp_dir, 'imagenet'))

        print("\nComputing FID (this may take several minutes)...")
        metrics = calculate_metrics(
            input1=gen_dataset_dir,
            input2=imagenet_dataset_dir,
            cuda=torch.cuda.is_available(),
            isc=True,
            fid=True,
            kid=True,
            verbose=True,
            batch_size=1
        )

    return metrics


def main():
    """Main function to compute and display FID."""
    if len(sys.argv) < 2:
        print("Usage: python compute_fid.py <generated_images_folder>")
        sys.exit(1)

    generated_images_folder = sys.argv[1]

    print("=" * 80)
    print("FID Computation between Generated Images and ImageNet")
    print("=" * 80)
    print(f"\nGenerated images folder: {generated_images_folder}")
    print(f"ImageNet training path: {IMAGENET_TRAIN_PATH}")
    print(f"Number of generated images to use: {NUM_GENERATED_IMAGES or 'all'}")
    print(f"Number of ImageNet images to use: {NUM_IMAGENET_IMAGES or 'all'}")
    print()

    # Check if paths exist
    if not os.path.exists(generated_images_folder):
        print(f"ERROR: Generated images folder does not exist: {generated_images_folder}")
        sys.exit(1)

    if not os.path.exists(IMAGENET_TRAIN_PATH):
        print(f"ERROR: ImageNet path does not exist: {IMAGENET_TRAIN_PATH}")
        print("Please update IMAGENET_TRAIN_PATH constant in this script.")
        return

    # Compute FID
    try:
        metrics = compute_fid(
            generated_images_folder,
            IMAGENET_TRAIN_PATH,
            NUM_GENERATED_IMAGES,
            NUM_IMAGENET_IMAGES
        )

        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"FID Score: {metrics['frechet_inception_distance']:.4f}")
        print("\nLower FID scores indicate better image quality and diversity.")
        print("Typical FID ranges:")
        print("  - Excellent: < 10")
        print("  - Good: 10-30")
        print("  - Fair: 30-50")
        print("  - Poor: > 50")
        print("=" * 80)

    except Exception as e:
        print(f"\nERROR during FID computation: {e}")
        raise


if __name__ == "__main__":
    main()
