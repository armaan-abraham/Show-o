#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=iris
#SBATCH --account=iris
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --constraint="48G"
#SBATCH --time=20:00:00
#SBATCH --output=/iris/u/armaana/jobs/logs/%x_%j.out
#SBATCH --error=/iris/u/armaana/jobs/logs/%x_%j.err

# Print job information
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Get the parent directory of this script
PARENT_DIR=/iris/u/armaana/Show-o-base/2

. /iris/u/armaana/Show-o-base/Show-o/.venv/bin/activate
cd "$PARENT_DIR"

# We could check nproc but CPU isolation sometimes messes with this I think
export NUM_DATALOADER_WORKERS=32

# Your commands here
echo "Starting job"
echo "Working directory: $(pwd)"
PYTHONPATH="$PARENT_DIR/training:$PYTHONPATH" accelerate launch --config_file accelerate_configs/multi_gpu_deepspeed_zero2.yaml training/train.py config=configs/train_easy_vanilla.yaml

echo "Job finished at: $(date)"