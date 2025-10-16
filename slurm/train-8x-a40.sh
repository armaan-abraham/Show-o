#!/bin/bash
#SBATCH --job-name=easy-train
#SBATCH --partition=iris
#SBATCH --account=iris
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a40:8
#SBATCH --time=04:00:00
#SBATCH --output=/iris/u/armaana/jobs/logs/%x_%j.out
#SBATCH --error=/iris/u/armaana/jobs/logs/%x_%j.err

# Print job information
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

cd /iris/u/armaana/Show-o-base/Show-o
. .venv/bin/activate

# We could check nproc but CPU isolation sometimes messes with this I think
export NUM_DATALOADER_WORKERS=64

# Your commands here
echo "Starting job"
PYTHONPATH=/iris/u/armaana/Show-o-base/Show-o/training:$PYTHONPATH accelerate launch --config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml training/train.py config=configs/train_easy_lg.yaml

echo "Job finished at: $(date)"