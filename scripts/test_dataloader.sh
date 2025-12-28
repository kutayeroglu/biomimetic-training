#!/bin/bash

#SBATCH --job-name=bio-dataloader
#SBATCH --output=logs/bio-dataloader-%j.out
#SBATCH --error=logs/bio-dataloader-%j.err

#SBATCH --container-image ghcr.io\#kutayeroglu/biomim
#SBATCH --container-mounts /stratch/dataset:/datasets
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --time=01:00:00

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting Job: $SLURM_JOB_ID ---"
echo "Running on host: $(hostname)"
echo "Allocated GPUs: $SLURM_GPUS_ON_NODE"
echo ""

cd "$HOME/projects/biomimetic-training"

export PYTHONPATH="$PWD:$PYTHONPATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_GPU
# Note: expandable_segments not supported on this platform
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set ImageNet data paths
export IMAGENET_TRAIN_DIR="/datasets/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC"
export IMAGENET_VAL_DIR="$HOME/datasets/imagenet/val"

echo "--- Executing dataloader test script ---"

python3 tests/test_in1k_loader.py

echo "--- Job Finished Successfully ---"

