#!/bin/bash

#SBATCH --job-name=bio-alexnet
#SBATCH --output=logs/bio-alexnet-%j.out
#SBATCH --error=logs/bio-alexnet-%j.err

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

echo "--- Executing models script ---"

python3 src/models.py

echo "--- Job Finished Successfully ---"

