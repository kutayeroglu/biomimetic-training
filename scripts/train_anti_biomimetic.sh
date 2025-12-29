#!/bin/bash

#SBATCH --job-name=anti-biomimetic-regimen
#SBATCH --output=logs/anti-biomimetic-%j.out
#SBATCH --error=logs/anti-biomimetic-%j.err

#SBATCH --container-image ghcr.io\#kutayeroglu/biomim
#SBATCH --container-mounts /stratch/dataset:/datasets
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=12G
#SBATCH --time=12:00:00

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting Job: $SLURM_JOB_ID ---"
echo "Running on host: $(hostname)"
echo "Allocated GPUs: $SLURM_GPUS_ON_NODE"
echo ""

cd "$HOME/projects/biomimetic-training"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_GPU

echo "--- Executing main script for Anti-Biomimetic Regimen ---"
# Ensure your W&B API key is available as in the standard regimen script
export WANDB_API_KEY=$(cat $HOME/.wandb_api_key)

python3 main.py \
    --data-dir /datasets/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC \
    --val-dir $HOME/datasets/imagenet/val \
    --batch-size 128 \
    --train-frac 1 \
    --val-frac 1 \
    --epochs 200 \
    --transition-epoch 100 \
    --phase1-blur-sigma 0.0 \
    --phase2-blur-sigma 4.0 \
    --phase2-grayscale \
    --num-workers 10 \
    --save-path anti_biomimetic_checkpoint.pth
    
echo "--- Job Finished Successfully ---"