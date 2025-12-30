#!/bin/bash

#SBATCH --job-name=biomim-regimen
#SBATCH --output=logs/biomim-regimen-%j.out
#SBATCH --error=logs/biomim-regimen-%j.err

#SBATCH --container-image ghcr.io\#kutayeroglu/biomim
#SBATCH --container-mounts /stratch/dataset:/datasets
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=16G
#SBATCH --time=2-12:00:00

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting Job: $SLURM_JOB_ID ---"
echo "Running on host: $(hostname)"
echo "Allocated GPUs: $SLURM_GPUS_ON_NODE"
echo ""

cd "$HOME/projects/biomimetic-training"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_GPU

echo "--- Executing main script (Biomimetic Regimen) ---"
export WANDB_API_KEY=$(cat $HOME/.wandb_api_key)

# The following arguments are added/modified compared to the standard regimen:
# 1. --transition-epoch 100: Sets the midpoint for AlexNet's 200 epochs
# 2. --phase1-blur-sigma 4.0: Applies the Gaussian blur specified in the study
# 3. --phase1-grayscale: Activates achromatic input for the first phase
# 4. --save-path: Renamed to avoid overwriting the standard regimen checkpoint

python3 main.py \
    --data-dir /datasets/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC \
    --val-dir $HOME/datasets/imagenet/val \
    --batch-size 128 \
    --train-frac 1 \
    --val-frac 1 \
    --epochs 200 \
    --num-workers 10 \
    --transition-epoch 100 \
    --phase1-blur-sigma 4.0 \
    --phase1-grayscale \
    --save-path biomimetic_checkpoint.pth
    
echo "--- Job Finished Successfully ---"