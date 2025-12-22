#!/bin/bash

#SBATCH --job-name=biomimetic
#SBATCH --output=logs/biomimetic-%j.out
#SBATCH --error=logs/biomimetic-%j.err

#SBATCH --container-image ghcr.io\#kutayeroglu/ijepa
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

export OMP_NUM_THREADS=$SLURM_CPUS_PER_GPU
# Note: expandable_segments not supported on this platform
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "--- Executing main script ---"

# Build command with optional extra args
# CMD_ARGS=(
#     --fname configs/in1k_alexnets.yaml
#     --devices cuda:0
# )

# Add EXTRA_ARGS if provided (split by spaces to handle multiple arguments)
# if [ -n "${EXTRA_ARGS:-}" ]; then
#     # Split EXTRA_ARGS by spaces and add to array
#     read -ra EXTRA_ARRAY <<< "${EXTRA_ARGS}"
#     CMD_ARGS+=("${EXTRA_ARRAY[@]}")
# fi

# python3 main.py "${CMD_ARGS[@]}"
python3 main.py

echo "--- Job Finished Successfully ---"

