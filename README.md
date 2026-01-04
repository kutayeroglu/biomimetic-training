# Biomimetic Training

## Purpose

This repository investigates how the proportion of "early visual experience" (blurry, achromatic inputs) influences neural network organization and behavior. The research is inspired by biological vision development, where early visual experience is characterized by limited acuity and color perception.

### Research Context

The project explores anti-biomimetic training regimens that mimic early visual development by training neural networks with degraded visual inputs during initial training phases. Specifically, we examine how:

- **Blurry inputs** (Gaussian blur with configurable sigma)
- **Achromatic inputs** (grayscale conversion)
- Applied during different phases of training

...affect the resulting network's feature representations, shape-texture bias, and overall behavior.

### Training Regimens

The codebase supports three main training regimens:

1. **Standard**: Normal training without any special input modifications
2. **Biomimetic**: Blurry, achromatic inputs during Phase 1 (early epochs), transitioning to normal inputs in Phase 2
3. **Anti-Biomimetic**: Normal inputs during Phase 1, transitioning to blurry, achromatic inputs in Phase 2

The original study used blurry, achromatic inputs for the first half of the training epochs, which corresponds to the biomimetic regimen with a transition epoch at the midpoint.

## Directory Structure

The project is organized into the following directories:

```
biomimetic-training/
├── src/                    # Core source code
│   ├── train.py           # Training loop with phase-based regimens
│   ├── models.py          # AlexNetModified architecture definition
│   ├── dataops/           # Data loading utilities
│   │   └── in1k.py        # ImageNet dataloaders and transforms
│   └── eval/              # Evaluation modules
│       ├── evaluate_bias.py        # Bias evaluation and ablation analysis
│       ├── utils.py                # Evaluation utilities
│       └── viz/                    # Visualization tools
│           ├── shape_texture_bias.py    # Shape-texture bias plotting
│           └── joint_color_freq.py      # Joint color-frequency analysis
├── scripts/               # Training and utility scripts (SLURM cluster execution)
│   ├── train_biomimetic.sh        # Biomimetic regimen training script
│   ├── train_base.sh              # Standard regimen training script
│   ├── train_anti_biomimetic.sh   # Anti-biomimetic regimen training script
│   ├── test_model.sh              # Model architecture testing script
│   ├── test_dataloader.sh         # DataLoader validation script
│   └── fetch_checkpoints.sh       # Checkpoint download utility
├── tests/                 # Test suite
│   └── test_in1k_loader.py        # ImageNet dataloader tests
├── logs/                  # Training logs and outputs (created during training)
├── main.py                # Main entry point for training
├── requirements.txt       # Python dependencies (excluding PyTorch)
├── Dockerfile             # Docker container definition with CUDA support
├── eval_biomim.ipynb      # Executor notebook for preparing analysis data
├── interpret_results.ipynb  # Shape/texture bias analysis notebook
└── plot_joint_color_freq.ipynb  # Joint color-frequency visualization notebook
```

### Directory Descriptions

- **`src/`**: Contains the core source code for the project.
  - `train.py`: Implements the training loop with support for phase-based training regimens
  - `models.py`: Defines the modified AlexNet architecture (`AlexNetModified`)
  - `dataops/`: Data loading and preprocessing utilities
    - `in1k.py`: ImageNet-1k dataloaders with support for blur and grayscale transforms
  - `eval/`: Evaluation and analysis modules
    - `evaluate_bias.py`: Performs ablation analysis to evaluate shape-texture bias
    - `utils.py`: Utility functions for evaluation
    - `viz/`: Visualization tools for plotting results

- **`scripts/`**: Contains bash scripts for training and testing. These scripts are primarily designed for SLURM cluster execution but can be adapted for local use. Each script handles a specific training regimen or testing task.

- **`tests/`**: Test suite for validating code functionality. Currently includes tests for the ImageNet dataloader.

- **`logs/`**: Output directory for training logs, checkpoints, and other runtime artifacts. This directory is created automatically during training.

- **Root-level files**:
  - `main.py`: Entry point for running training experiments with command-line arguments
  - `requirements.txt`: Python package dependencies (note: PyTorch must be installed separately)
  - `Dockerfile`: Container definition with CUDA-enabled PyTorch for GPU training
  - `*.ipynb`: Jupyter notebooks for evaluation, visualization, and result interpretation

## Prerequisites

Before installing and running this project, ensure you have the following:

- **Python 3.8+**: The project requires Python 3.8 or higher (tested with Python 3.10+ on Ubuntu 22.04)
- **PyTorch**: PyTorch 2.0+ is required. For GPU training (recommended), install PyTorch with CUDA support. See [PyTorch installation guide](https://pytorch.org/get-started/locally/) for platform-specific instructions.
  - **CUDA**: If using GPU training, CUDA 11.8+ is recommended (the Dockerfile uses CUDA 12.1)
- **ImageNet Dataset**: The ImageNet-1k dataset with training and validation sets is required. The dataset should be organized as:
  ```
  imagenet/
  ├── train/          # Training images (organized by class folders)
  └── val/            # Validation images (organized by class folders)
  ```
- **Weights & Biases (optional)**: For experiment tracking and visualization. Sign up at [wandb.ai](https://wandb.ai)
- **Git**: For cloning the repository
- **Virtual Environment** (recommended): Use `venv` or `conda` to manage dependencies

## Installation

### Clone the Repository

```bash
git clone <repository-url>
cd biomimetic-training
```

### Execute with Docker

For containerized execution, build and use the Docker image:

```bash
# Build the image (includes CUDA 12.1 and PyTorch 2.5.1)
docker build -t biomimetic-training .

# Run training with GPU support
docker run --gpus all -v /path/to/imagenet:/data/imagenet \
    biomimetic-training python main.py \
    --data-dir /data/imagenet/train \
    --val-dir /data/imagenet/val \
    --epochs 100
```

## Usage

### Quick Start

After completing installation, here's a quick workflow to get started:

1. **Verify Setup** (see Verification section below)
2. **Prepare ImageNet Dataset** - Ensure your ImageNet dataset is organized correctly
3. **Run Training** - Choose a training regimen (Standard, Biomimetic, or Anti-Biomimetic)
4. **Evaluate Results** - Use the evaluation tools and notebooks

### Verification

Before running full training experiments, verify that your installation and dataset are configured correctly:

#### Test the DataLoader

Test that the ImageNet dataloader works with your dataset:

```bash
# For local execution, you may need to modify the script or run directly:
export PYTHONPATH="$PWD:$PYTHONPATH"
python tests/test_in1k_loader.py
```

Or if using the test script (requires SLURM or manual modification):

```bash
bash scripts/test_dataloader.sh
```

#### Test the Model Architecture

Verify that the model can be instantiated correctly:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
python src/models.py
```

Or using the test script:

```bash
bash scripts/test_model.sh
```

#### Run a Small-Scale Training Test

Test the training pipeline with a small subset of data:

```bash
python main.py \
    --data-dir /path/to/imagenet/train \
    --val-dir /path/to/imagenet/val \
    --epochs 1 \
    --batch-size 32 \
    --train-frac 0.01 \
    --val-frac 0.01 \
    --save-path test_checkpoint.pth \
    --no-wandb
```

This will run a single epoch on 1% of the data to verify the training pipeline works.

### Training

The main training script is `main.py`. It supports various command-line arguments to configure training regimens.

#### Standard Training

Train a model with standard ImageNet training (no special input modifications):

```bash
python main.py \
    --data-dir /path/to/imagenet/train \
    --val-dir /path/to/imagenet/val \
    --epochs 100 \
    --batch-size 128 \
    --save-path standard_checkpoint.pth
```

#### Biomimetic Training

Train with biomimetic regimen (blurry, achromatic inputs in Phase 1, transitioning to normal inputs in Phase 2):

```bash
python main.py \
    --data-dir /path/to/imagenet/train \
    --val-dir /path/to/imagenet/val \
    --epochs 100 \
    --transition-epoch 50 \
    --phase1-blur-sigma 4.0 \
    --phase1-grayscale \
    --batch-size 128 \
    --save-path biomimetic_checkpoint.pth
```

#### Anti-Biomimetic Training

Train with anti-biomimetic regimen (normal inputs in Phase 1, transitioning to blurry, achromatic inputs in Phase 2):

```bash
python main.py \
    --data-dir /path/to/imagenet/train \
    --val-dir /path/to/imagenet/val \
    --epochs 100 \
    --transition-epoch 50 \
    --phase2-blur-sigma 4.0 \
    --phase2-grayscale \
    --batch-size 128 \
    --save-path anti_biomimetic_checkpoint.pth
```

### Using Training Scripts

The `scripts/` directory contains bash scripts primarily designed for SLURM cluster execution. These scripts can be adapted for local use:

**For SLURM clusters:**
```bash
# Submit a training job
sbatch scripts/train_biomimetic.sh
```

**For local execution:**
The scripts contain SLURM directives (SBATCH comments) that can be removed or ignored. You can:
1. Modify the scripts to remove SBATCH directives
2. Run the Python commands directly from the scripts
3. Set environment variables manually (PYTHONPATH, data paths, etc.)

Example adaptation for local use:
```bash
# Extract the core command from scripts/train_biomimetic.sh
# and modify paths as needed:
export PYTHONPATH="$PWD:$PYTHONPATH"
python main.py \
    --data-dir /your/local/path/to/imagenet/train \
    --val-dir /your/local/path/to/imagenet/val \
    --epochs 100 \
    --transition-epoch 50 \
    --phase1-blur-sigma 4.0 \
    --phase1-grayscale \
    --batch-size 128 \
    --save-path biomimetic_checkpoint.pth
```

**Note**: Always update data paths and cluster-specific configurations in the scripts before running.

### Key Training Arguments

- `--data-dir`: Path to ImageNet training data directory (required)
- `--val-dir`: Path to ImageNet validation directory (default: `data_dir/val`)
- `--epochs`: Total number of training epochs (default: 200)
- `--transition-epoch`: Epoch to switch from Phase 1 to Phase 2 (default: 0)
- `--phase1-blur-sigma`: Gaussian blur sigma for Phase 1 (default: 0.0)
- `--phase1-grayscale`: Use grayscale in Phase 1 (flag)
- `--phase2-blur-sigma`: Gaussian blur sigma for Phase 2 (default: 0.0)
- `--phase2-grayscale`: Use grayscale in Phase 2 (flag)
- `--batch-size`: Batch size for training (default: 128)
- `--num-workers`: Number of data loading workers (default: 8)
- `--save-path`: Path to save model checkpoint (default: `checkpoint.pth`)
- `--use-wandb`: Enable Weights & Biases logging (default: True)
- `--no-wandb`: Disable Weights & Biases logging

Run `python main.py --help` for a complete list of options.

### Evaluation

#### Bias Evaluation

Evaluate shape-texture bias using ablation analysis:

```python
from src.eval.evaluate_bias import run_evaluation

results = run_evaluation(
    model_path="biomimetic_checkpoint.pth",
    data_path="path/to/stylized_images",
    class_indices_path="data/categories16_class_indices.pkl",
    model_file="biomimetic",
    ranking_indices=["color", "fft_freq"],
    num_ab=48,
    result_path="texture_ablation.csv",
    overwrite=False,
)
```

