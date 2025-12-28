import argparse
import json
import os
from datetime import datetime

import torch

from src.models import AlexNetModified, get_training_setup
from src.train import train_model
from src.dataops.in1k import get_imagenet_dataloaders


def main():
    parser = argparse.ArgumentParser(description="Biomimetic Training")

    # Required arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to ImageNet training data directory",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default=None,
        help="Path to ImageNet validation data directory (default: data_dir/val)",
    )

    # Optional dataloader arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training (default: 128)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of data loading workers (default: 8)",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=None,
        help="Fraction of training data to use (default: None, use all)",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=None,
        help="Fraction of validation data to use (default: None, use all)",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Total number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--transition-epoch",
        type=int,
        default=0,
        help="Epoch to switch from Phase 1 to Phase 2 (default: 0, no transition)",
    )
    parser.add_argument(
        "--phase1-blur-sigma",
        type=float,
        default=0.0,
        help="Gaussian blur sigma for Phase 1 (default: 0.0)",
    )
    parser.add_argument(
        "--phase1-grayscale",
        action="store_true",
        help="Use grayscale in Phase 1 (default: False)",
    )
    parser.add_argument(
        "--phase2-blur-sigma",
        type=float,
        default=0.0,
        help="Gaussian blur sigma for Phase 2 (default: 0.0)",
    )
    parser.add_argument(
        "--phase2-grayscale",
        action="store_true",
        help="Use grayscale in Phase 2 (default: False)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="checkpoint.pth",
        help="Path to save model checkpoint (default: checkpoint.pth)",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        default=True,
        help="Use Weights & Biases for experiment tracking (default: True)",
    )
    parser.add_argument(
        "--no-wandb",
        dest="use_wandb",
        action="store_false",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="biomimetic-training",
        help="W&B project name (default: biomimetic-training)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (default: auto-generated from regimen)",
    )

    args = parser.parse_args()

    # Get imagenet dataloaders
    train_loader, val_loader = get_imagenet_dataloaders(
        data_dir=args.data_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )

    print("Dataloaders created successfully")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")

    # Initialize model
    model = AlexNetModified(num_classes=1000)

    # Handle device placement
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Setup training components
    optimizer, criterion = get_training_setup(model)

    # Print model information
    num_params = sum(p.numel() for p in model.parameters())
    print("\nModel initialized successfully")
    print(f"  - Device: {device}")
    print(f"  - Total parameters: {num_params:,}")

    # Collect all training parameters
    training_config = {
        # Data configuration
        "data_dir": args.data_dir,
        "val_dir": args.val_dir,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        # Model configuration
        "num_classes": 1000,
        "optimizer": "SGD",
        "learning_rate": 0.001,
        "momentum": 0.9,
        "nesterov": True,
        "criterion": "CrossEntropyLoss",
        # Training regimen
        "total_epochs": args.epochs,
        "transition_epoch": args.transition_epoch,
        "phase1_blur_sigma": args.phase1_blur_sigma,
        "phase1_grayscale": args.phase1_grayscale,
        "phase2_blur_sigma": args.phase2_blur_sigma,
        "phase2_grayscale": args.phase2_grayscale,
        # Hardware
        "device": device,
        # Paths
        "save_path": args.save_path,
        # Metadata
        "timestamp": datetime.now().isoformat(),
    }

    # Log parameters to console
    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    for key, value in training_config.items():
        print(f"  {key}: {value}")
    print("=" * 60 + "\n")

    # Save parameters to JSON file (alongside checkpoint)
    config_path = os.path.splitext(args.save_path)[0] + "_config.json"
    config_dir = os.path.dirname(config_path)
    if config_dir and not os.path.exists(config_dir):
        os.makedirs(config_dir, exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(training_config, f, indent=2)
    print(f"Configuration saved to: {config_path}\n")

    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        total_epochs=args.epochs,
        transition_epoch=args.transition_epoch,
        phase1_blur_sigma=args.phase1_blur_sigma,
        phase1_grayscale=args.phase1_grayscale,
        phase2_blur_sigma=args.phase2_blur_sigma,
        phase2_grayscale=args.phase2_grayscale,
        device=device,
        save_path=args.save_path,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    # Evaluate model?
    # TODO: Add evaluation


if __name__ == "__main__":
    main()
