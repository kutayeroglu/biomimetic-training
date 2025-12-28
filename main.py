import argparse

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

    print(f"Dataloaders created successfully")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")

    # Initialize model
    # TODO: Add model initialization

    # Train model
    # TODO: Add training call

    # Evaluate model?
    # TODO: Add evaluation


if __name__ == "__main__":
    main()
