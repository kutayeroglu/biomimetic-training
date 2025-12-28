import time
import os
import sys
from dataops.in1k import get_imagenet_dataloaders


def test_imagenet_loading():
    # Get training directory (parent of train folder)
    TRAIN_DIR = os.getenv("IMAGENET_TRAIN_DIR")
    if not TRAIN_DIR and len(sys.argv) > 1:
        TRAIN_DIR = sys.argv[1]
    if not TRAIN_DIR:
        raise ValueError(
            "IMAGENET_TRAIN_DIR environment variable or command-line argument required"
        )

    # Get validation directory (full path to val folder)
    VAL_DIR = os.getenv("IMAGENET_VAL_DIR")
    if not VAL_DIR and len(sys.argv) > 2:
        VAL_DIR = sys.argv[2]
    if not VAL_DIR:
        raise ValueError(
            "IMAGENET_VAL_DIR environment variable or command-line argument required"
        )

    print("Testing dataloader:")
    print(f"  - Training data at: {TRAIN_DIR}/train")
    print(f"  - Validation data at: {VAL_DIR}")

    try:
        start_time = time.time()

        # 1. LOAD A TINY SUBSET
        # We use a very small fraction (0.001 = 0.1%) to make initialization fast
        train_loader, val_loader = get_imagenet_dataloaders(
            data_dir=TRAIN_DIR,
            val_dir=VAL_DIR,
            batch_size=4,  # Small batch for testing
            num_workers=2,  # Low workers for debugging
            train_frac=0.001,  # Load only 0.1% of training data
            val_frac=0.01,  # Load only 1% of validation data
        )

        print("Dataloaders created successfully.")
        print(f"  - Train subset size: {len(train_loader.sampler)} images")
        print(f"  - Val subset size:   {len(val_loader.sampler)} images")

        # 2. VERIFY BATCH STRUCTURE
        # Fetch one batch from the training loader
        print("Fetching first batch...")
        images, labels = next(iter(train_loader))

        print("Batch fetched successfully.")
        print(f"  - Image batch shape: {images.shape} (Expected: [4, 3, 227, 227])")
        print(f"  - Labels batch shape: {labels.shape}")

        # 3. VERIFY NORMALIZATION
        # Check if values are in the expected range [-1, 1]
        min_val = images.min().item()
        max_val = images.max().item()
        mean_val = images.mean().item()

        print(f"  - Pixel min: {min_val:.4f} (Expected: approx -1.0)")
        print(f"  - Pixel max: {max_val:.4f} (Expected: approx 1.0)")
        print(f"  - Pixel mean: {mean_val:.4f} (Expected: approx 0.0)")

        # 4. VERIFY RESIZING
        # Check if the inputs are strictly 227x227
        if images.shape[2:] == (227, 227):
            print("Shape check passed: Images are 227x227.")
        else:
            print(f"Shape check failed: Got {images.shape[2:]}")

        elapsed = time.time() - start_time
        print(f"Test finished in {elapsed:.2f} seconds.")

    except Exception as e:
        print(f"Test Failed: {e}")
        if "FileNotFoundError" in str(e):
            print(
                "  -> Hint: Check if your TRAIN_DIR contains a 'train' folder and VAL_DIR points to the 'val' folder."
            )


if __name__ == "__main__":
    test_imagenet_loading()
