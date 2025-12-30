import os
import re
import glob
import pickle
import torch
import numpy as np
import pandas as pd

# Import your model architecture
from src.models import AlexNetModified
from src.eval.utils import calc_rf_indices, load_images, make_decision

# --- Constants & Configuration ---
IMG_SIZE = (227, 227, 3)  # Match TensorFlow version
NUM_CLASSES = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Executor Function ---


def run_evaluation(
    model_path,
    data_path,
    class_indices_path,
    model_file="",
    ranking_indices=["color", "fft_freq"],
    num_ab=48,
    n_top_col_pixel=48,
    result_path=None,
    overwrite=False,
):
    """
    This function tests the ablation of the neural network model on texture transferred images.

    Args:
        model_path: Path to the model checkpoint file
        data_path: Path to directory containing PNG images
        class_indices_path: Path to categories16_class_indices.pkl
        model_file: Name identifier for the model (for result naming)
        ranking_indices: List of ranking modes ["color", "fft_freq", "fft_az"]
        num_ab: Maximum number of filters to ablate
        n_top_col_pixel: Number of top color pixels for color metric
        result_path: Path to save results CSV (optional)
        overwrite: Whether to overwrite existing results
    """
    print("=" * 80)
    print(f"--- Starting Evaluation on: {DEVICE} ---")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Model path: {model_path}")
    print(f"  Data path: {data_path}")
    print(f"  Class indices path: {class_indices_path}")
    print(f"  Model file identifier: {model_file if model_file else '(none)'}")
    print(f"  Ranking indices: {ranking_indices}")
    print(f"  Max ablation: {num_ab}")
    print(f"  Top color pixels: {n_top_col_pixel}")
    print(f"  Result path: {result_path if result_path else '(not saving)'}")
    print(f"  Overwrite: {overwrite}")
    print("-" * 80)

    # 1. Load 16-category index mapping
    print("\n[Step 1/7] Loading 16-category class indices...")
    print(f"  Reading from: {class_indices_path}")
    with open(class_indices_path, "rb") as f:
        cate16_class_indices = pickle.load(f)
    print(
        f"  ✓ Loaded {len(cate16_class_indices)} categories: {list(cate16_class_indices.keys())}"
    )

    # 2. Load test dataset (all images upfront, no transforms)
    print("\n[Step 2/7] Loading test dataset...")
    print(f"  Searching for PNG images in: {data_path}")
    test_files = glob.glob(os.path.join(data_path, "*.png"))
    print(f"  Found {len(test_files)} PNG files")

    if len(test_files) == 0:
        print(f"  ⚠ WARNING: No PNG files found in {data_path}")
        return pd.DataFrame()

    x_test, y_test = load_images(test_files, IMG_SIZE, padding_mode="reflect")
    print(f"  ✓ Loaded {len(test_files)} images")
    print(f"  Image shape: {x_test.shape}, dtype: {x_test.dtype}")
    print(f"  Sample labels: {y_test[:3] if len(y_test) >= 3 else y_test}")

    # Convert to torch tensors (uint8 -> float32, normalized to [0, 1])
    print("\n[Step 3/7] Converting images to PyTorch tensors...")
    # Note: No normalization transforms - just convert to [0, 1] range
    x_test_tensor = torch.from_numpy(x_test).float() / 255.0
    # Permute from [N, H, W, C] to [N, C, H, W] for PyTorch
    x_test_tensor = x_test_tensor.permute(0, 3, 1, 2).to(DEVICE)
    print(f"  ✓ Converted to tensor shape: {x_test_tensor.shape}")
    print(f"  Tensor dtype: {x_test_tensor.dtype}, device: {x_test_tensor.device}")
    print(
        f"  Value range: [{x_test_tensor.min().item():.3f}, {x_test_tensor.max().item():.3f}]"
    )

    # 3. Initialize and Load Model
    print("\n[Step 4/7] Loading model...")
    print(f"  Initializing AlexNetModified with {NUM_CLASSES} classes...")
    model = AlexNetModified(num_classes=NUM_CLASSES)
    print(f"  Loading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()
    print(f"  ✓ Model loaded successfully")
    print(f"  Model device: {next(model.parameters()).device}")
    print(f"  Model in eval mode: {not model.training}")

    # 4. Get conv1 layer and calculate rankings
    print("\n[Step 5/7] Calculating filter rankings...")
    conv1 = model.features[0]
    original_weights = (
        conv1.weight.data.cpu().numpy()
    )  # [out_channels, in_channels, H, W]
    print(f"  Conv1 layer weights shape: {original_weights.shape}")
    print(f"  Number of filters (out_channels): {original_weights.shape[0]}")
    print(f"  Calculating rankings (n_top_col_pixel={n_top_col_pixel})...")

    color_index, fft_freq_index, fft_az_index = calc_rf_indices(
        original_weights, n_top_col_pixel=n_top_col_pixel, return_rank=True
    )
    print(f"  ✓ Rankings calculated:")
    print(f"    Color index shape: {color_index.shape}")
    print(f"    FFT frequency index shape: {fft_freq_index.shape}")
    print(f"    FFT azimuth index shape: {fft_az_index.shape}")
    print(f"    First 5 color-ranked filters: {color_index[:5]}")
    print(f"    First 5 fft_freq-ranked filters: {fft_freq_index[:5]}")

    # 5. Result pandas setup
    print("\n[Step 6/7] Setting up results dataframe...")
    res_pd = pd.DataFrame(index=cate16_class_indices.keys())
    print(f"  ✓ DataFrame initialized with {len(res_pd)} categories")

    # 6. Ablation Loop (matching TensorFlow structure)
    print("\n[Step 7/7] Starting ablation experiments...")
    print("=" * 80)

    total_iterations = len([False, True]) * len(ranking_indices) * (num_ab + 1)
    current_iteration = 0

    for reverse_order in [False, True]:
        reverse_str = "reverse" if reverse_order else "forward"
        print(f"\n--- Processing {reverse_str} order ---")

        for ranking_index in ranking_indices:
            if ranking_index == "color":
                index = color_index
            elif ranking_index == "fft_freq":
                index = fft_freq_index
            elif ranking_index == "fft_az":
                index = fft_az_index
            else:
                raise ValueError("Invalid ranking_index: " + ranking_index)

            if reverse_order:
                index = np.flip(index)
                ranking_index_name = ranking_index + "_reverse"
            else:
                ranking_index_name = ranking_index

            print(f"\n  Ranking method: {ranking_index_name}")
            print(f"  Filters to process: {len(index)}")

            for ablation in range(num_ab + 1):
                current_iteration += 1
                progress_pct = (current_iteration / total_iterations) * 100
                print(
                    f"\n  [{current_iteration}/{total_iterations} ({progress_pct:.1f}%)] Ablation {ablation}/{num_ab} ({ranking_index_name})"
                )

                # Apply ablation
                with torch.no_grad():
                    modified_weights = original_weights.copy()
                    # Zero out the first 'ablation' filters (lowest metric values)
                    filters_to_zero = index[:ablation]
                    modified_weights[filters_to_zero, :, :, :] = 0
                    conv1.weight.data = torch.from_numpy(modified_weights).to(DEVICE)

                if ablation > 0:
                    print(
                        f"    Zeroed out {ablation} filters: {filters_to_zero.tolist()}"
                    )

                # Make predictions on all images at once
                print(f"    Making predictions on {len(x_test_tensor)} images...")
                with torch.no_grad():
                    pred = model(x_test_tensor)  # [N, 1000]
                    probabilities = torch.softmax(pred, dim=1).cpu().numpy()
                print(f"    ✓ Predictions complete. Shape: {probabilities.shape}")

                # Count shape-texture stats
                print(f"    Counting shape-texture statistics...")
                count_dict = {}
                for cate in cate16_class_indices.keys():
                    count_dict[cate] = [
                        0,
                        0,
                        0,
                    ]  # [shape_correct, other, texture_correct]

                processed_images = 0
                skipped_images = 0
                for i in range(len(y_test)):
                    probs = probabilities[i]
                    assert len(probs) == NUM_CLASSES, (
                        "probabilities length is not correct"
                    )
                    cates = y_test[i].split("-")
                    shape_cate = re.sub("[^A-Za-z]+", "", cates[0])
                    text_cate = re.sub("[^A-Za-z]+", "", cates[1])
                    if shape_cate == text_cate:
                        skipped_images += 1
                        continue

                    decision = make_decision(probs, cate16_class_indices)
                    count = count_dict[shape_cate]
                    if shape_cate == decision:
                        count[0] = count[0] + 1
                    elif text_cate == decision:
                        count[2] = count[2] + 1
                    else:
                        count[1] = count[1] + 1
                    count_dict[shape_cate] = count
                    processed_images += 1

                print(
                    f"    ✓ Statistics complete. Processed: {processed_images}, Skipped: {skipped_images}"
                )

                # Show sample statistics for first category
                if len(count_dict) > 0:
                    first_cat = list(count_dict.keys())[0]
                    sample_counts = count_dict[first_cat]
                    print(
                        f"    Sample counts for '{first_cat}': shape={sample_counts[0]}, other={sample_counts[1]}, texture={sample_counts[2]}"
                    )

                # Save to dataframe
                col_name = (
                    f"{ranking_index_name}_{model_file}_ablation_{ablation}"
                    if model_file
                    else f"{ranking_index_name}_ablation_{ablation}"
                )
                res_pd[col_name] = [
                    count_dict[cat] for cat in cate16_class_indices.keys()
                ]
                print(f"    ✓ Saved to column: {col_name}")

    print("\n" + "=" * 80)
    print("Ablation experiments complete!")
    print(f"Total columns in results: {len(res_pd.columns)}")
    print(f"Total categories: {len(res_pd)}")

    # 7. Save results
    if result_path:
        print(f"\n[Final Step] Saving results...")
        print(f"  Target path: {result_path}")
        os.makedirs(
            os.path.dirname(result_path) if os.path.dirname(result_path) else ".",
            exist_ok=True,
        )

        if os.path.exists(result_path) and not overwrite:
            print(f"  Existing file found. Loading previous results...")
            res_pd_prev = pd.read_csv(result_path, index_col=0)
            print(f"  Previous results shape: {res_pd_prev.shape}")
            res_pd = pd.concat([res_pd_prev, res_pd], axis=1)
            print(f"  Merged results shape: {res_pd.shape}")
        else:
            if os.path.exists(result_path):
                print(f"  ⚠ Overwriting existing file")
            else:
                print(f"  Creating new file")

        res_pd.to_csv(result_path)
        print(f"  ✓ Results saved successfully to {result_path}")
        print(f"  Final dataframe shape: {res_pd.shape}")
    else:
        print(f"\n[Final Step] Results not saved (result_path not provided)")

    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)

    return res_pd


if __name__ == "__main__":
    # Example usage
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
    print("\nEvaluation Complete.")
