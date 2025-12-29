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
IMG_SIZE = (256, 256, 3)  # Match TensorFlow version
NUM_CLASSES = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Executor Function ---

def run_evaluation(model_path, data_path, class_indices_path, model_file="", 
                  ranking_indices=["color", "fft_freq"], num_ab=48, n_top_col_pixel=48,
                  result_path=None, overwrite=False):
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
    print(f"--- Starting Evaluation on: {DEVICE} ---")
    
    # 1. Load 16-category index mapping
    with open(class_indices_path, "rb") as f:
        cate16_class_indices = pickle.load(f)
    
    # 2. Load test dataset (all images upfront, no transforms)
    test_files = glob.glob(os.path.join(data_path, "*.png"))
    x_test, y_test = load_images(test_files, IMG_SIZE)
    print(f"Loaded {len(test_files)} images")
    
    # Convert to torch tensors (uint8 -> float32, normalized to [0, 1])
    # Note: No normalization transforms - just convert to [0, 1] range
    x_test_tensor = torch.from_numpy(x_test).float() / 255.0
    # Permute from [N, H, W, C] to [N, C, H, W] for PyTorch
    x_test_tensor = x_test_tensor.permute(0, 3, 1, 2).to(DEVICE)
    
    # 3. Initialize and Load Model
    model = AlexNetModified(num_classes=NUM_CLASSES)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()
    
    # 4. Get conv1 layer and calculate rankings
    conv1 = model.features[0]
    original_weights = conv1.weight.data.cpu().numpy()  # [out_channels, in_channels, H, W]
    
    # Calculate rankings using same method as TensorFlow version
    color_index, fft_freq_index, fft_az_index = calc_rf_indices(
        original_weights, 
        n_top_col_pixel=n_top_col_pixel,
        return_rank=True
    )
    
    # 5. Result pandas setup
    res_pd = pd.DataFrame(index=cate16_class_indices.keys())
    
    # 6. Ablation Loop (matching TensorFlow structure)
    for reverse_order in [False, True]:
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
            
            for ablation in range(num_ab + 1):
                print(f"Currently working on ablation {ablation} ({ranking_index_name})")
                
                # Apply ablation
                with torch.no_grad():
                    modified_weights = original_weights.copy()
                    # Zero out the first 'ablation' filters (lowest metric values)
                    modified_weights[index[:ablation], :, :, :] = 0
                    conv1.weight.data = torch.from_numpy(modified_weights).to(DEVICE)
                
                # Make predictions on all images at once
                with torch.no_grad():
                    pred = model(x_test_tensor)  # [N, 1000]
                    probabilities = torch.softmax(pred, dim=1).cpu().numpy()
                
                # Count shape-texture stats
                count_dict = {}
                for cate in cate16_class_indices.keys():
                    count_dict[cate] = [0, 0, 0]
                
                for i in range(len(y_test)):
                    probs = probabilities[i]
                    assert len(probs) == NUM_CLASSES, "probabilities length is not correct"
                    cates = y_test[i].split("-")
                    shape_cate = re.sub('[^A-Za-z]+', '', cates[0])
                    text_cate = re.sub('[^A-Za-z]+', '', cates[1])
                    if shape_cate == text_cate:
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
                
                # Save to dataframe
                col_name = f"{ranking_index_name}_{model_file}_ablation_{ablation}" if model_file else f"{ranking_index_name}_ablation_{ablation}"
                res_pd[col_name] = [count_dict[cat] for cat in cate16_class_indices.keys()]
    
    # 7. Save results
    if result_path:
        os.makedirs(os.path.dirname(result_path) if os.path.dirname(result_path) else '.', exist_ok=True)
        if os.path.exists(result_path) and not overwrite:
            res_pd_prev = pd.read_csv(result_path, index_col=0)
            res_pd = pd.concat([res_pd_prev, res_pd], axis=1)
        res_pd.to_csv(result_path)
        print(f"Results saved to {result_path}")
    
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
        overwrite=False
    )
    print("\nEvaluation Complete.")
