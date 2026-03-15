import torch
import numpy as np
import matplotlib.pyplot as plt
from scripts.generate_simple_synthetic_data import GaussianStarDataset
from models.dense_grid_model import DenseGridModel

def run_inference(model_path="checkpoints/dense_grid_final.pth", threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = DenseGridModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. Generate Test Data
    dataset = GaussianStarDataset(min_stars=500, max_stars=1500)
    image_tensor, target_tensor, true_catalogue = dataset.generate_chunk()
    
    # 3. Forward Pass
    with torch.no_grad():
        input_tensor = image_tensor.unsqueeze(0).to(device)
        prediction = model(input_tensor).squeeze(0).cpu().numpy() # [128, 128, 5, 5]
    
    # 4. Post-Processing
    predicted_stars = []
    cell_size = dataset.cell_size
    grid_size = dataset.grid_size
    
    for y in range(grid_size):
        for x in range(grid_size):
            for k in range(5):
                slot = prediction[y, x, k]
                p, dx, dy, m, c = slot
                
                if p > threshold:
                    # Reconstruct coordinates (no pad)
                    global_x = (x * cell_size) + dx
                    global_y = (y * cell_size) + dy
                    predicted_stars.append((global_x, global_y, m, c, p))

    print(f"Ground Truth Stars: {len(true_catalogue)}")
    print(f"Predicted Stars (p > {threshold}): {len(predicted_stars)}")
    
    # 5. Visualization
    from matplotlib.colors import LogNorm
    img = image_tensor.squeeze().numpy()
    img_min = img.min()
    if img_min <= 0: img = img - img_min + 1e-3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    ax1.imshow(img, cmap='inferno', origin='lower', norm=LogNorm())
    ax1.set_title(f"Ground Truth ({len(true_catalogue)} stars)")
    for x, y, m, c in true_catalogue:
        ax1.plot(x, y, 'g+', markersize=10, alpha=0.6)
    
    ax2.imshow(img, cmap='inferno', origin='lower', norm=LogNorm())
    ax2.set_title(f"Model Predictions ({len(predicted_stars)} stars)")
    for x, y, m, c, p in predicted_stars:
        ax2.plot(x, y, 'r+', markersize=10, alpha=min(1.0, p))

    plt.suptitle("Dense Grid Model Inference: Edge-to-Edge (256x256)", fontsize=16)
    plt.savefig("inference_comparison.png")
    print("Comparison saved to inference_comparison.png")

if __name__ == "__main__":
    run_inference()
