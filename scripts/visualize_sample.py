import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from src.data.dataset import PregeneratedDataset
from matplotlib.colors import LogNorm

def visualize_sample(data_dir, sample_idx=0, output_path="sample_visualization.png"):
    # 1. Load the dataset (handles re-densification)
    dataset = PregeneratedDataset(data_dir, K=3, shape_size=7)
    image, target = dataset[sample_idx]
    
    # 2. Extract components
    # image: [1, 256, 256]
    # target: [128, 128, 3, 54] (5: p, dx, dy, m, c | 49: shape)
    
    img_np = image.squeeze().numpy()
    
    # Find active stars in the target
    obj_mask = target[..., 0] == 1.0
    active_indices = torch.nonzero(obj_mask)
    
    print(f"Found {len(active_indices)} stars in sample {sample_idx}")
    
    # 3. Create Visualization
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)
    
    # Left: The full 256x256 image
    ax_img = fig.add_subplot(gs[:, 0:2])
    im = ax_img.imshow(img_np, cmap='inferno', origin='lower', norm=LogNorm(vmin=10, vmax=1000))
    ax_img.set_title(f"Input Image (256x256)\n{len(active_indices)} Stars")
    plt.colorbar(im, ax=ax_img, label='Flux')
    
    # Right: A 3x3 grid of individual 7x7 PSF shapes
    num_psfs = min(9, len(active_indices))
    psf_gs = gs[:, 2].subgridspec(3, 3)
    
    for i in range(num_psfs):
        idx = active_indices[i]
        # Extract the 49 shape values and reshape to 7x7
        shape_7x7 = target[idx[0], idx[1], idx[2], 5:].view(7, 7).numpy()
        
        ax_psf = fig.add_subplot(psf_gs[i // 3, i % 3])
        ax_psf.imshow(shape_7x7, cmap='viridis', origin='lower')
        ax_psf.axis('off')
        if i == 1:
            ax_psf.set_title("Individual 7x7 PSF Shapes")

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Visualization saved to {output_path}")

if __name__ == "__main__":
    # Use the debug data we just generated
    debug_dir = "data/debug/train"
    if os.path.exists(debug_dir):
        visualize_sample(debug_dir)
    else:
        print(f"❌ Error: Debug data not found at {debug_dir}")
