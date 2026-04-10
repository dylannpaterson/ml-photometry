import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_psf_basis(basis_path="master_psf_library.pt", output_path="psf_basis_visualization.png"):
    if not os.path.exists(basis_path):
        print(f"❌ Error: Basis file not found at {basis_path}")
        return

    print(f"📂 Loading PSF basis from {basis_path}...")
    data = torch.load(basis_path, map_location="cpu", weights_only=False)
    
    if isinstance(data, dict):
        print("🔍 Detected dictionary format, extracting components...")
        mean_psf = data['mean_psf']
        psf_basis = data['eigen_psfs']
        n_pca = data['n_pca']
        S = data['shape_size']
    else:
        # Expected shape: [N_PCA + 1, S*S]
        n_rows, n_pix = data.shape
        S = int(np.sqrt(n_pix))
        n_pca = n_rows - 1
        psf_basis = data[:-1, :].reshape(n_pca, S, S)
        mean_psf = data[-1, :].reshape(S, S)
    
    print(f"📊 Basis Info: S={S}, N_PCA={n_pca}")
    
    # 2. Setup Plot
    n_plot = min(11, n_pca) # Plot up to 11 components (plus mean = 12 total)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    # Plot Mean PSF
    im = axes[0].imshow(mean_psf, cmap='magma', origin='lower')
    axes[0].set_title("Mean PSF", fontsize=14)
    plt.colorbar(im, ax=axes[0])
    
    # Plot PCA Components (Eigen-images)
    for i in range(n_plot):
        comp = psf_basis[i]
        # Use seismic or bwr for components as they have +/- values
        im = axes[i+1].imshow(comp, cmap='seismic', origin='lower')
        axes[i+1].set_title(f"PCA Comp {i}", fontsize=14)
        plt.colorbar(im, ax=axes[i+1])
        
    # Hide unused axes
    for j in range(n_plot + 1, len(axes)):
        axes[j].axis('off')
        
    plt.suptitle(f"PSF Basis Visualization: {basis_path}\n(Mean + First {n_plot} PCA Components)", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    print(f"✅ Visualization saved to {output_path}")

if __name__ == "__main__":
    visualize_psf_basis()
