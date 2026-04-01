import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_psf_basis(basis_path="stage0_psf_basis.pt", output_path="psf_basis_visualization.png"):
    if not os.path.exists(basis_path):
        print(f"❌ Error: Basis file not found at {basis_path}")
        return

    print(f"📂 Loading PSF basis from {basis_path}...")
    # Expected shape: [N_PCA + 1, 961]
    basis = torch.load(basis_path, map_location="cpu")
    
    if isinstance(basis, dict):
        print("🔍 Detected dictionary format, attempting to extract components...")
        # Add logic here if the .pt is a dict, but verified earlier it's a Tensor
        return

    n_rows, n_pix = basis.shape
    S = int(np.sqrt(n_pix))
    n_pca = n_rows - 1
    
    print(f"📊 Basis Shape: {basis.shape} (S={S}, N_PCA={n_pca})")
    
    # 1. Extract Mean and Components
    psf_basis = basis[:-1, :] # [N_PCA, 961]
    mean_psf = basis[-1, :].reshape(S, S) # [S, S]
    
    # 2. Setup Plot
    n_plot = min(10, n_pca) # Plot up to 10 components
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    # Plot Mean PSF
    im = axes[0].imshow(mean_psf, cmap='magma', origin='lower')
    axes[0].set_title("Mean PSF", fontsize=14)
    fig.colorbar(im, ax=axes[0])
    
    # Plot PCA Components (Eigen-images)
    for i in range(n_plot):
        comp = psf_basis[i].reshape(S, S).numpy()
        # Use seismic or bwr for components as they have +/- values
        im = axes[i+1].imshow(comp, cmap='seismic', origin='lower')
        axes[i+1].set_title(f"PCA Comp {i}", fontsize=14)
        fig.colorbar(im, ax=axes[i+1])
        
    # Hide unused axes
    for j in range(n_plot + 1, len(axes)):
        axes[j].axis('off')
        
    plt.suptitle(f"PSF Basis Visualization: {basis_path}\n(Mean + First {n_plot} PCA Components)", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    print(f"✅ Visualization saved to {output_path}")

if __name__ == "__main__":
    visualize_psf_basis()
