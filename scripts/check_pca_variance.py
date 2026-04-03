import torch
import numpy as np
import matplotlib.pyplot as plt
from castor.data.stage0_gaussian import GaussianPretrainingProvider

def check_pca_variance():
    print("Generating 100 PSFs and computing PCA...")
    # Initializing provider to get access to library generation logic
    provider = GaussianPretrainingProvider(num_samples=1)
    
    # Generate a fresh batch of 100 PSFs using the new optical-only logic
    raw_library = provider._generate_optical_library(100, provider.S)
    
    # Flatten for PCA
    N, H, W = raw_library.shape
    data = torch.from_numpy(raw_library).float().view(N, H * W)
    
    # Compute Mean
    mean_psf = data.mean(dim=0)
    centered_data = data - mean_psf
    
    # SVD
    U, S, V = torch.pca_lowrank(centered_data, q=50) # Check up to 50 components
    
    # Variance Explained
    total_var = torch.sum(S**2)
    var_explained = torch.cumsum(S**2, dim=0) / total_var
    
    # Find thresholds
    n99 = torch.where(var_explained >= 0.99)[0][0].item() + 1
    n999 = torch.where(var_explained >= 0.999)[0][0].item() + 1
    
    print(f"Components for 99.0% variance: {n99}")
    print(f"Components for 99.9% variance: {n999}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, 51), var_explained.numpy(), 'o-', label="Cumulative Variance")
    plt.axhline(y=0.99, color='r', linestyle='--', alpha=0.5, label="99%")
    plt.axhline(y=0.999, color='g', linestyle='--', alpha=0.5, label="99.9%")
    plt.axvline(x=10, color='k', linestyle=':', label="Current Limit (10)")
    
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Fraction of Variance Explained")
    plt.title("PCA Variance Analysis (Optical-Only Library)")
    plt.xlim(0, 30) # Zoom in on the first 30 components
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig("pca_variance_check.png")
    print("Saved plot to pca_variance_check.png")

if __name__ == "__main__":
    check_pca_variance()
