import torch
import numpy as np
import matplotlib.pyplot as plt
from castor.data.stage0_gaussian import GaussianPretrainingProvider

def check_pca_variance():
    print("Generating 100 PSFs and computing PCA...")
    # Initialize the provider (this automatically builds the library)
    provider = GaussianPretrainingProvider(image_size=256)
    
    # Generate the raw library of 100 elliptical Gaussians
    raw_library = provider._generate_elliptical_library(100, provider.render_kernel_size)
    
    N, H, W = raw_library.shape
    data = torch.from_numpy(raw_library).float().view(N, H * W)
    centered_data = data - data.mean(dim=0)
    
    # Run full-rank PCA (q=100) to see the entire spectrum of variance
    U, S, V = torch.pca_lowrank(centered_data, q=100)
    
    # Calculate Explained Variance
    # Variance is proportional to the square of the singular values (S)
    eigenvalues = (S ** 2).numpy()
    total_variance = np.sum(eigenvalues)
    
    explained_variance_ratio = eigenvalues / total_variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Find how many components are needed for 99% and 99.9% variance
    n_99 = np.argmax(cumulative_variance >= 0.99) + 1
    n_999 = np.argmax(cumulative_variance >= 0.999) + 1
    
    print(f"Components for 99.0% variance: {n_99}")
    print(f"Components for 99.9% variance: {n_999}")
    
    # Plot the Elbow Curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 101), cumulative_variance, marker='o', markersize=4, linestyle='-')
    plt.axvline(x=20, color='r', linestyle='--', label=f'Current N_PCA = 20 ({cumulative_variance[19]*100:.2f}%)')
    plt.axvline(x=n_99, color='g', linestyle='--', label=f'99% Variance (N = {n_99})')
    
    plt.title("Cumulative Explained Variance of PSF Shapes")
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Cumulative Variance Retained")
    plt.xlim(0, 30) # Zoom in on the first 30 components
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig("pca_variance_check.png")
    print("Saved plot to pca_variance_check.png")

if __name__ == "__main__":
    check_pca_variance()
