import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from castor.data.stage0_gaussian import GaussianPretrainingProvider
from castor.cloud.config_utils import load_config

def check_objectness_labels(config_path="config/local_fast.yaml"):
    config = load_config(config_path)
    data_cfg = config["data_params"]
    
    provider = GaussianPretrainingProvider(
        num_samples=100,
        min_stars=data_cfg["min_stars"],
        max_stars=data_cfg["max_stars"],
        image_size=data_cfg["image_size"],
        max_capacity_per_cell=data_cfg["max_capacity_per_cell"],
        shape_size=data_cfg["shape_size"]
    )
    
    all_p = []
    all_fluxes = []
    
    print(f"Sampling 20 chunks to check objectness label distribution...")
    for i in range(20):
        sparse_sample = provider.generate_chunk()
        # base_grid: [grid_h, grid_w, K, 4 + N_PCA]
        # index 0 is objectness (p)
        base_grid = sparse_sample["base_grid"].numpy()
        mask = base_grid[..., 0] > 0.0
        p_labels = base_grid[mask, 0]
        fluxes = base_grid[mask, 3] # raw flux
        
        all_p.extend(p_labels.tolist())
        all_fluxes.extend(fluxes.tolist())
        
    all_p = np.array(all_p)
    all_fluxes = np.array(all_fluxes)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(all_p, bins=20, range=(0, 1), color='blue', alpha=0.7)
    plt.axvline(0.5, color='red', linestyle='--', label='50% Threshold')
    plt.title(f"Objectness Label (p) Distribution\n(Mean: {np.mean(all_p):.3f})")
    plt.xlabel("Target Objectness (0 to 1)")
    plt.ylabel("Count")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(all_fluxes, all_p, alpha=0.1, s=1)
    plt.xscale('log')
    plt.title("Objectness vs Flux")
    plt.xlabel("Flux (Photons)")
    plt.ylabel("Target Objectness (p)")
    
    plt.tight_layout()
    plt.savefig("objectness_check.png")
    print(f"✅ Objectness check saved to objectness_check.png")
    print(f"Total Stars: {len(all_p)}")
    print(f"Stars with p > 0.5: {np.sum(all_p > 0.5)} ({100*np.mean(all_p > 0.5):.1f}%)")
    print(f"Stars with p > 0.9: {np.sum(all_p > 0.9)} ({100*np.mean(all_p > 0.9):.1f}%)")

if __name__ == "__main__":
    check_objectness_labels()
