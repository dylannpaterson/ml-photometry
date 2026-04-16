import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.data.stage0_gaussian import GaussianPretrainingProvider
from src.cloud.config_utils import load_config

def check_completeness(config_path="config/local_fast.yaml"):
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
    
    all_completeness = []
    all_fluxes = []
    
    print(f"Sampling 20 chunks to check completeness distribution...")
    for i in range(20):
        sparse_sample = provider.generate_chunk()
        # base_grid: [grid_h, grid_w, K, 5]
        # index 4 is completeness
        base_grid = sparse_sample["base_grid"].numpy()
        mask = base_grid[..., 0] == 1.0
        comps = base_grid[mask, 4]
        fluxes = 10**base_grid[mask, 3]
        
        all_completeness.extend(comps.tolist())
        all_fluxes.extend(fluxes.tolist())
        
    all_completeness = np.array(all_completeness)
    all_fluxes = np.array(all_fluxes)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(all_completeness, bins=20, range=(0, 1), color='blue', alpha=0.7)
    plt.axvline(0.5, color='red', linestyle='--', label='50% Threshold')
    plt.title(f"Completeness Distribution\n(Mean: {np.mean(all_completeness):.3f})")
    plt.xlabel("Completeness (0 to 1)")
    plt.ylabel("Count")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(all_fluxes, all_completeness, alpha=0.1, s=1)
    plt.xscale('log')
    plt.title("Completeness vs Flux")
    plt.xlabel("Flux (Photons)")
    plt.ylabel("Completeness")
    
    plt.tight_layout()
    plt.savefig("completeness_check.png")
    print(f"✅ Completeness check saved to completeness_check.png")
    print(f"Total Stars: {len(all_completeness)}")
    print(f"Stars with c > 0.5: {np.sum(all_completeness > 0.5)} ({100*np.mean(all_completeness > 0.5):.1f}%)")
    print(f"Stars with c > 0.9: {np.sum(all_completeness > 0.9)} ({100*np.mean(all_completeness > 0.9):.1f}%)")

if __name__ == "__main__":
    check_completeness()
