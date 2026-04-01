import torch
import numpy as np
from castor.data.stage0_gaussian import GaussianPretrainingProvider
from castor.cloud.config_utils import load_config
import os

def check_density(config_path="config/config.yaml", num_chunks=5):
    config = load_config(config_path)
    data_cfg = config["data_params"]
    
    provider = GaussianPretrainingProvider(
        min_stars=data_cfg["min_stars"],
        max_stars=data_cfg["max_stars"],
        image_size=data_cfg["image_size"],
        max_capacity_per_cell=data_cfg["max_capacity_per_cell"],
        shape_size=data_cfg["shape_size"],
        global_stretch_scale=data_cfg.get("GLOBAL_STRETCH_SCALE", 10.0)
    )
    
    print(f"Checking star density using config: {config_path}")
    print(f"Config constraints: min_stars={data_cfg['min_stars']}, max_stars={data_cfg['max_stars']}")
    print("-" * 50)
    
    all_counts = []
    for i in range(num_chunks):
        sample = provider.generate_chunk()
        # base_grid: [G, G, K, 5] where [..., 0] is probability (1.0 for star)
        base_grid = sample["base_grid"]
        num_stars = (base_grid[..., 0] == 1.0).sum().item()
        all_counts.append(num_stars)
        print(f"Chunk {i}: {num_stars} detectable stars")
        
    print("-" * 50)
    print(f"Average detectable stars per 256x256 chunk: {np.mean(all_counts):.1f}")
    print(f"Min: {np.min(all_counts)}, Max: {np.max(all_counts)}")

if __name__ == "__main__":
    check_density()
