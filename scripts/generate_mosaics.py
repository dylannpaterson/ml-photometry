import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from src.data.stage0_gaussian import GaussianPretrainingProvider
from src.cloud.config_utils import load_config
import shutil

def generate_mosaic(idx, output_dir, params):
    """Generates a full SCA-sized mosaic (4088x4088) and its grid target."""
    # Proportional star count for SCA size
    # 4088x4088 is approx 255x larger than 256x256
    sca_min_stars = params['min_stars'] * 200
    sca_max_stars = params['max_stars'] * 200
    
    provider = GaussianPretrainingProvider(
        min_stars=sca_min_stars,
        max_stars=sca_max_stars,
        image_size=4088,
        max_capacity_per_cell=params['max_capacity_per_cell'],
        shape_size=params['shape_size']
    )
    
    print(f"Generating Mosaic {idx} (approx {sca_max_stars} stars)...")
    # Using the provider to generate the large image and sparse data
    sparse_sample = provider.generate_chunk()
    
    # Redensify into full SCA grid [1022, 1022, K, 87]
    # 4088 / 4 = 1022
    grid_size = 1022
    K = provider.K
    S2 = provider.S * provider.S
    # We use float16 for the target to keep file size manageable (~700MB per target)
    target = torch.zeros((grid_size, grid_size, K, 5 + S2 + 1), dtype=torch.float16)
    
    target[..., :5] = sparse_sample["base_grid"]
    shapes = sparse_sample["shapes"]
    indices = sparse_sample["indices"]
    if len(indices) > 0:
        for i in range(len(indices)):
            y, x, k = indices[i]
            target[y, x, k, 5:5+S2] = shapes[i]
    target[..., -1] = sparse_sample["background_map"].unsqueeze(-1)
    
    # Save image and target
    image_path = os.path.join(output_dir, f"mosaic_{idx:03d}_img.pt")
    target_path = os.path.join(output_dir, f"mosaic_{idx:03d}_tgt.pt")
    
    # Store image as float32 for precision, target as float16
    torch.save(sparse_sample["image"].squeeze(0), image_path)
    torch.save(target, target_path)
    print(f"✅ Saved Mosaic {idx} to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Pregenerate SCA Mosaics for high-speed sampling")
    parser.add_argument("--num", type=int, default=10, help="Number of full SCAs to generate")
    parser.add_argument("--config", default="config/config.yaml")
    
    args = parser.parse_args()
    config = load_config(args.config)
    data_cfg = config["data_params"]
    
    output_dir = os.path.join(config["curriculum"]["stage0"]["data_dir"], "mosaics")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    params = {
        "min_stars": data_cfg["min_stars"],
        "max_stars": data_cfg["max_stars"],
        "max_capacity_per_cell": data_cfg["max_capacity_per_cell"],
        "shape_size": data_cfg.get("shape_size", 9)
    }
    
    for i in range(args.num):
        generate_mosaic(i, output_dir, params)

if __name__ == "__main__":
    main()
