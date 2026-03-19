import argparse
import os
import torch
import numpy as np
from src.data.stage0_gaussian import GaussianPretrainingProvider
from src.cloud.config_utils import load_config
import shutil

def generate_mosaic(idx, output_dir, params):
    """Generates a full SCA-sized mosaic (4088x4088) and saves it compactly."""
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
    sparse_sample = provider.generate_chunk()
    
    # Compact Storage: Save image as .npy and sparse data as .pt
    image_path = os.path.join(output_dir, f"mosaic_{idx:03d}_img.npy")
    sparse_path = os.path.join(output_dir, f"mosaic_{idx:03d}_sparse.pt")
    
    # Save image for memory mapping
    np.save(image_path, sparse_sample["image"].squeeze(0).numpy())
    
    # Save sparse data components directly (very small)
    sparse_data = {
        "base_grid": sparse_sample["base_grid"],
        "background_map": sparse_sample["background_map"],
        "shapes": sparse_sample["shapes"],
        "indices": sparse_sample["indices"]
    }
    torch.save(sparse_data, sparse_path)
    print(f"✅ Saved Compact Mosaic {idx}")

def main():
    parser = argparse.ArgumentParser(description="Pregenerate Compact SCA Mosaics")
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
