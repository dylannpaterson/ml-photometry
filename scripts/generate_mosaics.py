import argparse
import os
import torch
import numpy as np
from src.data.stage0_gaussian import GaussianPretrainingProvider
from src.cloud.config_utils import load_config
import shutil

def generate_mosaic(idx, output_dir, params, mosaic_size, cell_size):
    """Generates a large mosaic and saves it compactly as dense arrays."""
    training_size = params['image_size']
    area_ratio = (mosaic_size / training_size)**2
    
    sca_min_stars = int(params['min_stars'] * area_ratio * 0.8)
    sca_max_stars = int(params['max_stars'] * area_ratio * 0.8)
    
    provider = GaussianPretrainingProvider(
        min_stars=sca_min_stars,
        max_stars=sca_max_stars,
        image_size=mosaic_size,
        max_capacity_per_cell=params['max_capacity_per_cell'],
        shape_size=params['shape_size']
    )
    provider.cell_size = cell_size
    provider.grid_size = mosaic_size // cell_size
    
    print(f"Generating Mosaic {idx} ({mosaic_size}x{mosaic_size}, cell_size={cell_size}, approx {sca_max_stars} stars)...")
    sparse_sample = provider.generate_chunk()
    
    # 1. Save Dense Image (RAW PHOTONS for GaussianMosaicDataset to stretch on the fly)
    image_path = os.path.join(output_dir, f"mosaic_{idx:03d}_img.npy")
    np.save(image_path, sparse_sample["raw_image"].numpy().astype(np.float32))
    
    # 2. Redensify Target Grid (Pre-compute EVERYTHING)
    base_grid = sparse_sample["base_grid"] # [G, G, K, 5]
    # IMPORTANT: generate_chunk returns background_map which is ALREADY STRETCHED.
    # We need to save the ABSOLUTE PHYSICAL PHOTONS for the mosaic to be robust.
    # We'll calculate it from smooth_bg + unresolved_img inside generate_chunk? 
    # No, let's just make generate_chunk return it.
    bg_linear_abs = sparse_sample.get("bg_linear_abs", None)
    if bg_linear_abs is None:
        # Fallback if not updated yet: 
        # (This is why we centralize!)
        pass

    G = provider.grid_size
    K = provider.K
    S2 = params['shape_size']**2
    
    # Dense Target Shape: [G, G, (K * (5 + S2)) + 1]
    star_grid = np.zeros((G, G, K, 5 + S2), dtype=np.float32)
    star_grid[..., :5] = base_grid.numpy()
    
    indices = sparse_sample["indices"]
    shapes = sparse_sample["shapes"]
    if len(indices) > 0:
        for i in range(len(indices)):
            y, x, k = indices[i]
            star_grid[y, x, k, 5:5+S2] = shapes[i].numpy()
            
    # Flatten K and Append Background
    flattened_stars = star_grid.reshape(G, G, -1)
    
    # We want absolute linear BG here
    # I will update generate_chunk to return bg_linear_grid
    bg_map = sparse_sample["bg_linear_grid"]
    dense_target = np.concatenate([flattened_stars, bg_map.numpy()[..., np.newaxis]], axis=-1)
            
    target_path = os.path.join(output_dir, f"mosaic_{idx:03d}_target.npy")
    np.save(target_path, dense_target)
    
    print(f"✅ Saved Dense Mosaic {idx} (Image + Target)")

def main():
    parser = argparse.ArgumentParser(description="Pregenerate Compact Mosaics")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--num", type=int, default=None, help="Override number of mosaics to generate")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    stage_key = f"stage{args.stage}"
    stage_cfg = config["curriculum"][stage_key]
    data_cfg = config["data_params"]
    
    mos_cfg = stage_cfg.get("mosaic_params", {"num_mosaics": 10, "mosaic_size": 4088})
    num_mosaics = mos_cfg["num_mosaics"]
    mosaic_size = mos_cfg["mosaic_size"]
    cell_size = stage_cfg.get("cell_size", 4)
    
    output_dir = os.path.join(stage_cfg["data_dir"], "mosaics")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    params = {
        "min_stars": data_cfg["min_stars"],
        "max_stars": data_cfg["max_stars"],
        "image_size": data_cfg["image_size"],
        "max_capacity_per_cell": data_cfg["max_capacity_per_cell"],
        "shape_size": data_cfg.get("shape_size", 9)
    }
    
    for i in range(num_mosaics):
        generate_mosaic(i, output_dir, params, mosaic_size, cell_size)

if __name__ == "__main__":
    main()
