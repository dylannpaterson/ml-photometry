import argparse
import os
import torch
import numpy as np
import pandas as pd
from castor.data.stage0_gaussian import GaussianPretrainingProvider
from castor.cloud.config_utils import load_config
import shutil
from castor.constants import DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL, SHAPE_SIZE

def generate_mosaic(idx, output_dir, params, mosaic_size, cell_size):
    """Generates a large mosaic and saves it compactly as physics image + catalog."""
    training_size = params['image_size']
    area_ratio = (mosaic_size / training_size)**2
    
    # Scale stars for the full mosaic area
    sca_min_stars = int(params['min_stars'] * area_ratio)
    sca_max_stars = int(params['max_stars'] * area_ratio)
    
    provider = GaussianPretrainingProvider(
        min_stars=sca_min_stars,
        max_stars=sca_max_stars,
        image_size=mosaic_size,
        max_capacity_per_cell=params['max_capacity_per_cell'],
        shape_size=params['shape_size']
    )
    provider.cell_size = cell_size
    provider.grid_size = mosaic_size // cell_size
    
    # NEW: Draw Line-of-Sight Parameters for the Bulge
    rc_loc = np.random.uniform(14.5, 16.5)
    rc_scale = np.random.uniform(0.2, 0.5)
    rc_fraction = np.random.uniform(0.05, 0.20)
    
    # Instrument params (Roman wide-band proxy)
    exp_time = np.random.uniform(20.0, 100.0)
    zp = 26.5
    sky_mag = 22.0
    
    print(f"Generating Mosaic {idx} ({mosaic_size}x{mosaic_size}, approx {sca_max_stars} stars)...")
    print(f"  RC_Mag={rc_loc:.2f}, exp={exp_time:.1f}s")
    
    # Use the speed-hack rendering
    sample = provider.generate_chunk(
        rc_params=(rc_loc, rc_scale, rc_fraction),
        exp_params=(exp_time, zp, sky_mag)
    )
    
    # 1. Save Clean Physics Image (Noiseless Photons)
    image_path = os.path.join(output_dir, f"mosaic_{idx:03d}_img.npy")
    np.save(image_path, sample["physics_image"].numpy().astype(np.float32))
    
    # 2. Save Rich Parquet Catalog (Astrophysics + Metadata)
    catalog = sample["catalog"]
    catalog['exp_time'] = exp_time
    catalog['zp'] = zp
    catalog['sky_mag'] = sky_mag
    
    cat_path = os.path.join(output_dir, f"mosaic_{idx:03d}_catalog.parquet")
    catalog.to_parquet(cat_path, index=False)
    
    print(f"✅ Saved Macro-Sparse Mosaic {idx}")

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
    
    # Standardize on the new Medium-Mosaic Strategy
    num_mosaics = 100 # Default for local
    if args.num is not None:
        num_mosaics = args.num
        
    mosaic_size = 1024 # 1/16th area of full SCA
    cell_size = stage_cfg.get("cell_size", DEFAULT_CELL_SIZE)
    
    output_dir = os.path.join(stage_cfg["data_dir"], "mosaics")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    params = {
        "min_stars": data_cfg["min_stars"],
        "max_stars": data_cfg["max_stars"],
        "image_size": data_cfg["image_size"], # 256
        "max_capacity_per_cell": data_cfg["max_capacity_per_cell"],
        "shape_size": data_cfg.get("shape_size", SHAPE_SIZE)
    }
    
    # We need to scale the star density correctly for the 1024 canvas
    # The config min/max is usually for 256x256. 
    # Area ratio is (1024/256)^2 = 16.
    
    for i in range(num_mosaics):
        generate_mosaic(i, output_dir, params, mosaic_size, cell_size)

if __name__ == "__main__":
    main()
