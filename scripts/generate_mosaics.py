"""
Mosaic generation script for the Castor pipeline.

This script acts as a wrapper for generating simulated astronomical mosaics 
using parameters defined in the curriculum configuration.
"""

import argparse
import os
import torch
import numpy as np
import time
from castor.data.stage0_gaussian import generate_mosaic_data
from castor.cloud.config_utils import load_config
from castor.constants import SHAPE_SIZE

def main():
    """
    Main entry point for mosaic generation.
    """
    parser = argparse.ArgumentParser(description="Thin wrapper for Mosaic Generation")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    
    args = parser.parse_args()
    cfg = load_config(args.config)
    stage_key = f"stage{args.stage}"
    stage_cfg = cfg["curriculum"][stage_key]
    
    num = args.num if args.num else stage_cfg["mosaic_params"]["num_mosaics"]
    out = args.output_dir if args.output_dir else os.path.join(stage_cfg["data_dir"], "mosaics")
    os.makedirs(out, exist_ok=True)
    
    params = {
        "min_stars": cfg["data_params"]["min_stars"], 
        "max_stars": cfg["data_params"]["max_stars"], 
        "image_size": cfg["data_params"]["image_size"]
    }
    
    mosaic_size = stage_cfg["mosaic_params"]["mosaic_size"]
    
    # 2. Generation Loop
    for i in range(num):
        start_time = time.time()
        full_image, bg_image, structured_cat, meta, psf_1x = generate_mosaic_data(
            mosaic_size, params
        )
        
        # 3. Save Outputs
        base_name = f"mosaic_{i:03d}"
        np.save(os.path.join(out, f"{base_name}_img.npy"), full_image)
        np.save(os.path.join(out, f"{base_name}_bg.npy"), bg_image)
        np.save(os.path.join(out, f"{base_name}_cat.npy"), structured_cat)
        np.save(os.path.join(out, f"{base_name}_meta.npy"), meta)
        np.save(os.path.join(out, f"{base_name}_psf.npy"), psf_1x)
        
        print(f"✅ {base_name} done in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
