import argparse
import os
import torch
import numpy as np
import time
from castor.data.stage0_gaussian import generate_mosaic_data, generate_field_realistic_psf_library, _compute_eigen_psfs
from castor.cloud.config_utils import load_config
from castor.constants import SHAPE_SIZE, N_PCA_COMPONENTS

def main():
    parser = argparse.ArgumentParser(description="Thin wrapper for Mosaic Generation")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--psf_library", type=str, default=None)
    
    args = parser.parse_args()
    cfg = load_config(args.config)
    stage_key = f"stage{args.stage}"
    stage_cfg = cfg["curriculum"][stage_key]
    
    # 1. Load or Generate Master PSF Library
    if args.psf_library and os.path.exists(args.psf_library):
        master_data = torch.load(args.psf_library, map_location='cpu', weights_only=False)
        master_psf_data = (master_data['eigen_psfs'], master_data['weights_lib'], master_data['mean_psf'])
    else:
        kb_array = generate_field_realistic_psf_library(num_psfs=100, grid_size=SHAPE_SIZE)
        master_psf_data = _compute_eigen_psfs(kb_array, n_components=N_PCA_COMPONENTS)
        
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
        full_image, structured_cat, meta, psf_lib_save = generate_mosaic_data(
            mosaic_size, params, master_psf_data
        )
        
        # 3. Save Outputs
        base_name = f"mosaic_{i:03d}"
        np.save(os.path.join(out, f"{base_name}_img.npy"), full_image)
        np.save(os.path.join(out, f"{base_name}_cat.npy"), structured_cat)
        np.save(os.path.join(out, f"{base_name}_meta.npy"), meta)
        np.save(os.path.join(out, f"{base_name}_psf_lib.npy"), psf_lib_save)
        
        print(f"✅ {base_name} done in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
