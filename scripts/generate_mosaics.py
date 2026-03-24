import argparse
import os
import torch
import numpy as np
import pandas as pd
from castor.data.stage0_gaussian import GaussianPretrainingProvider, sample_bulge_magnitudes
from castor.cloud.config_utils import load_config
import shutil
import time
from castor.constants import DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL, SHAPE_SIZE
from scipy.signal import fftconvolve

def generate_mosaic(idx, output_dir, params, mosaic_size, cell_size):
    """Generates a large seamless mosaic by generating a global catalog and chunked rendering."""
    start_time = time.time()
    training_size = params['image_size'] # 256
    area_ratio = (mosaic_size / training_size)**2 # 16
    
    # 1. Global Physical Parameters
    rc_loc = np.random.uniform(14.5, 16.5)
    rc_scale = np.random.uniform(0.2, 0.5)
    rc_enhancement = np.random.uniform(5.0, 15.0)
    exp_time = np.random.uniform(30.0, 60.0)
    zp, sky_mag = 26.5, 22.0
    
    n_stars_total = int(np.random.uniform(params['min_stars'], params['max_stars']) * area_ratio)
    
    print(f"Generating Global Catalog for Mosaic {idx} ({n_stars_total:,} stars)...")
    mags = sample_bulge_magnitudes(n_stars_total, rc_loc, rc_scale, rc_enhancement, m_min=12.0, m_max=32.0)
    fluxes = exp_time * (10 ** (-0.4 * (mags - zp)))
    x_centers = np.random.uniform(0, mosaic_size, len(mags))
    y_centers = np.random.uniform(0, mosaic_size, len(mags))
    
    # 2. Render the physics image in 4x4 chunks (with buffer to handle PSF overlap)
    full_image = np.zeros((mosaic_size, mosaic_size), dtype=np.float32)
    n_side = mosaic_size // training_size
    buffer = 20 # Pixels
    
    provider = GaussianPretrainingProvider(image_size=training_size)
    
    for iy in range(n_side):
        for ix in range(n_side):
            y0, y1 = iy * training_size, (iy + 1) * training_size
            x0, x1 = ix * training_size, (ix + 1) * training_size
            
            mask = (x_centers > x0 - buffer) & (x_centers < x1 + buffer) & \
                   (y_centers > y0 - buffer) & (y_centers < y1 + buffer)
            
            cx, cy, cf = x_centers[mask] - x0, y_centers[mask] - y0, fluxes[mask]
            grid_h, grid_w = training_size + 2*buffer, training_size + 2*buffer
            chunk_signal = np.zeros((grid_h, grid_w), dtype=np.float32)
            
            px = np.clip(np.floor((cx + buffer - np.floor(cx + buffer)) * provider.n_sub).astype(int), 0, provider.n_sub - 1)
            py = np.clip(np.floor((cy + buffer - np.floor(cy + buffer)) * provider.n_sub).astype(int), 0, provider.n_sub - 1)
            ix0, iy0 = np.floor(cx + buffer).astype(int), np.floor(cy + buffer).astype(int)
            
            for i in range(provider.n_sub):
                for j in range(provider.n_sub):
                    p_mask = (px == i) & (py == j)
                    if p_mask.any():
                        phase_map, _, _ = np.histogram2d(
                            iy0[p_mask], ix0[p_mask], 
                            bins=[grid_h, grid_w], 
                            range=[[0, grid_h], [0, grid_w]], 
                            weights=cf[p_mask]
                        )
                        chunk_signal += fftconvolve(phase_map, provider.kernel_bank[(i, j)], mode='same').astype(np.float32)
            
            full_image[y0:y1, x0:x1] = chunk_signal[buffer:-buffer, buffer:-buffer]

    # 3. Filter and Save Catalog
    valid_mask = mags < 27.0
    catalog = pd.DataFrame({
        'x': x_centers[valid_mask],
        'y': y_centers[valid_mask],
        'flux': fluxes[valid_mask],
        'mag': mags[valid_mask]
    })
    
    half = SHAPE_SIZE // 2
    grid_range = np.arange(-half, half + 1)
    sy, sx = np.meshgrid(grid_range, grid_range, indexing='ij')
    psf_flat = np.exp(-(sx**2 + sy**2) / (2 * 1.5**2))
    psf_flat = (psf_flat / psf_flat.sum()).astype(np.float32).flatten()
    catalog['shape'] = [psf_flat] * len(catalog)
    
    catalog['exp_time'] = exp_time
    catalog['zp'] = zp
    catalog['sky_mag'] = sky_mag
    
    image_path = os.path.join(output_dir, f"mosaic_{idx:03d}_img.npy")
    np.save(image_path, full_image)
    
    cat_path = os.path.join(output_dir, f"mosaic_{idx:03d}_catalog.parquet")
    catalog.to_parquet(cat_path, index=False)
    
    duration = time.time() - start_time
    print(f"✅ Saved Seamless Mosaic {idx} in {duration:.2f}s ({len(catalog):,} potential targets)")

def main():
    parser = argparse.ArgumentParser(description="Pregenerate Seamless Compact Mosaics")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--num", type=int, default=None)
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    stage_key = f"stage{args.stage}"
    stage_cfg = config["curriculum"][stage_key]
    data_cfg = config["data_params"]
    
    num_mosaics = stage_cfg["mosaic_params"].get("num_mosaics", 100)
    if args.num is not None: num_mosaics = args.num
        
    mosaic_size = stage_cfg["mosaic_params"].get("mosaic_size", 1024)
    cell_size = stage_cfg.get("cell_size", DEFAULT_CELL_SIZE)
    
    output_dir = os.path.join(stage_cfg["data_dir"], "mosaics")
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    params = {
        "min_stars": data_cfg["min_stars"],
        "max_stars": data_cfg["max_stars"],
        "image_size": data_cfg["image_size"],
        "max_capacity_per_cell": data_cfg["max_capacity_per_cell"],
        "shape_size": data_cfg.get("shape_size", SHAPE_SIZE)
    }
    
    for i in range(num_mosaics):
        generate_mosaic(i, output_dir, params, mosaic_size, cell_size)

if __name__ == "__main__":
    main()
