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

# GPU/JAX Acceleration
try:
    import jax
    # We use JAX even on CPU because it's faster and allows the same 'Bake and Drop' logic
    HAS_JAX = True
    from castor.data.gpu_renderer import render_generate_and_filter_gpu
    # Check if actually using GPU for logging
    if any(d.platform == 'gpu' for d in jax.devices()):
        print("🚀 JAX GPU Acceleration Enabled")
    else:
        print("🐢 JAX CPU Acceleration Enabled (No GPU found)")
except ImportError:
    HAS_JAX = False
    print("⚠️ JAX not found, falling back to slow NumPy path")

def generate_mosaic(idx, output_dir, params, mosaic_size, cell_size):
    """
    Generates a large seamless mosaic using the 'Bake and Drop' strategy.
    Simulates full physics (8M stars) in the pixels, but culls the catalog 
    to save RAM and speed up training.
    """
    start_time = time.time()
    training_size = params['image_size'] # 256
    area_ratio = (mosaic_size / training_size)**2 # 16
    
    # 1. Global Physical Parameters
    rc_loc = np.random.uniform(14.5, 16.5)
    rc_scale = np.random.uniform(0.2, 0.5)
    rc_enhancement = np.random.uniform(5.0, 15.0)
    exp_time = np.random.uniform(30.0, 60.0)
    zp, sky_mag = 26.5, 22.0
    
    # Full massive population (physics)
    n_stars_total = int(np.random.uniform(params['min_stars'], params['max_stars']) * area_ratio)
    
    print(f"Generating Global Catalog for Mosaic {idx} ({n_stars_total:,} stars)...")
    mags = sample_bulge_magnitudes(n_stars_total, rc_loc, rc_scale, rc_enhancement, m_min=12.0, m_max=32.0)
    fluxes = exp_time * (10 ** (-0.4 * (mags - zp)))
    
    provider = GaussianPretrainingProvider(image_size=training_size)
    
    if HAS_JAX:
        # --- JAX PATH: Render entire mosaic at once with full physics ---
        kb_array = np.zeros((16, provider.kernel_size, provider.kernel_size), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                kb_array[j * 4 + i] = provider.kernel_bank[(i, j)]
        
        # This renders ALL stars into full_image, but returns coordinates for only visible ones (mag < 27.5)
        full_image, x_v, y_valid, flux_v, mag_v = render_generate_and_filter_gpu(
            fluxes, mags, kb_array, mosaic_size
        )
    else:
        # --- NumPy CPU PATH (Slow) ---
        x_centers = np.random.uniform(0, mosaic_size, len(mags))
        y_centers = np.random.uniform(0, mosaic_size, len(mags))
        
        full_image = np.zeros((mosaic_size, mosaic_size), dtype=np.float32)
        # For simplicity in fallback, we loop phases (matching original logic)
        px = np.clip(np.floor((x_centers - np.floor(x_centers)) * provider.n_sub).astype(int), 0, provider.n_sub - 1)
        py = np.clip(np.floor((y_centers - np.floor(y_centers)) * provider.n_sub).astype(int), 0, provider.n_sub - 1)
        ix0, iy0 = np.floor(x_centers).astype(int), np.floor(y_centers).astype(int)
        
        for i in range(provider.n_sub):
            for j in range(provider.n_sub):
                p_mask = (px == i) & (py == j)
                if p_mask.any():
                    phase_map, _, _ = np.histogram2d(
                        iy0[p_mask], ix0[p_mask], 
                        bins=[mosaic_size, mosaic_size], 
                        range=[[0, mosaic_size], [0, mosaic_size]], 
                        weights=fluxes[p_mask]
                    )
                    full_image += fftconvolve(phase_map, provider.kernel_bank[(i, j)], mode='same').astype(np.float32)
        
        # Cull for catalog
        v_mask = mags < 27.5
        x_v, y_valid, flux_v, mag_v = x_centers[v_mask], y_centers[v_mask], fluxes[v_mask], mags[v_mask]

    # 2. Save Catalog as Structured NumPy for MMAP
    cat_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('flux', 'f4'), ('mag', 'f4'),
        ('shape', 'f4', (SHAPE_SIZE**2,))
    ]
    
    n_visible = len(x_v)
    structured_cat = np.zeros(n_visible, dtype=cat_dtype)
    structured_cat['x'] = x_v
    structured_cat['y'] = y_valid
    structured_cat['flux'] = flux_v
    structured_cat['mag'] = mag_v
    
    # Pre-compute target PSF
    half = SHAPE_SIZE // 2
    grid_range = np.arange(-half, half + 1)
    sy, sx = np.meshgrid(grid_range, grid_range, indexing='ij')
    psf_flat = np.exp(-(sx**2 + sy**2) / (2 * 1.5**2))
    psf_flat = (psf_flat / psf_flat.sum()).astype(np.float32).flatten()
    structured_cat['shape'] = psf_flat

    # 3. Save Files
    image_path = os.path.join(output_dir, f"mosaic_{idx:03d}_img.npy")
    cat_path = os.path.join(output_dir, f"mosaic_{idx:03d}_cat.npy")
    meta_path = os.path.join(output_dir, f"mosaic_{idx:03d}_meta.npy")
    
    np.save(image_path, full_image)
    np.save(cat_path, structured_cat)
    np.save(meta_path, np.array([exp_time, zp, sky_mag], dtype=np.float32))
    
    duration = time.time() - start_time
    print(f"✅ Saved Optimized Mosaic {idx} in {duration:.2f}s ({n_visible:,} potential targets)")

def main():
    parser = argparse.ArgumentParser(description="Pregenerate Optimized Compact Mosaics")
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
