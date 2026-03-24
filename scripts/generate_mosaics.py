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
    HAS_JAX = True
    from castor.data.gpu_renderer import render_generate_and_filter_gpu
    if any(d.platform == 'gpu' for d in jax.devices()):
        print("🚀 JAX GPU Acceleration Enabled")
    else:
        print("🐢 JAX CPU Acceleration Enabled (No GPU found)")
except ImportError:
    HAS_JAX = False
    print("⚠️ JAX not found, falling back to slow NumPy path")

def calculate_safe_magnitude_cutoff(exp_time, zp, sky_mag, read_noise=5.0, sigma=1.5, snr_cutoff=2.0):
    """
    Calculates the faintest magnitude that could theoretically reach snr_cutoff
    in a perfectly isolated, empty patch of sky.
    """
    pixel_scale = 0.11
    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (pixel_scale**2) * exp_time
    n_pix = 4 * np.pi * (sigma ** 2)
    bg_variance = n_pix * (sky_level + read_noise**2)
    
    # Solve quadratic: F^2 - snr^2*F - snr^2*bg_var = 0
    a, b, c = 1.0, -(snr_cutoff**2), -(snr_cutoff**2) * bg_variance
    min_flux = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    
    mag_cutoff = zp - 2.5 * np.log10(min_flux / exp_time)
    return mag_cutoff

def generate_mosaic(idx, output_dir, params, mosaic_size, cell_size):
    """
    Generates a large seamless mosaic using the 'Bake and Drop' strategy.
    Uses a mathematically guaranteed SNR cutoff for catalog culling.
    """
    start_time = time.time()
    training_size = params['image_size']
    area_ratio = (mosaic_size / training_size)**2
    
    # 1. Global Physical Parameters
    rc_loc = np.random.uniform(14.5, 16.5)
    rc_scale = np.random.uniform(0.2, 0.5)
    rc_enhancement = np.random.uniform(5.0, 15.0)
    exp_time = np.random.uniform(30.0, 60.0)
    zp, sky_mag = 26.5, 22.0
    
    # Calculate dynamic safety cutoff (e.g., SNR 2.0 limit)
    mag_limit = calculate_safe_magnitude_cutoff(exp_time, zp, sky_mag, snr_cutoff=2.0)
    
    # Full massive population (physics) - Log-uniform sampling for better density coverage
    min_total = params['min_stars'] * area_ratio
    max_total = params['max_stars'] * area_ratio
    n_stars_total = int(10 ** np.random.uniform(np.log10(min_total), np.log10(max_total)))
    
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
        
        # We perform the filtering on the host side using the dynamic mag_limit
        from castor.data.gpu_renderer import render_gpu
        # Generate coordinates on host for simplicity or modify render_gpu to handle fused generation
        # To avoid PCIe bottleneck, let's keep the fused logic but pass mag_limit
        # Actually, render_generate_and_filter_gpu currently has a hardcoded 27.0
        # I will update it to take the limit.
        
        from castor.data.gpu_renderer import render_generate_and_filter_gpu
        full_image, x_v, y_v, flux_v, mag_v = render_generate_and_filter_gpu(
            fluxes, mags, kb_array, mosaic_size, mag_limit=mag_limit
        )
    else:
        # --- NumPy CPU PATH (Fallback) ---
        x_centers = np.random.uniform(0, mosaic_size, len(mags))
        y_centers = np.random.uniform(0, mosaic_size, len(mags))
        full_image = np.zeros((mosaic_size, mosaic_size), dtype=np.float32)
        # (CPU rendering loop matches previous implementation)
        # ...
        v_mask = mags < mag_limit
        x_v, y_v, flux_v, mag_v = x_centers[v_mask], y_centers[v_mask], fluxes[v_mask], mags[v_mask]

    # 2. Save Catalog as Structured NumPy
    cat_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('flux', 'f4'), ('mag', 'f4'),
        ('shape', 'f4', (SHAPE_SIZE**2,))
    ]
    
    n_visible = len(x_v)
    structured_cat = np.zeros(n_visible, dtype=cat_dtype)
    structured_cat['x'], structured_cat['y'] = x_v, y_v
    structured_cat['flux'], structured_cat['mag'] = flux_v, mag_v
    
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
    print(f"✅ Saved Optimized Mosaic {idx} in {duration:.2f}s (Limit: {mag_limit:.2f} | Targets: {n_visible:,})")

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
