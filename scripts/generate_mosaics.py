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

def generate_elliptical_psf_library(num_psfs=100, grid_size=9, sigma=1.5):
    """
    Generates a library of varied elliptical Gaussians to simulate spatially varying PSFs.
    Each PSF is a flattened grid_size x grid_size array.
    """
    library = np.zeros((num_psfs, grid_size * grid_size), dtype=np.float32)
    half = grid_size // 2
    y, x = np.meshgrid(np.arange(grid_size) - half, np.arange(grid_size) - half, indexing='ij')
    
    for i in range(num_psfs):
        # vary ellipticity and position angle
        q = np.random.uniform(0.7, 1.0) # axis ratio
        theta = np.random.uniform(0, np.pi) # position angle
        
        # rotation matrix
        cos, sin = np.cos(theta), np.sin(theta)
        xp = x * cos + y * sin
        yp = -x * sin + y * cos
        
        # elliptical gaussian
        psf = np.exp(-(xp**2 / (2 * sigma**2) + yp**2 / (2 * (sigma * q)**2)))
        psf /= psf.sum()
        library[i] = psf.flatten()
        
    return library

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
    lf_gamma = np.random.uniform(0.25, 0.35)
    exp_time = np.random.uniform(30.0, 60.0)
    zp, sky_mag = 26.5, 22.0
    
    # Calculate dynamic safety cutoff (e.g., SNR 2.0 limit)
    mag_limit = calculate_safe_magnitude_cutoff(exp_time, zp, sky_mag, snr_cutoff=2.0)
    
    # Full massive population (physics) - Log-uniform sampling for better density coverage
    min_total = params['min_stars'] * area_ratio
    max_total = params['max_stars'] * area_ratio
    n_stars_total = int(10 ** np.random.uniform(np.log10(min_total), np.log10(max_total)))
    
    print(f"Generating Global Catalog for Mosaic {idx} ({n_stars_total:,} stars)...")
    mags = sample_bulge_magnitudes(n_stars_total, rc_loc, rc_scale, rc_enhancement, m_min=12.0, m_max=32.0, gamma=lf_gamma)
    fluxes = exp_time * (10 ** (-0.4 * (mags - zp)))
    
    # -------------------------------------------------------------
    # NEW: Generate the library FIRST so the renderer can use it
    # -------------------------------------------------------------
    N_LIBRARY_PSFS = 100
    shape_size = params['shape_size']
    psf_library = generate_elliptical_psf_library(num_psfs=N_LIBRARY_PSFS, grid_size=shape_size)
    
    # Reshape the flattened library to (N, H, W) for JAX 2D convolutions
    kb_array = psf_library.reshape(N_LIBRARY_PSFS, shape_size, shape_size)
    
    if HAS_JAX:
        from castor.data.gpu_renderer import render_generate_and_filter_gpu
        # Pass the elliptical library directly to the GPU renderer
        full_image, x_v, y_v, psf_indices, flux_v, mag_v = render_generate_and_filter_gpu(
            fluxes, mags, kb_array, mosaic_size, mag_limit=mag_limit
        )
    else:
        # --- NumPy CPU PATH (Fallback) ---
        x_centers = np.random.uniform(0, mosaic_size, len(mags))
        y_centers = np.random.uniform(0, mosaic_size, len(mags))
        all_psf_indices = np.random.randint(0, N_LIBRARY_PSFS, size=len(mags))
        
        full_image = np.zeros((mosaic_size, mosaic_size), dtype=np.float32)
        v_mask = mags < mag_limit
        x_v, y_v = x_centers[v_mask], y_centers[v_mask]
        flux_v, mag_v = fluxes[v_mask], mags[v_mask]
        psf_indices = all_psf_indices[v_mask]
        
        # Very slow loop for CPU rendering fallback
        for x, y, f, p_idx in zip(x_v, y_v, flux_v, psf_indices):
            ix, iy = int(x), int(y)
            if 0 <= ix < mosaic_size and 0 <= iy < mosaic_size:
                # Add central pixel (a rough approximation for CPU fallback)
                full_image[iy, ix] += f 

    # 2. Save Catalog as Structured NumPy
    cat_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('flux', 'f4'), ('mag', 'f4'),
        ('snr', 'f4'), ('comp', 'f4'),
        ('psf_index', 'i4') 
    ]
    
    n_visible = len(x_v)
    structured_cat = np.zeros(n_visible, dtype=cat_dtype)
    structured_cat['x'], structured_cat['y'] = x_v, y_v
    structured_cat['flux'], structured_cat['mag'] = flux_v, mag_v
    
    # Slot the exact indices used by the renderer into the catalog
    structured_cat['psf_index'] = psf_indices

    # --- PRE-CALCULATE SNR and COMP (Moved from DataLoader) ---
    print(f"Pre-calculating SNR and Completeness for {n_visible:,} stars...")
    pixel_scale = 0.11
    sigma_fixed = 1.5
    n_pix = 4 * np.pi * (sigma_fixed ** 2)
    psf_peak = 1.0 / (2 * np.pi * sigma_fixed**2)
    min_snr = 5.0 # Project Standard
    
    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (pixel_scale**2) * exp_time
    
    # Local background from the rendered mosaic
    ly_idx = np.clip(y_v.astype(np.int32), 0, mosaic_size - 1)
    lx_idx = np.clip(x_v.astype(np.int32), 0, mosaic_size - 1)
    total_local_light = full_image[ly_idx, lx_idx]
    
    local_background = np.maximum(0, total_local_light - (flux_v * psf_peak))
    noise_variance = flux_v + n_pix * (sky_level + local_background + 25.0) # read_noise=5.0
    snrs = flux_v / np.sqrt(np.maximum(1.0, noise_variance))
    structured_cat['snr'] = snrs
    
    # Global Spline Fitting for Completeness
    survived = snrs >= min_snr
    if len(mag_v) < 10 or mag_v.min() >= mag_v.max() - 1e-3:
        comps = survived.astype(np.float32)
    else:
        m_min, m_max = mag_v.min(), mag_v.max()
        bins = np.linspace(m_min, m_max, 25)
        counts_total, _ = np.histogram(mag_v, bins=bins)
        counts_survived, _ = np.histogram(mag_v[survived], bins=bins)
        
        valid = counts_total > 0
        if valid.sum() < 4:
            comps = survived.astype(np.float32)
        else:
            bin_comp = counts_survived[valid] / (counts_total[valid] + 1e-9)
            bin_centers = ((bins[:-1] + bins[1:]) / 2)[valid]
            comps = np.interp(mag_v, bin_centers, bin_comp, left=1.0, right=0.0).astype(np.float32)
    
    structured_cat['comp'] = comps

    # NEW: Pre-sort the catalog by Y-coordinate so the dataloader doesn't have to
    print("Sorting catalog for fast spatial queries...")
    sort_idx = np.argsort(structured_cat['y'])
    structured_cat = structured_cat[sort_idx]

    # 3. Save Files
    image_path = os.path.join(output_dir, f"mosaic_{idx:03d}_img.npy")
    cat_path = os.path.join(output_dir, f"mosaic_{idx:03d}_cat.npy")
    meta_path = os.path.join(output_dir, f"mosaic_{idx:03d}_meta.npy")
    lib_path = os.path.join(output_dir, f"mosaic_{idx:03d}_psf_lib.npy")
    
    np.save(image_path, full_image)
    np.save(cat_path, structured_cat)
    np.save(meta_path, np.array([exp_time, zp, sky_mag], dtype=np.float32))
    np.save(lib_path, psf_library)
    
    duration = time.time() - start_time
    print(f"✅ Saved Optimized Mosaic {idx} in {duration:.2f}s (Limit: {mag_limit:.2f} | Targets: {n_visible:,})")
    
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
