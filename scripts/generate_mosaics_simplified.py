import argparse
import os
import torch
import numpy as np
import pandas as pd
from castor.data.stage0_gaussian import sample_bulge_magnitudes
from castor.cloud.config_utils import load_config
import time
from scipy.signal import fftconvolve
from scipy.ndimage import map_coordinates

# Enforce a 4x Upscale-Safe Odd Size (129 = 4 * 32 + 1). 
# This guarantees exact convolution centers (idx 64) and prevents 0.5 pixel parity shifts.
SHAPE_SIZE = 129  

try:
    import jax
    HAS_JAX = True
    from castor.data.gpu_renderer import render_generate_and_filter_gpu, is_gpu_available
    if is_gpu_available():
        print("🚀 JAX GPU Acceleration Enabled")
    else:
        print("🐢 JAX CPU detected, using optimized NumPy path instead")
except ImportError:
    HAS_JAX = False
    print("⚠️ JAX not found, using slow NumPy path")

def calculate_safe_magnitude_cutoff(exp_time, zp, sky_mag, read_noise=5.0, sigma=1.5, snr_cutoff=2.0):
    pixel_scale = 0.11
    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (pixel_scale**2) * exp_time
    n_pix = 4 * np.pi * (sigma ** 2)
    bg_variance = n_pix * (sky_level + read_noise**2)
    a, b, c = 1.0, -(snr_cutoff**2), -(snr_cutoff**2) * bg_variance
    min_flux = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    return zp - 2.5 * np.log10(min_flux / exp_time)

def generate_field_realistic_psf_library(num_psfs=1, grid_size=129):
    """Generates a simple, single centralized core PSF."""
    print(f"📡 Generating Master OPTICAL PSF Library ({num_psfs} PSFs)...")
    library = np.zeros((num_psfs, grid_size, grid_size), dtype=np.float32)
    half = grid_size // 2

    y, x = np.meshgrid(np.arange(grid_size) - half, np.arange(grid_size) - half, indexing='ij')
    for i in range(num_psfs):
        s_opt = 0.45 
        opt_core = np.exp(-(x**2 + y**2) / (2 * s_opt**2)) # Simple clean Gaussian
        psf = opt_core / (opt_core.sum() + 1e-9)
        library[i] = psf
        
    return library

def scatter_bincount(mosaic_size, flat_indices, weights):
    return np.bincount(flat_indices, weights=weights, minlength=mosaic_size*mosaic_size).reshape(mosaic_size, mosaic_size)

def generate_mosaic(idx, output_dir, params, mosaic_size, single_psf):
    start_time = time.time()
    area_ratio = (mosaic_size / params['image_size'])**2
    exp_time, zp, sky_mag = np.random.uniform(30.0, 60.0), 26.5, 22.0
    mag_limit = calculate_safe_magnitude_cutoff(exp_time, zp, sky_mag, snr_cutoff=2.0)
    
    n_stars_total = int(10 ** np.random.uniform(np.log10(params['min_stars'] * area_ratio), np.log10(params['max_stars'] * area_ratio)))
    print(f"📦 Mosaic {idx}: Rendering {n_stars_total:,} stars...")
    mags = sample_bulge_magnitudes(n_stars_total, np.random.uniform(14.5, 16.5), np.random.uniform(0.2, 0.5), np.random.uniform(5.0, 15.0), m_min=12.0, m_max=32.0, gamma=np.random.uniform(0.25, 0.35))
    fluxes = exp_time * (10 ** (-0.4 * (mags - zp)))
    
    use_jax = HAS_JAX and is_gpu_available()
    if use_jax:
        # Note: render_generate_and_filter_gpu might need adjustment for the single_psf instead of library
        # For now, we'll follow the NumPy path if JAX isn't ready for this specific simplified call
        full_image, x_v, y_v, psf_v, flux_v, mag_v, weights_v = render_generate_and_filter_gpu(
            fluxes, mags, np.zeros((1, 10)), single_psf, np.zeros((10, SHAPE_SIZE, SHAPE_SIZE)), mosaic_size, mag_limit=mag_limit
        )
    else:
        px, py = np.random.uniform(0, mosaic_size, len(fluxes)), np.random.uniform(0, mosaic_size, len(fluxes))
        x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
        dx, dy = px - x0, py - y0
        valid = (x0 >= 0) & (x0 < mosaic_size-1) & (y0 >= 0) & (y0 < mosaic_size-1)
        indices = np.concatenate([y0[valid] * mosaic_size + x0[valid], y0[valid] * mosaic_size + x0[valid] + 1, (y0[valid]+1) * mosaic_size + x0[valid], (y0[valid]+1) * mosaic_size + x0[valid] + 1])
        f_v = fluxes[valid]
        vals = np.concatenate([f_v * (1-dx[valid]) * (1-dy[valid]), f_v * dx[valid] * (1-dy[valid]), f_v * (1-dx[valid]) * dy[valid], f_v * dx[valid] * dy[valid]])
        base_grid = scatter_bincount(mosaic_size, indices, vals)
        
        # Simple single convolution
        full_image = fftconvolve(base_grid, single_psf, mode='same')
        # full_image = np.maximum(0, full_image)
        
        # FOR VERIFICATION: Include all stars
        # v_mask = mags < mag_limit
        v_mask = np.ones_like(mags, dtype=bool)
        x_v, y_v, flux_v, mag_v = px[v_mask], py[v_mask], fluxes[v_mask], mags[v_mask]

    # Recalculate SNR cleanly without jitter
    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (0.11**2) * exp_time
    total_local_light = map_coordinates(full_image, [y_v, x_v], order=1, mode='nearest')
    
    k_half = SHAPE_SIZE // 2
    star_own_peak = flux_v * single_psf[k_half, k_half]
    confusion_bg = np.maximum(0, total_local_light - star_own_peak)
    eff_area = 1.0 / np.sum(single_psf**2)
    noise_variance = flux_v + eff_area * (sky_level + confusion_bg + 25.0)
    
    cat_dtype = [('x', 'f4'), ('y', 'f4'), ('flux', 'f4'), ('mag', 'f4'), ('snr', 'f4')]
    structured_cat = np.zeros(len(x_v), dtype=cat_dtype)
    structured_cat['x'], structured_cat['y'], structured_cat['flux'], structured_cat['mag'] = x_v, y_v, flux_v, mag_v
    structured_cat['snr'] = flux_v / np.sqrt(np.maximum(1.0, noise_variance))
    structured_cat = structured_cat[np.argsort(structured_cat['y'])]

    np.save(os.path.join(output_dir, f"mosaic_{idx:03d}_img.npy"), full_image)
    np.save(os.path.join(output_dir, f"mosaic_{idx:03d}_cat.npy"), structured_cat)
    np.save(os.path.join(output_dir, f"mosaic_{idx:03d}_psf_lib.npy"), single_psf.reshape(1, -1))
    
    print(f"✅ Mosaic {idx} done in {time.time() - start_time:.2f}s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    stage_key = f"stage{args.stage}"
    stage_cfg = cfg["curriculum"][stage_key]
    
    # Bypass complex initialization and use just the single master PSF
    kb_array = generate_field_realistic_psf_library(num_psfs=1, grid_size=SHAPE_SIZE)
    single_psf = kb_array[0]
    
    num = args.num if args.num else stage_cfg["mosaic_params"]["num_mosaics"]
    out = args.output_dir if args.output_dir else os.path.join(stage_cfg["data_dir"], "mosaics")
    os.makedirs(out, exist_ok=True)
    
    params = {"min_stars": cfg["data_params"]["min_stars"], "max_stars": cfg["data_params"]["max_stars"], "image_size": cfg["data_params"]["image_size"]}
    
    for i in range(num): 
        generate_mosaic(i, out, params, stage_cfg["mosaic_params"]["mosaic_size"], single_psf)

if __name__ == "__main__": 
    main()
