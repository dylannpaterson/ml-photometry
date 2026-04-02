#!/usr/bin/env python3
import os
import torch
import numpy as np
import json
import argparse
import time
from tqdm import tqdm
from castor.cloud.config_utils import load_config
from castor.data.stage0_gaussian import sample_bulge_magnitudes
from castor.constants import GLOBAL_STRETCH_SCALE
from scipy.signal import fftconvolve

# Astronomical Libraries
from astropy.io import fits
from astropy.wcs import WCS

def paczynski_magnification(t, t0, tE, u0):
    """
    Standard Paczynski microlensing magnification formula.
    """
    u = np.sqrt(u0**2 + ((t - t0) / tE)**2)
    A = (u**2 + 2) / (u * np.sqrt(u**2 + 4) + 1e-9)
    return A

def create_roman_wcs(mosaic_size, pixel_scale=0.11):
    """
    Creates a basic WCS for a Roman-like mosaic.
    """
    w = WCS(naxis=2)
    w.wcs.crpix = [mosaic_size / 2.0, mosaic_size / 2.0]
    w.wcs.cdelt = [-pixel_scale / 3600.0, pixel_scale / 3600.0]
    w.wcs.crval = [270.0, -30.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w

def generate_elliptical_psf_library(num_psfs=100, grid_size=31, sigma=1.5):
    """
    Generates a library of varied elliptical Gaussians.
    """
    library = np.zeros((num_psfs, grid_size, grid_size), dtype=np.float32)
    half = grid_size // 2
    y, x = np.meshgrid(np.arange(grid_size) - half, np.arange(grid_size) - half, indexing='ij')
    for i in range(num_psfs):
        q = np.random.uniform(0.7, 1.0)
        theta = np.random.uniform(0, np.pi)
        cos, sin = np.cos(theta), np.sin(theta)
        xp = x * cos + y * sin
        yp = -x * sin + y * cos
        psf = np.exp(-(xp**2 / (2 * sigma**2) + yp**2 / (2 * (sigma * q)**2)))
        psf /= (psf.sum() + 1e-9)
        library[i] = psf
    return library

def render_numpy(x, y, fluxes, psf_indices, kernel_bank, mosaic_size):
    """
    Robust NumPy renderer using grouped FFT convolution.
    Used for the initial static background render.
    """
    full_image = np.zeros((mosaic_size, mosaic_size), dtype=np.float32)
    num_kernels = kernel_bank.shape[0]
    
    for i in range(num_kernels):
        mask = (psf_indices == i)
        if not mask.any(): continue
        
        px, py, pf = x[mask], y[mask], fluxes[mask]
        x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
        dx, dy = px - x0, py - y0
        
        phase_map = np.zeros((mosaic_size, mosaic_size), dtype=np.float32)
        m00 = (x0 >= 0) & (x0 < mosaic_size) & (y0 >= 0) & (y0 < mosaic_size)
        m10 = (x0+1 >= 0) & (x0+1 < mosaic_size) & (y0 >= 0) & (y0 < mosaic_size)
        m01 = (x0 >= 0) & (x0 < mosaic_size) & (y0+1 >= 0) & (y0+1 < mosaic_size)
        m11 = (x0+1 >= 0) & (x0+1 < mosaic_size) & (y0+1 >= 0) & (y0+1 < mosaic_size)
        
        np.add.at(phase_map, (y0[m00], x0[m00]), pf[m00] * (1-dx[m00]) * (1-dy[m00]))
        np.add.at(phase_map, (y0[m10], x0[m10]+1), pf[m10] * dx[m10] * (1-dy[m10]))
        np.add.at(phase_map, (y0[m01]+1, x0[m01]), pf[m01] * (1-dx[m01]) * dy[m01])
        np.add.at(phase_map, (y0[m11]+1, x0[m11]+1), pf[m11] * dx[m11] * dy[m11])
        
        full_image += fftconvolve(phase_map, kernel_bank[i], mode='same')
        
    return full_image

def get_star_stamp(x, y, flux, psf, mosaic_size):
    """
    Renders a single star into a localized stamp for super-fast time series.
    Returns (stamp, y_slice, x_slice)
    """
    half = psf.shape[0] // 2
    ix, iy = int(x), int(y)
    dx, dy = x - ix, y - iy
    
    # 1. Create a slightly larger phase map for the bilinear shift
    phase_map = np.zeros((psf.shape[0] + 1, psf.shape[1] + 1), dtype=np.float32)
    # Place flux at relative center (half, half)
    phase_map[half, half] = flux * (1-dx) * (1-dy)
    phase_map[half, half+1] = flux * dx * (1-dy)
    phase_map[half+1, half] = flux * (1-dx) * dy
    phase_map[half+1, half+1] = flux * dx * dy
    
    # 2. Convolve the local phase map
    stamp = fftconvolve(phase_map, psf, mode='same')
    
    # 3. Calculate where this stamp lands in the full image
    y0, y1 = iy - half, iy - half + stamp.shape[0]
    x0, x1 = ix - half, ix - half + stamp.shape[1]
    
    # 4. Handle edge cropping
    y0_c, y1_c = max(0, y0), min(mosaic_size, y1)
    x0_c, x1_c = max(0, x0), min(mosaic_size, x1)
    
    sy0, sy1 = y0_c - y0, stamp.shape[0] - (y1 - y1_c)
    sx0, sx1 = x0_c - x0, stamp.shape[1] - (x1 - x1_c)
    
    return stamp[sy0:sy1, sx0:sx1], slice(y0_c, y1_c), slice(x0_c, x1_c)

def main():
    parser = argparse.ArgumentParser(description="Generate Roman Microlensing Time Series (FITS)")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--outdir", default="data/timeseries_test")
    parser.add_argument("--mosaic_size", type=int, default=None)
    parser.add_argument("--target_mag", type=float, default=19.0)
    parser.add_argument("--num_epochs", type=int, default=1728)
    parser.add_argument("--cadence", type=float, default=1.0)
    parser.add_argument("--t0", type=float, default=36.0)
    parser.add_argument("--tE", type=float, default=20.0)
    parser.add_argument("--u0", type=float, default=0.1)
    parser.add_argument("--center_on_t0", action="store_true")
    parser.add_argument("--format", choices=["fits", "npy"], default="fits")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data_params"]
    stage0_cfg = config["curriculum"]["stage0"]
    
    mosaic_size = args.mosaic_size or stage0_cfg["mosaic_params"].get("mosaic_size", 1024)
    n_library_psfs = data_cfg.get("n_library_psfs", 100)
    read_noise = data_cfg["physics_params"].get("read_noise", 5.0)
    pixel_scale = data_cfg["physics_params"].get("pixel_scale", 0.11)
    zp = data_cfg["physics_params"].get("zp", 26.5)
    sky_mag = data_cfg["physics_params"].get("sky_mag", 22.0)
    exp_time = (data_cfg["physics_params"]["exp_time_min"] + data_cfg["physics_params"]["exp_time_max"]) / 2
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # --- Step A: Generate Catalog ---
    print(f"🌌 Generating background catalog for {mosaic_size}x{mosaic_size} field...")
    n_stars_base = int(np.random.uniform(data_cfg['min_stars'], data_cfg['max_stars']))
    area_ratio = (mosaic_size / 256)**2
    n_stars_total = int(n_stars_base * area_ratio)
    
    rc_loc = np.random.uniform(14.5, 16.5)
    rc_scale = np.random.uniform(0.2, 0.5)
    rc_enhancement = np.random.uniform(5.0, 15.0)
    lf_gamma = np.random.uniform(0.25, 0.35)
    
    mags = sample_bulge_magnitudes(n_stars_total, rc_loc, rc_scale, rc_enhancement, m_min=12.0, m_max=32.0, gamma=lf_gamma)
    fluxes = exp_time * (10 ** (-0.4 * (mags - zp)))
    x_centers = np.random.uniform(0, mosaic_size, len(mags))
    y_centers = np.random.uniform(0, mosaic_size, len(mags))
    psf_indices = np.random.randint(0, n_library_psfs, size=len(mags))
    
    # --- Step B: Target Selection ---
    center = mosaic_size / 2
    dist_from_center = np.sqrt((x_centers - center)**2 + (y_centers - center)**2)
    central_mask = dist_from_center < (mosaic_size * 0.2)
    
    if central_mask.any():
        target_idx = np.where(central_mask)[0][np.argmin(np.abs(mags[central_mask] - args.target_mag))]
    else:
        target_idx = np.argmin(np.abs(mags - args.target_mag))
        
    print(f"🎯 Selected Target Star: Mag={mags[target_idx]:.3f} at ({x_centers[target_idx]:.1f}, {y_centers[target_idx]:.1f})")
    
    # --- Step C: Optimized Rendering Strategy ---
    kb = generate_elliptical_psf_library(num_psfs=n_library_psfs)
    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (pixel_scale**2) * exp_time
    
    # 1. Render all stars EXCEPT the target once
    print("🎬 Rendering static background field (this may take a minute)...")
    static_mask = np.ones(len(mags), dtype=bool)
    static_mask[target_idx] = False
    
    static_clean = render_numpy(x_centers[static_mask], y_centers[static_mask], 
                               fluxes[static_mask], psf_indices[static_mask], 
                               kb, mosaic_size)
    
    # 2. Pre-render the target star's footprint at baseline flux
    target_psf = kb[psf_indices[target_idx]]
    target_stamp_baseline, y_slice, x_slice = get_star_stamp(x_centers[target_idx], y_centers[target_idx], 
                                                            fluxes[target_idx], target_psf, mosaic_size)
    
    # --- Step D: Time Series Loop ---
    cadence_days = args.cadence / 24.0
    if args.center_on_t0:
        times = (args.t0 - (args.num_epochs // 2) * cadence_days) + np.arange(args.num_epochs) * cadence_days
    else:
        times = np.arange(args.num_epochs) * cadence_days
    
    # Metadata
    event_params = {
        "u0": args.u0, "tE": args.tE, "t0": args.t0,
        "target_id": int(target_idx), "target_x": float(x_centers[target_idx]),
        "target_y": float(y_centers[target_idx]), "base_mag": float(mags[target_idx]),
        "base_flux": float(fluxes[target_idx])
    }
    with open(os.path.join(args.outdir, "event_params.json"), "w") as f:
        json.dump(event_params, f, indent=4)
    
    wcs = create_roman_wcs(mosaic_size, pixel_scale)
    
    print(f"🎬 Generating {len(times)} epochs using optimized stamp modulation...")
    for i, t in enumerate(tqdm(times)):
        A = paczynski_magnification(t, args.t0, args.tE, args.u0)
        
        # Fast update: modulate the pre-rendered target stamp
        epoch_clean = static_clean.copy()
        epoch_clean[y_slice, x_slice] += target_stamp_baseline * A
        
        # Add Independent Noise
        noisy_image = np.random.poisson(np.maximum(0, epoch_clean + sky_level)).astype(np.float32)
        noisy_image += np.random.normal(0, read_noise, noisy_image.shape)
        
        if args.format == "fits":
            header = wcs.to_header()
            header['EXPTIME'], header['ZP'], header['SKYMAG'] = exp_time, zp, sky_mag
            header['MAGNIF'], header['EPOCH_T'] = A, t
            fits.PrimaryHDU(data=noisy_image, header=header).writeto(os.path.join(args.outdir, f"epoch_{i:04d}.fits"), overwrite=True)
        else:
            np.save(os.path.join(args.outdir, f"epoch_{i:04d}.npy"), noisy_image)
    
    print(f"✨ Success! Results in: {args.outdir}")

if __name__ == "__main__":
    main()
