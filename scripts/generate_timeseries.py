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
from castor.constants import GLOBAL_STRETCH_SCALE, SHAPE_SIZE, N_PCA_COMPONENTS
from scipy.signal import fftconvolve
from scipy.ndimage import map_coordinates

# Astronomical Libraries
from astropy.io import fits
from astropy.wcs import WCS

def paczynski_magnification(t, t0, tE, u0):
    """ Standard Paczynski microlensing magnification formula. """
    u = np.sqrt(u0**2 + ((t - t0) / tE)**2)
    A = (u**2 + 2) / (u * np.sqrt(u**2 + 4) + 1e-9)
    return A

def create_roman_wcs(mosaic_size, RA=266.417, Dec=-29.008, pixel_scale=0.11, crpix_offset=[0, 0]):
    """ Creates a WCS for a Roman-like mosaic with optional crpix offset for drift. """
    w = WCS(naxis=2)
    # Note: FITS pixels are 1-indexed, crpix usually refers to center of pixel (0.5)
    w.wcs.crpix = [mosaic_size / 2.0 + crpix_offset[0], mosaic_size / 2.0 + crpix_offset[1]]
    w.wcs.cdelt = [-pixel_scale / 3600.0, pixel_scale / 3600.0]
    w.wcs.crval = [RA, Dec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w

def calculate_safe_magnitude_cutoff(exp_time, zp, sky_mag, read_noise=5.0, sigma=1.5, snr_cutoff=2.0):
    pixel_scale = 0.11
    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (pixel_scale**2) * exp_time
    n_pix = 4 * np.pi * (sigma ** 2)
    bg_variance = n_pix * (sky_level + read_noise**2)
    a, b, c = 1.0, -(snr_cutoff**2), -(snr_cutoff**2) * bg_variance
    min_flux = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    return zp - 2.5 * np.log10(min_flux / exp_time)

def generate_field_realistic_psf_library(num_psfs=100, grid_size=SHAPE_SIZE):
    """ Generates a library of PRISTINE optical PSFs. """
    library = np.zeros((num_psfs, grid_size, grid_size), dtype=np.float32)
    half = grid_size // 2
    optical_template = None
    if os.path.exists("roman_psf_prior.pt"):
        try:
            optical_template = torch.load("roman_psf_prior.pt", map_location='cpu', weights_only=True).numpy()
        except Exception as e: print(f"⚠️ PSF Load Failed: {e}")

    y, x = np.meshgrid(np.arange(grid_size) - half, np.arange(grid_size) - half, indexing='ij')
    for i in range(num_psfs):
        fx, fy = np.random.uniform(-2048, 2048), np.random.uniform(-2048, 2048)
        r_norm = np.sqrt(fx**2 + fy**2) / 2896.0
        q_opt = np.random.uniform(0.9, 1.0) - (0.1 * r_norm)
        theta = np.arctan2(fy, fx) + np.random.normal(0, 0.1)
        cos, sin = np.cos(theta), np.sin(theta)
        xp, yp = x * cos + y * sin, -x * sin + y * cos
        s_opt = 0.45
        opt_core = np.exp(-(xp**2 / (2 * s_opt**2) + yp**2 / (2 * (s_opt * q_opt)**2)))
        opt_core /= (opt_core.sum() + 1e-9)
        if optical_template is not None:
            from scipy.ndimage import rotate
            rotated = rotate(optical_template, np.random.uniform(0, 360), reshape=False, order=3, mode='constant', cval=0.0)
            psf = fftconvolve(rotated, opt_core, mode='same')
        else:
            psf = opt_core
        psf = np.maximum(0, psf); library[i] = psf / (psf.sum() + 1e-9)
    return library

def _compute_eigen_psfs(large_library, n_components=N_PCA_COMPONENTS):
    N, H, W = large_library.shape
    data = torch.from_numpy(large_library).float().view(N, H * W)
    mean_psf = data.mean(dim=0)
    centered_data = data - mean_psf
    U, S, V = torch.pca_lowrank(centered_data, q=n_components)
    eigen_psfs = V.t().view(n_components, H, W).numpy()
    psf_weights = (U * S).numpy() 
    return eigen_psfs, psf_weights, mean_psf.view(H, W).numpy()

def _paint_stamps(image, x, y, fluxes, stamps):
    """ Optimized stamp painting loop. """
    N, S, _ = stamps.shape
    half = S // 2
    H, W = image.shape
    sj, sk = np.meshgrid(np.arange(S), np.arange(S), indexing='ij')
    sj, sk = sj.flatten() - half, sk.flatten() - half
    for i in range(N):
        px, py, f = x[i], y[i], fluxes[i]
        ix, iy = int(px), int(py)
        dx, dy = px - ix, py - iy
        w00, w10, w01, w11 = (1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy
        target_y, target_x = iy + sj, ix + sk
        mask = (target_y >= 0) & (target_y < H - 1) & (target_x >= 0) & (target_x < W - 1)
        if not mask.any(): continue
        ty, tx = target_y[mask], target_x[mask]; val = stamps[i].flatten()[mask] * f
        np.add.at(image, (ty, tx), val * w00)
        np.add.at(image, (ty, tx+1), val * w10)
        np.add.at(image, (ty+1, tx), val * w01)
        np.add.at(image, (ty+1, tx+1), val * w11)

def render_numpy_fidelity(x, y, fluxes, mags, psf_indices, eigen_psfs, mean_psf, psf_weights_lib, mosaic_size, mag_limit):
    """ Hybrid Renderer: Mean PSF for all, PCA Corrections for Bright. """
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    dx, dy = x - x0, y - y0
    valid = (x0 >= 0) & (x0 < mosaic_size-1) & (y0 >= 0) & (y0 < mosaic_size-1)
    indices = np.concatenate([y0[valid] * mosaic_size + x0[valid], y0[valid] * mosaic_size + x0[valid] + 1, (y0[valid]+1) * mosaic_size + x0[valid], (y0[valid]+1) * mosaic_size + x0[valid] + 1])
    f_v = fluxes[valid]
    vals = np.concatenate([f_v * (1-dx[valid]) * (1-dy[valid]), f_v * dx[valid] * (1-dy[valid]), f_v * (1-dx[valid]) * dy[valid], f_v * dx[valid] * dy[valid]])
    base_grid = np.bincount(indices, weights=vals, minlength=mosaic_size*mosaic_size).reshape(mosaic_size, mosaic_size)
    full_image = fftconvolve(base_grid, mean_psf, mode='same')
    
    is_bright = mags < mag_limit
    if is_bright.any():
        b_x, b_y, b_f, b_psf_idx = x[is_bright], y[is_bright], fluxes[is_bright], psf_indices[is_bright]
        b_weights = psf_weights_lib[b_psf_idx]
        batch_size = 5000
        for s_idx in range(0, len(b_x), batch_size):
            e_idx = min(s_idx + batch_size, len(b_x))
            correction_stamps = (b_weights[s_idx:e_idx] @ eigen_psfs.reshape(N_PCA_COMPONENTS, -1)).reshape(-1, SHAPE_SIZE, SHAPE_SIZE)
            _paint_stamps(full_image, b_x[s_idx:e_idx], b_y[s_idx:e_idx], b_f[s_idx:e_idx], correction_stamps)
    return full_image

def get_star_stamp_pca(x, y, flux, weights, eigen_psfs, mean_psf, mosaic_size):
    psf = mean_psf + (weights @ eigen_psfs.reshape(N_PCA_COMPONENTS, -1)).reshape(SHAPE_SIZE, SHAPE_SIZE)
    psf = np.maximum(0, psf); psf /= (psf.sum() + 1e-9)
    half = SHAPE_SIZE // 2; ix, iy = int(x), int(y); dx, dy = x - ix, y - iy
    phase_map = np.zeros((SHAPE_SIZE + 1, SHAPE_SIZE + 1), dtype=np.float32)
    phase_map[half, half], phase_map[half, half+1], phase_map[half+1, half], phase_map[half+1, half+1] = flux*(1-dx)*(1-dy), flux*dx*(1-dy), flux*(1-dx)*dy, flux*dx*dy
    stamp = fftconvolve(phase_map, psf, mode='same')
    y0, y1, x0, x1 = iy - half, iy - half + stamp.shape[0], ix - half, ix - half + stamp.shape[1]
    return stamp, y0, y1, x0, x1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--outdir", default="data/microlensing_stack_drift")
    parser.add_argument("--mosaic_size", type=int, default=512)
    parser.add_argument("--target_mag", type=float, default=21.0)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--cadence", type=float, default=1.0)
    parser.add_argument("--t0", type=float, default=36.0)
    parser.add_argument("--tE", type=float, default=20.0)
    parser.add_argument("--u0", type=float, default=0.1)
    parser.add_argument("--drift_rate", type=float, default=0.5, help="Pixels per day")
    parser.add_argument("--drift_angle", type=float, default=45.0, help="Angle in degrees")
    args = parser.parse_args()
    config = load_config(args.config); data_cfg = config["data_params"]; os.makedirs(args.outdir, exist_ok=True)
    mosaic_size = args.mosaic_size; n_lib = 100; zp, sky_mag = 26.5, 22.0
    read_noise = data_cfg["physics_params"].get("read_noise", 5.0)
    exp_time = (data_cfg["physics_params"]["exp_time_min"] + data_cfg["physics_params"]["exp_time_max"]) / 2
    mag_limit = calculate_safe_magnitude_cutoff(exp_time, zp, sky_mag)

    kb_array = generate_field_realistic_psf_library(num_psfs=n_lib)
    eigen_psfs, psf_weights_lib, mean_psf = _compute_eigen_psfs(kb_array)
    s_jit = np.random.normal(0.127, 0.01); q_jit, t_jit = np.random.uniform(0.8, 1.0), np.random.uniform(0, np.pi)
    kh = SHAPE_SIZE // 2; gy, gx = np.meshgrid(np.arange(SHAPE_SIZE) - kh, np.arange(SHAPE_SIZE) - kh, indexing='ij')
    cos, sin = np.cos(t_jit), np.sin(t_jit); gxp, gyp = gx * cos + gy * sin, -gx * sin + gy * cos
    jitter_kernel = np.exp(-(gxp**2 / (2 * s_jit**2) + gyp**2 / (2 * (s_jit * q_jit)**2))); jitter_kernel /= (jitter_kernel.sum() + 1e-9)

    # --- CALCULATE MAX DRIFT ---
    cadence_days = args.cadence / 24.0
    times = (args.t0 - (args.num_epochs // 2) * cadence_days) + np.arange(args.num_epochs) * cadence_days
    total_time_span = times.max() - times.min()
    max_drift_pix = args.drift_rate * total_time_span
    padding = int(max_drift_pix) + 10
    super_size = mosaic_size + padding * 2

    print(f"🌌 Generating background catalog for {super_size} super-field (Drift: {args.drift_rate} px/day)...")
    n_requested = int(np.random.uniform(data_cfg['min_stars'], data_cfg['max_stars']) * (super_size / 256)**2)
    mags = sample_bulge_magnitudes(n_requested, 15.5, 0.35, 10.0, m_min=12.0, m_max=32.0, gamma=0.3)
    n_total = len(mags)
    fluxes = exp_time * (10 ** (-0.4 * (mags - zp)))
    x_c, y_c, psf_idx = np.random.uniform(0, super_size, n_total), np.random.uniform(0, super_size, n_total), np.random.randint(0, n_lib, n_total)
    
    t_idx = np.argmin(np.abs(mags - args.target_mag))
    
    # Target Celestial Center
    target_RA, target_Dec = 266.417, -29.008
    # Initial Super-mosaic position: center + small random subpixel offset
    x_c[t_idx] = super_size / 2.0 + np.random.uniform(-0.5, 0.5)
    y_c[t_idx] = super_size / 2.0 + np.random.uniform(-0.5, 0.5)
    
    # Calculate the target's celestial coordinates based on the initial WCS (at dt=0)
    # At dt=0, x0=padding, y0=padding. 
    # Frame coord at dt=0 is x_c[t_idx] - padding
    initial_wcs = create_roman_wcs(mosaic_size, RA=target_RA, Dec=target_Dec, crpix_offset=[0, 0])
    obj_ra, obj_dec = initial_wcs.all_pix2world(x_c[t_idx] - padding, y_c[t_idx] - padding, 0)
    
    print(f"🎯 Selected Target Star: Mag={mags[t_idx]:.3f} at celestial ({obj_ra:.6f}, {obj_dec:.6f})")
    
    print("🎬 Rendering static super-background field (High Fidelity)...")
    s_mask = np.ones(n_total, dtype=bool); s_mask[t_idx] = False
    static_super = render_numpy_fidelity(x_c[s_mask], y_c[s_mask], fluxes[s_mask], mags[s_mask], psf_idx[s_mask], eigen_psfs, mean_psf, psf_weights_lib, super_size, mag_limit)
    t_stamp, sy0, sy1, sx0, sx1 = get_star_stamp_pca(x_c[t_idx], y_c[t_idx], fluxes[t_idx], psf_weights_lib[psf_idx[t_idx]], eigen_psfs, mean_psf, super_size)
    
    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (0.11**2) * exp_time
    drift_vec = np.array([np.cos(np.radians(args.drift_angle)), np.sin(np.radians(args.drift_angle))]) * args.drift_rate

    print(f"🎬 Generating {len(times)} epochs with Pointing Drift...")
    for i, t in enumerate(tqdm(times)):
        A = paczynski_magnification(t, args.t0, args.tE, args.u0)
        dt = t - times[0]
        curr_drift = drift_vec * dt
        
        y0, x0 = int(padding + curr_drift[1]), int(padding + curr_drift[0])
        y1, x1 = y0 + mosaic_size, x0 + mosaic_size
        epoch_clean = static_super[y0:y1, x0:x1].copy()
        
        cur_sy0, cur_sy1, cur_sx0, cur_sx1 = int(sy0 - y0), int(sy1 - y0), int(sx0 - x0), int(sx1 - x0)
        st_y0, st_y1 = max(0, -cur_sy0), min(t_stamp.shape[0], mosaic_size - cur_sy0)
        st_x0, st_x1 = max(0, -cur_sx0), min(t_stamp.shape[1], mosaic_size - cur_sx0)
        ta_y0, ta_y1 = max(0, cur_sy0), min(mosaic_size, cur_sy1)
        ta_x0, ta_x1 = max(0, cur_sx0), min(mosaic_size, cur_sx1)
        
        if ta_y1 > ta_y0 and ta_x1 > ta_x0:
            epoch_clean[ta_y0:ta_y1, ta_x0:ta_x1] += t_stamp[st_y0:st_y1, st_x0:st_x1] * A
        
        epoch_clean = fftconvolve(epoch_clean, jitter_kernel, mode='same')
        noisy = np.random.poisson(np.maximum(0, epoch_clean + sky_level)).astype(np.float32)
        noisy += np.random.normal(0, read_noise, noisy.shape)
        
        # Synchronized WCS: Update CRPIX to keep celestial coords static
        wcs = create_roman_wcs(mosaic_size, RA=target_RA, Dec=target_Dec, crpix_offset=[-curr_drift[0], -curr_drift[1]])
        header = wcs.to_header()
        header.update({
            'EXPTIME': exp_time, 'ZP': zp, 'SKYMAG': sky_mag, 'MAGNIF': A, 'EPOCH_T': t, 
            'OBJ_X': float(x_c[t_idx] - x0), 'OBJ_Y': float(y_c[t_idx] - y0),
            'OBJ_RA': float(obj_ra), 'OBJ_DEC': float(obj_dec),
            'DRIFT_X': curr_drift[0], 'DRIFT_Y': curr_drift[1]
        })
        fits.PrimaryHDU(data=noisy, header=header).writeto(os.path.join(args.outdir, f"epoch_{i:04d}.fits"), overwrite=True)
    
    print(f"✨ Success! Drifting stack ready in: {args.outdir}")

if __name__ == "__main__": main()
