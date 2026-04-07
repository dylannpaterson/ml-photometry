import argparse
import os
import torch
import numpy as np
import pandas as pd
from castor.data.stage0_gaussian import sample_bulge_magnitudes
from castor.cloud.config_utils import load_config
import shutil
import time
from castor.constants import DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL, SHAPE_SIZE, N_PCA_COMPONENTS
from scipy.signal import fftconvolve
from numba import njit
from scipy.ndimage import map_coordinates

# GPU/JAX Acceleration
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

def calculate_safe_magnitude_cutoff(exp_time, zp, sky_mag, read_noise=5.0, sigma=1.5, snr_cutoff=1.0):
    # Reduced default n_pix to 12.0 to match jittered Roman NEA
    pixel_scale = 0.11
    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (pixel_scale**2) * exp_time
    n_pix = 12.0
    bg_variance = n_pix * (sky_level + read_noise**2)
    a, b, c = 1.0, -(snr_cutoff**2), -(snr_cutoff**2) * bg_variance
    min_flux = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    return zp - 2.5 * np.log10(min_flux / exp_time)

def generate_field_realistic_psf_library(num_psfs=100, grid_size=127, oversample=4):
    print(f"📡 Generating Master OPTICAL PSF Library ({num_psfs} PSFs, {oversample}x oversampled)...")
    S = grid_size * oversample
    library = np.zeros((num_psfs, S, S), dtype=np.float32)
    half = S // 2
    optical_template = None
    if os.path.exists("roman_psf_prior_4x.pt"):
        try:
            optical_template = torch.load("roman_psf_prior_4x.pt", map_location='cpu', weights_only=False).numpy()
            if optical_template.shape[0] != S:
                from scipy.ndimage import zoom
                scale = S / optical_template.shape[0]
                optical_template = zoom(optical_template, scale, order=3)
        except Exception as e: print(f"⚠️ Oversampled PSF Load Failed: {e}")

    y, x = np.meshgrid(np.arange(S) - half, np.arange(S) - half, indexing='ij')
    for i in range(num_psfs):
        fx, fy = np.random.uniform(-2048, 2048), np.random.uniform(-2048, 2048)
        r_norm = np.sqrt(fx**2 + fy**2) / 2896.0
        q_opt = np.random.uniform(0.9, 1.0) - (0.1 * r_norm)
        theta = np.arctan2(fy, fx) + np.random.normal(0, 0.1)
        cos, sin = np.cos(theta), np.sin(theta)
        xp, yp = x * cos + y * sin, -x * sin + y * cos
        s_opt = 0.45 * oversample 
        opt_core = np.exp(-(xp**2 / (2 * s_opt**2) + yp**2 / (2 * (s_opt * q_opt)**2)))
        opt_core /= (opt_core.sum() + 1e-9)
        if optical_template is not None:
            from scipy.ndimage import rotate
            rotated = rotate(optical_template, np.random.uniform(0, 360), reshape=False, order=3, mode='constant', cval=0.0)
            psf = fftconvolve(rotated, opt_core, mode='same')
        else:
            psf = opt_core
        psf = np.maximum(0, psf)
        library[i] = psf / (psf.sum() + 1e-9)
    return library

def _compute_eigen_psfs(large_library, n_components=10):
    N, H, W = large_library.shape
    data = torch.from_numpy(large_library).float().view(N, H * W)
    mean_psf = data.mean(dim=0)
    centered_data = data - mean_psf
    U, S, V = torch.pca_lowrank(centered_data, q=n_components)
    eigen_psfs = V.t().view(n_components, H, W).numpy()
    psf_weights = (U * S).numpy() 
    return eigen_psfs, psf_weights, mean_psf.view(H, W).numpy()

def generate_mosaic(idx, output_dir, params, mosaic_size, cell_size, master_psf_data):
    start_time = time.time()
    area_ratio = (mosaic_size / params['image_size'])**2
    exp_time, zp, sky_mag = np.random.uniform(30.0, 60.0), 26.5, 22.0
    # FIX: Set snr_cutoff to 1.0 to see the full confusion roll-off in visualization
    mag_limit = calculate_safe_magnitude_cutoff(exp_time, zp, sky_mag, snr_cutoff=1.0)
    
    n_stars_total = int(10 ** np.random.uniform(np.log10(params['min_stars'] * area_ratio), np.log10(params['max_stars'] * area_ratio)))
    print(f"📦 Mosaic {idx}: Rendering {n_stars_total:,} stars...")
    mags = sample_bulge_magnitudes(n_stars_total, np.random.uniform(14.5, 16.5), np.random.uniform(0.2, 0.5), np.random.uniform(5.0, 15.0), m_min=12.0, m_max=32.0, gamma=np.random.uniform(0.25, 0.35))
    fluxes = exp_time * (10 ** (-0.4 * (mags - zp)))
    
    eigen_psfs, psf_weights_lib, mean_psf = master_psf_data
    O = 4 # Oversampling
    
    repr_idx = np.random.randint(0, len(psf_weights_lib))
    repr_weights = psf_weights_lib[repr_idx]
    
    def get_upsampled(img, scale):
        from scipy.ndimage import zoom
        return zoom(img, scale, order=3)

    if mean_psf.shape[0] == SHAPE_SIZE:
        mean_psf_4x = get_upsampled(mean_psf, O)
        eigen_psfs_4x = np.array([get_upsampled(e, O) for e in eigen_psfs])
    else:
        mean_psf_4x = mean_psf
        eigen_psfs_4x = eigen_psfs

    repr_psf_4x = mean_psf_4x + np.tensordot(repr_weights, eigen_psfs_4x, axes=1)
    repr_psf_4x = np.maximum(0, repr_psf_4x)
    repr_psf_4x /= (repr_psf_4x.sum() + 1e-9)

    s_jit, q_jit, theta_jit = np.random.normal(0.127, 0.01), np.random.uniform(0.8, 1.0), np.random.uniform(0, np.pi)
    S_jit_high = SHAPE_SIZE * O
    k_half_high = S_jit_high // 2
    gy, gx = np.meshgrid(np.arange(S_jit_high) - k_half_high, np.arange(S_jit_high) - k_half_high, indexing='ij')
    cos, sin = np.cos(theta_jit), np.sin(theta_jit)
    s_jit_high = s_jit * O
    gxp, gyp = gx * cos + gy * sin, -gx * sin + gy * cos
    jitter_kernel_high = np.exp(-(gxp**2 / (2 * s_jit_high**2) + gyp**2 / (2 * (s_jit_high * q_jit)**2)))
    jitter_kernel_high /= (jitter_kernel_high.sum() + 1e-9)

    print(f"🎬 Applying jitter to representative PSF...")
    repr_psf_jit_4x = fftconvolve(repr_psf_4x, jitter_kernel_high, mode='same')

    # FIX: Use Proper Binning (Area Integration) rather than decimation
    psf_library = np.zeros((O, O, SHAPE_SIZE, SHAPE_SIZE), dtype=np.float32)
    # Pad to allow shifting
    padded_psf = np.pad(repr_psf_jit_4x, ((0, O), (0, O)))
    for dy_idx in range(O):
        for dx_idx in range(O):
            # Binning: Summing 4x4 sub-pixels into 1 detector pixel
            window = padded_psf[dy_idx : dy_idx + SHAPE_SIZE*O, dx_idx : dx_idx + SHAPE_SIZE*O]
            binned = window.reshape(SHAPE_SIZE, O, SHAPE_SIZE, O).mean(axis=(1, 3))
            psf_library[dy_idx, dx_idx] = binned / (np.sum(binned) + 1e-9)

    px, py = np.random.uniform(0, mosaic_size, len(fluxes)), np.random.uniform(0, mosaic_size, len(fluxes))
    full_image = np.zeros((mosaic_size, mosaic_size), dtype=np.float32)
    x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
    dx_idx = np.clip(np.floor((px - x0) * O).astype(int), 0, O-1)
    dy_idx = np.clip(np.floor((py - y0) * O).astype(int), 0, O-1)
    valid = (x0 >= 0) & (x0 < mosaic_size) & (y0 >= 0) & (y0 < mosaic_size)
    
    print(f"🎨 Rendering {len(px):,} stars using representative PSF phases...")
    for dyi in range(O):
        for dxi in range(O):
            mask = valid & (dx_idx == dxi) & (dy_idx == dyi)
            if not mask.any(): continue
            flat_indices = y0[mask] * mosaic_size + x0[mask]
            grid = np.bincount(flat_indices, weights=fluxes[mask], minlength=mosaic_size*mosaic_size).reshape(mosaic_size, mosaic_size)
            full_image += fftconvolve(grid, psf_library[dyi, dxi], mode='same')

    full_image = np.maximum(0, full_image)
    
    v_mask = mags < mag_limit
    x_v, y_v, flux_v, mag_v = px[v_mask], py[v_mask], fluxes[v_mask], mags[v_mask]

    half = SHAPE_SIZE // 2
    centered_psf = psf_library[O//2, O//2]
    N_eff = 1.0 / (np.sum(centered_psf**2) + 1e-9)
    print(f"📊 Calculated N_eff: {N_eff:.2f}")
    
    actual_pixel_values = full_image[y0[v_mask], x0[v_mask]]
    star_peaks = psf_library[dy_idx[v_mask], dx_idx[v_mask], half, half]
    
    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (0.11**2) * exp_time
    confusion_light = np.maximum(0.0, actual_pixel_values - (flux_v * star_peaks))
    
    read_noise = 5.0
    noise_variance = flux_v + N_eff * (sky_level + confusion_light + read_noise**2)
    
    cat_dtype = [('x', 'f4'), ('y', 'f4'), ('flux', 'f4'), ('mag', 'f4'), ('snr', 'f4')] + [(f'w{i}', 'f2') for i in range(N_PCA_COMPONENTS)]
    structured_cat = np.zeros(len(x_v), dtype=cat_dtype)
    structured_cat['x'], structured_cat['y'], structured_cat['flux'], structured_cat['mag'] = x_v, y_v, flux_v, mag_v
    structured_cat['snr'] = flux_v / np.sqrt(np.maximum(1.0, noise_variance))
    for i in range(N_PCA_COMPONENTS): structured_cat[f'w{i}'] = repr_weights[i]
    structured_cat = structured_cat[np.argsort(structured_cat['y'])]

    # Save outputs: Peak-normalize for visualizer
    repr_psf_1x = repr_psf_jit_4x.reshape(SHAPE_SIZE, O, SHAPE_SIZE, O).mean(axis=(1, 3))
    norm_val = np.max(repr_psf_1x)
    repr_psf_1x /= (norm_val + 1e-9)
    lib_save = np.zeros((N_PCA_COMPONENTS + 1, SHAPE_SIZE * SHAPE_SIZE), dtype=np.float32)
    lib_save[-1] = repr_psf_1x.flatten()
    
    for suffix, data in [('img', full_image), ('cat', structured_cat), ('meta', np.array([exp_time, zp, sky_mag, s_jit, q_jit, theta_jit])), ('psf_lib', lib_save)]:
        np.save(os.path.join(output_dir, f"mosaic_{idx:03d}_{suffix}.npy"), data)
    print(f"✅ Mosaic {idx} done in {time.time() - start_time:.2f}s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--psf_library", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    stage_key = f"stage{args.stage}"
    stage_cfg = cfg["curriculum"][stage_key]
    if args.psf_library and os.path.exists(args.psf_library):
        master_data = torch.load(args.psf_library, map_location='cpu', weights_only=False)
        master_psf_data = (master_data['eigen_psfs'], master_data['weights_lib'], master_data['mean_psf'])
    else:
        kb_array = generate_field_realistic_psf_library(num_psfs=100, grid_size=SHAPE_SIZE)
        master_psf_data = _compute_eigen_psfs(kb_array, n_components=N_PCA_COMPONENTS)
    num = args.num if args.num else stage_cfg["mosaic_params"]["num_mosaics"]
    out = args.output_dir if args.output_dir else os.path.join(stage_cfg["data_dir"], "mosaics")
    os.makedirs(out, exist_ok=True)
    params = {"min_stars": cfg["data_params"]["min_stars"], "max_stars": cfg["data_params"]["max_stars"], "image_size": cfg["data_params"]["image_size"]}
    for i in range(num): generate_mosaic(i, out, params, stage_cfg["mosaic_params"]["mosaic_size"], stage_cfg["cell_size"], master_psf_data)

if __name__ == "__main__": main()
