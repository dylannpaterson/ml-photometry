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

def _numpy_paint_hybrid(image, x, y, fluxes, stamps):
    """ Vectorized stamp painting using NumPy. """
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
        ty, tx = target_y[mask], target_x[mask]
        val = stamps[i].flatten()[mask] * f
        np.add.at(image, (ty, tx), val * w00)
        np.add.at(image, (ty, tx+1), val * w10)
        np.add.at(image, (ty+1, tx), val * w01)
        np.add.at(image, (ty+1, tx+1), val * w11)

def calculate_safe_magnitude_cutoff(exp_time, zp, sky_mag, read_noise=5.0, sigma=1.5, snr_cutoff=2.0):
    pixel_scale = 0.11
    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (pixel_scale**2) * exp_time
    n_pix = 4 * np.pi * (sigma ** 2)
    bg_variance = n_pix * (sky_level + read_noise**2)
    a, b, c = 1.0, -(snr_cutoff**2), -(snr_cutoff**2) * bg_variance
    min_flux = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    return zp - 2.5 * np.log10(min_flux / exp_time)

def generate_field_realistic_psf_library(num_psfs=100, grid_size=127, oversample=4):
    """
    Generates a library of PRISTINE optical PSFs at OVERSAMPLED resolution.
    """
    print(f"📡 Generating Master OPTICAL PSF Library ({num_psfs} PSFs, {oversample}x oversampled)...")
    S = grid_size * oversample
    library = np.zeros((num_psfs, S, S), dtype=np.float32)
    half = S // 2
    optical_template = None
    if os.path.exists("roman_psf_prior_4x.pt"):
        try:
            optical_template = torch.load("roman_psf_prior_4x.pt", map_location='cpu', weights_only=True).numpy()
            # If the template size doesn't match, we might need to resize or pad
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
        
        # Scale optical core for oversampling
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

def compute_spatially_correlated_weights(x, y, mosaic_size, weights_lib):
    """
    Map PCA weights as a smooth polynomial function of (x, y) coordinates.
    """
    # Normalize coordinates to [-1, 1]
    nx = (x / mosaic_size) * 2 - 1
    ny = (y / mosaic_size) * 2 - 1
    
    # Simple 2nd order polynomial for each PCA component
    # In a real scenario, these coefficients would come from a model of the instrument
    num_stars = len(x)
    n_pca = weights_lib.shape[1]
    
    # We use the weights_lib as a pool of "realistic" weight vectors
    # and interpolate between a few "anchor" points in the field.
    num_anchors = 5
    anchors_x = np.array([0, 0, mosaic_size, mosaic_size, mosaic_size/2])
    anchors_y = np.array([0, mosaic_size, 0, mosaic_size, mosaic_size/2])
    anchor_indices = np.random.randint(0, len(weights_lib), num_anchors)
    anchor_weights = weights_lib[anchor_indices]
    
    # Radial Basis Function interpolation (simplified)
    weights = np.zeros((num_stars, n_pca))
    for i in range(num_anchors):
        dist_sq = (x - anchors_x[i])**2 + (y - anchors_y[i])**2
        # Gaussian RBF
        w = np.exp(-dist_sq / (2 * (mosaic_size/1.5)**2))
        weights += w[:, np.newaxis] * anchor_weights[i]
    
    # Normalize or add some local variation
    weights += np.random.normal(0, 0.05 * np.std(weights_lib, axis=0), weights.shape)
    
    return weights

def generate_mosaic(idx, output_dir, params, mosaic_size, cell_size, master_psf_data):
    start_time = time.time()
    area_ratio = (mosaic_size / params['image_size'])**2
    exp_time, zp, sky_mag = np.random.uniform(30.0, 60.0), 26.5, 22.0
    mag_limit = calculate_safe_magnitude_cutoff(exp_time, zp, sky_mag, snr_cutoff=2.0)
    
    n_stars_total = int(10 ** np.random.uniform(np.log10(params['min_stars'] * area_ratio), np.log10(params['max_stars'] * area_ratio)))
    print(f"📦 Mosaic {idx}: Rendering {n_stars_total:,} stars...")
    mags = sample_bulge_magnitudes(n_stars_total, np.random.uniform(14.5, 16.5), np.random.uniform(0.2, 0.5), np.random.uniform(5.0, 15.0), m_min=12.0, m_max=32.0, gamma=np.random.uniform(0.25, 0.35))
    fluxes = exp_time * (10 ** (-0.4 * (mags - zp)))
    
    eigen_psfs_4x, psf_weights_lib, mean_psf_4x = master_psf_data
    O = 4 # Oversampling
    
    px, py = np.random.uniform(0, mosaic_size, len(fluxes)), np.random.uniform(0, mosaic_size, len(fluxes))
    
    # 3. Enforce Spatial Correlation for PCA Weights
    physical_weights = compute_spatially_correlated_weights(px, py, mosaic_size, psf_weights_lib)

    # 4. Apply Jitter Before Pixelation
    s_jit = np.random.normal(0.127, 0.01) 
    q_jit = np.random.uniform(0.8, 1.0)
    theta_jit = np.random.uniform(0, np.pi)
    
    # Build the 2D jitter kernel at OVERSAMPLED resolution
    S_jit_high = SHAPE_SIZE * O
    k_half_high = S_jit_high // 2
    gy, gx = np.meshgrid(np.arange(S_jit_high) - k_half_high, np.arange(S_jit_high) - k_half_high, indexing='ij')
    cos, sin = np.cos(theta_jit), np.sin(theta_jit)
    s_jit_high = s_jit * O
    gxp, gyp = gx * cos + gy * sin, -gx * sin + gy * cos
    jitter_kernel_high = np.exp(-(gxp**2 / (2 * s_jit_high**2) + gyp**2 / (2 * (s_jit_high * q_jit)**2)))
    jitter_kernel_high /= (jitter_kernel_high.sum() + 1e-9)

    # Convolve mean and eigen PSFs with jitter BEFORE binning
    print(f"🎬 Applying jitter to PSF library (O={O}x)...")
    mean_psf_jit_4x = fftconvolve(mean_psf_4x, jitter_kernel_high, mode='same')
    eigen_psfs_jit_4x = np.array([fftconvolve(e, jitter_kernel_high, mode='same') for e in eigen_psfs_4x])

    # Pre-compute 16 shifted 1x PSFs for the mean component
    mean_psf_library = np.zeros((O, O, SHAPE_SIZE, SHAPE_SIZE), dtype=np.float32)
    for dy_idx in range(O):
        for dx_idx in range(O):
            mean_psf_library[dy_idx, dx_idx] = mean_psf_jit_4x[dy_idx::O, dx_idx::O][:SHAPE_SIZE, :SHAPE_SIZE]

    # Rendering
    full_image = np.zeros((mosaic_size, mosaic_size), dtype=np.float32)
    x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
    dx_idx = np.clip(np.floor((px - x0) * O).astype(int), 0, O-1)
    dy_idx = np.clip(np.floor((py - y0) * O).astype(int), 0, O-1)
    valid = (x0 >= 0) & (x0 < mosaic_size) & (y0 >= 0) & (y0 < mosaic_size)
    
    # 1. Base Pass using 16 sub-pixel grids (Fast!)
    print(f"🎨 Rendering {len(px):,} stars using 16 sub-pixel grids...")
    for dyi in range(O):
        for dxi in range(O):
            mask = valid & (dx_idx == dxi) & (dy_idx == dyi)
            if not mask.any(): continue
            flat_indices = y0[mask] * mosaic_size + x0[mask]
            grid = np.bincount(flat_indices, weights=fluxes[mask], minlength=mosaic_size*mosaic_size).reshape(mosaic_size, mosaic_size)
            full_image += fftconvolve(grid, mean_psf_library[dyi, dxi], mode='same')

    # 2. Correction pass for bright stars
    is_bright = mags < mag_limit
    if is_bright.any():
        print(f"✨ Applying PCA corrections to {is_bright.sum():,} bright stars...")
        sig_indices = np.where(is_bright)[0]
        sig_weights = physical_weights[sig_indices]
        half = SHAPE_SIZE // 2
        for i, idx_sig in enumerate(sig_indices):
            ix, iy, f = x0[idx_sig], y0[idx_sig], fluxes[idx_sig]
            dxi, dyi = dx_idx[idx_sig], dy_idx[idx_sig]
            
            # Reconstruct oversampled jittered PSF for this star
            weights = sig_weights[i]
            high_psf = mean_psf_jit_4x + np.tensordot(weights, eigen_psfs_jit_4x, axes=1)
            # Bin down with shift
            stamp = high_psf[dyi::O, dxi::O][:SHAPE_SIZE, :SHAPE_SIZE]
            
            y0_s, y1_s = max(0, iy-half), min(mosaic_size, iy+half+1)
            x0_s, x1_s = max(0, ix-half), min(mosaic_size, ix+half+1)
            sy0, sy1 = half - (iy - y0_s), half + (y1_s - iy)
            sx0, sx1 = half - (ix - x0_s), half + (x1_s - ix)
            
            # We subtract the mean contribution already added
            full_image[y0_s:y1_s, x0_s:x1_s] += (stamp[sy0:sy1, sx0:sx1] - mean_psf_library[dyi, dxi, sy0:sy1, sx0:sx1]) * f

    full_image = np.maximum(0, full_image)
    
    v_mask = mags < mag_limit
    x_v, y_v, flux_v, mag_v = px[v_mask], py[v_mask], fluxes[v_mask], mags[v_mask]
    final_weights_v = physical_weights[v_mask]

    # Recalculate Rigorous SNR
    half = SHAPE_SIZE // 2
    # 1. Calculate N_eff using 1x PSF normalized to peak=1
    centered_psf = mean_psf_library[O//2, O//2]
    psf_peak_val = np.max(centered_psf)
    psf_norm_to_peak = centered_psf / (psf_peak_val + 1e-9)
    N_eff = np.sum(psf_norm_to_peak**2)
    print(f"📊 Calculated N_eff: {N_eff:.2f}")
    
    # 2. Extract true local light using integer pixel indexing
    actual_pixel_values = full_image[y0[v_mask], x0[v_mask]]
    
    # 3. Calculate phase-dependent peak fraction
    peaks = mean_psf_library[:, :, half, half]
    max_peak = np.max(peaks)
    min_peak = np.min(peaks)
    
    dx_sub = px[v_mask] - x0[v_mask]
    dy_sub = py[v_mask] - y0[v_mask]
    dist_from_pixel_center = np.sqrt((dx_sub - 0.5)**2 + (dy_sub - 0.5)**2)
    phase_dependent_peak = max_peak - (max_peak - min_peak) * (dist_from_pixel_center / 0.7071)
    
    # Reconstruct star peaks including PCA
    star_peaks = phase_dependent_peak.copy()
    for i in range(len(x_v)):
        dxi, dyi = dx_idx[v_mask][i], dy_idx[v_mask][i]
        star_peaks[i] += np.dot(final_weights_v[i], eigen_psfs_jit_4x[:, dyi::O, dxi::O][:, half, half])
        
    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (0.11**2) * exp_time
    # 4. Calculate Confusion Noise
    confusion_light = np.maximum(0.0, actual_pixel_values - (flux_v * star_peaks))
    
    # 5. Calculate Final SNR
    noise_variance = flux_v + N_eff * (sky_level + confusion_light + 25.0)
    
    cat_dtype = [('x', 'f4'), ('y', 'f4'), ('flux', 'f4'), ('mag', 'f4'), ('snr', 'f4')] + [(f'w{i}', 'f2') for i in range(N_PCA_COMPONENTS)]
    structured_cat = np.zeros(len(x_v), dtype=cat_dtype)
    structured_cat['x'], structured_cat['y'], structured_cat['flux'], structured_cat['mag'] = x_v, y_v, flux_v, mag_v
    structured_cat['snr'] = flux_v / np.sqrt(np.maximum(1.0, noise_variance))
    for i in range(N_PCA_COMPONENTS): structured_cat[f'w{i}'] = final_weights_v[:, i]
    structured_cat = structured_cat[np.argsort(structured_cat['y'])]

    # Save outputs: Normalize saved library by peak of mean PSF for visualizer consistency
    mean_psf_1x = mean_psf_jit_4x[0::O, 0::O][:SHAPE_SIZE, :SHAPE_SIZE]
    eigen_psfs_1x = eigen_psfs_jit_4x[:, 0::O, 0::O][:, :SHAPE_SIZE, :SHAPE_SIZE]
    
    norm_val = np.max(mean_psf_1x)
    mean_psf_1x /= norm_val
    eigen_psfs_1x /= norm_val
    
    lib_save = np.concatenate([eigen_psfs_1x.reshape(N_PCA_COMPONENTS, -1), 
                               mean_psf_1x.reshape(1, -1)], axis=0)
    
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
