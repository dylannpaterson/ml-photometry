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
import scipy.fft
from numba import njit

# GPU/JAX Acceleration
try:
    import jax
    HAS_JAX = True
    from castor.data.gpu_renderer import render_generate_and_filter_gpu, is_gpu_available
    if is_gpu_available():
        print("🚀 JAX GPU Acceleration Enabled")
    else:
        print("🐢 JAX CPU detected, using optimized NumPy/Numba path instead")
except ImportError:
    HAS_JAX = False
    print("⚠️ JAX not found, using slow NumPy path")

@njit(cache=True)
def _numba_paint_hybrid(image, x, y, fluxes, stamps):
    """
    Ultra-fast sub-pixel accurate stamp painting.
    Equivalent to 1st order bilinear scattering + convolution.
    """
    N, S, _ = stamps.shape
    half = S // 2
    H, W = image.shape
    for i in range(N):
        # Coordinates and flux
        px, py, f = x[i], y[i], fluxes[i]
        ix, iy = int(px), int(py)
        dx, dy = px - ix, py - iy
        
        # 4-pixel bilinear weights
        w00, w10, w01, w11 = (1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy
        
        # Paint the stamp
        for sj in range(S):
            curr_y = iy + sj - half
            if curr_y < 0 or curr_y >= H - 1: continue
            for sk in range(S):
                curr_x = ix + sk - half
                if curr_x < 0 or curr_x >= W - 1: continue
                
                val = stamps[i, sj, sk] * f
                image[curr_y, curr_x] += val * w00
                image[curr_y, curr_x+1] += val * w10
                image[curr_y+1, curr_x] += val * w01
                image[curr_y+1, curr_x+1] += val * w11

def calculate_safe_magnitude_cutoff(exp_time, zp, sky_mag, read_noise=5.0, sigma=1.5, snr_cutoff=2.0):
    pixel_scale = 0.11
    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (pixel_scale**2) * exp_time
    n_pix = 4 * np.pi * (sigma ** 2)
    bg_variance = n_pix * (sky_level + read_noise**2)
    a, b, c = 1.0, -(snr_cutoff**2), -(snr_cutoff**2) * bg_variance
    min_flux = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    return zp - 2.5 * np.log10(min_flux / exp_time)

def generate_field_realistic_psf_library(num_psfs=100, grid_size=127):
    print(f"📡 Generating Master PSF Library ({num_psfs} PSFs)...")
    library = np.zeros((num_psfs, grid_size, grid_size), dtype=np.float32)
    half = grid_size // 2
    
    optical_template = None
    if os.path.exists("roman_psf_prior.pt"):
        try:
            optical_template = torch.load("roman_psf_prior.pt", map_location='cpu', weights_only=True).numpy()
            print("🛰️ Using roman_psf_prior.pt as optical template")
        except Exception as e:
            print(f"⚠️ Failed to load template: {e}")

    y, x = np.meshgrid(np.arange(grid_size) - half, np.arange(grid_size) - half, indexing='ij')
    
    for i in range(num_psfs):
        fx, fy = np.random.uniform(-2048, 2048), np.random.uniform(-2048, 2048)
        r_norm = np.sqrt(fx**2 + fy**2) / 2896.0
        q = np.random.uniform(0.85, 1.0) - (0.15 * r_norm)
        s_jit = np.random.uniform(0.3, 0.6) + (0.2 * r_norm)
        theta = np.arctan2(fy, fx) + np.random.normal(0, 0.2)
        cos, sin = np.cos(theta), np.sin(theta)
        xp, yp = x * cos + y * sin, -x * sin + y * cos
        jitter_kernel = np.exp(-(xp**2 / (2 * s_jit**2) + yp**2 / (2 * (s_jit * q)**2)))
        jitter_kernel /= (jitter_kernel.sum() + 1e-9)
        
        if optical_template is not None:
            from scipy.ndimage import rotate
            rotated = rotate(optical_template, np.random.uniform(0, 360), reshape=False, order=3, mode='constant', cval=0.0)
            psf = fftconvolve(rotated, jitter_kernel, mode='same')
        else:
            psf = jitter_kernel
            
        psf = np.maximum(0, psf)
        library[i] = psf / (psf.sum() + 1e-9)
    return library

def _compute_eigen_psfs(large_library, n_components=20):
    N, H, W = large_library.shape
    data = torch.from_numpy(large_library).float().view(N, H * W)
    mean_psf = data.mean(dim=0)
    centered_data = data - mean_psf
    U, S, V = torch.pca_lowrank(centered_data, q=n_components)
    eigen_psfs = V.t().view(n_components, H, W).numpy()
    psf_weights = (U * S).numpy() 
    return eigen_psfs, psf_weights, mean_psf.view(H, W).numpy()

def scatter_bincount(mosaic_size, flat_indices, weights):
    return np.bincount(flat_indices, weights=weights, minlength=mosaic_size*mosaic_size).reshape(mosaic_size, mosaic_size)

def generate_mosaic(idx, output_dir, params, mosaic_size, cell_size, master_psf_data):
    start_time = time.time()
    area_ratio = (mosaic_size / params['image_size'])**2
    exp_time, zp, sky_mag = np.random.uniform(30.0, 60.0), 26.5, 22.0
    mag_limit = calculate_safe_magnitude_cutoff(exp_time, zp, sky_mag, snr_cutoff=2.0)
    
    n_stars_total = int(10 ** np.random.uniform(np.log10(params['min_stars'] * area_ratio), np.log10(params['max_stars'] * area_ratio)))
    print(f"📦 Mosaic {idx}: Rendering {n_stars_total:,} stars...")
    
    mags = sample_bulge_magnitudes(n_stars_total, np.random.uniform(14.5, 16.5), np.random.uniform(0.2, 0.5), np.random.uniform(5.0, 15.0), m_min=12.0, m_max=32.0, gamma=np.random.uniform(0.25, 0.35))
    fluxes = exp_time * (10 ** (-0.4 * (mags - zp)))
    
    eigen_psfs, psf_weights_lib, mean_psf = master_psf_data
    
    # Decide path: JAX only if GPU is available
    use_jax = HAS_JAX and is_gpu_available()
    
    if use_jax:
        full_image, x_v, y_v, psf_indices, flux_v, mag_v, final_weights_v = render_generate_and_filter_gpu(
            fluxes, mags, psf_weights_lib, mean_psf, eigen_psfs, mosaic_size, mag_limit=mag_limit
        )
    else:
        # --- Optimized Hybrid NumPy/Numba Path ---
        px, py = np.random.uniform(0, mosaic_size, len(fluxes)), np.random.uniform(0, mosaic_size, len(fluxes))
        all_psf_indices = np.random.randint(0, 100, size=len(fluxes))
        
        # 1. Base Pass (All Stars convolved with Mean PSF)
        # 4-pixel bilinear scattering into a single grid
        x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
        dx, dy = px - x0, py - y0
        valid = (x0 >= 0) & (x0 < mosaic_size-1) & (y0 >= 0) & (y0 < mosaic_size-1)
        
        indices = np.concatenate([y0[valid] * mosaic_size + x0[valid], 
                                  y0[valid] * mosaic_size + x0[valid] + 1,
                                  (y0[valid]+1) * mosaic_size + x0[valid],
                                  (y0[valid]+1) * mosaic_size + x0[valid] + 1])
        
        f_v = fluxes[valid]
        vals = np.concatenate([f_v * (1-dx[valid]) * (1-dy[valid]),
                               f_v * dx[valid] * (1-dy[valid]),
                               f_v * (1-dx[valid]) * dy[valid],
                               f_v * dx[valid] * dy[valid]])
        
        base_grid = scatter_bincount(mosaic_size, indices, vals)
        
        with scipy.fft.set_backend(scipy.fft, workers=-1):
            full_image = fftconvolve(base_grid, mean_psf, mode='same')
            
        # 2. Eigen Correction Pass (Bright Stars Only - Numba Direct Paint)
        is_bright = mags < mag_limit
        if is_bright.any():
            b_px, b_py, b_f = px[is_bright], py[is_bright], fluxes[is_bright]
            b_weights = psf_weights_lib[all_psf_indices[is_bright]]
            
            # Pre-calculate unique Eigen-PSF sum for each bright star
            # Correction PSF = Sum(w_j * eigen_psf_j)
            # [N_bright, 20] @ [20, 127*127] -> [N_bright, 127*127]
            correction_stamps = (b_weights @ eigen_psfs.reshape(N_PCA_COMPONENTS, -1)).reshape(-1, SHAPE_SIZE, SHAPE_SIZE)
            
            # Ultra-fast Numba painting (Reduces 20 FFTs to ~0.5s of compute)
            _numba_paint_hybrid(full_image, b_px, b_py, b_f, correction_stamps)

        full_image = np.maximum(0, full_image)
        v_mask = mags < mag_limit
        x_v, y_v, flux_v, mag_v, final_weights_v = px[v_mask], py[v_mask], fluxes[v_mask], mags[v_mask], psf_weights_lib[all_psf_indices[v_mask]]

    cat_dtype = [('x', 'f4'), ('y', 'f4'), ('flux', 'f4'), ('mag', 'f4'), ('snr', 'f4')] + [(f'w{i}', 'f2') for i in range(N_PCA_COMPONENTS)]
    structured_cat = np.zeros(len(x_v), dtype=cat_dtype)
    structured_cat['x'], structured_cat['y'], structured_cat['flux'], structured_cat['mag'] = x_v, y_v, flux_v, mag_v
    for i in range(N_PCA_COMPONENTS): structured_cat[f'w{i}'] = final_weights_v[:, i]

    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (0.11**2) * exp_time
    total_local = full_image[np.clip(y_v.astype(int), 0, mosaic_size-1), np.clip(x_v.astype(int), 0, mosaic_size-1)]
    noise_var = flux_v + (4 * np.pi * 1.5**2) * (sky_level + np.maximum(0, total_local - flux_v*0.07) + 25.0)
    structured_cat['snr'] = flux_v / np.sqrt(np.maximum(1.0, noise_var))
    structured_cat = structured_cat[np.argsort(structured_cat['y'])]

    for suffix, data in [('img', full_image), ('cat', structured_cat), ('meta', np.array([exp_time, zp, sky_mag])), ('psf_lib', np.concatenate([eigen_psfs.reshape(N_PCA_COMPONENTS, -1), mean_psf.reshape(1, -1)], axis=0))]:
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
        master_data = torch.load(args.psf_library, map_location='cpu')
        if isinstance(master_data, dict):
            master_psf_data = (master_data['eigen_psfs'], master_data['weights_lib'], master_data['mean_psf'])
        else:
            master_psf_data = master_data
    else:
        kb_array = generate_field_realistic_psf_library(num_psfs=100, grid_size=SHAPE_SIZE)
        master_psf_data = _compute_eigen_psfs(kb_array, n_components=N_PCA_COMPONENTS)

    num = args.num if args.num else stage_cfg["mosaic_params"]["num_mosaics"]
    out = args.output_dir if args.output_dir else os.path.join(stage_cfg["data_dir"], "mosaics")
    os.makedirs(out, exist_ok=True)
    
    params = {"min_stars": cfg["data_params"]["min_stars"], "max_stars": cfg["data_params"]["max_stars"], "image_size": cfg["data_params"]["image_size"]}
    for i in range(num): generate_mosaic(i, out, params, stage_cfg["mosaic_params"]["mosaic_size"], stage_cfg["cell_size"], master_psf_data)

if __name__ == "__main__":
    main()
