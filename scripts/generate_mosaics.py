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

def generate_field_realistic_psf_library(num_psfs=100, grid_size=127):
    """
    Generates a library of PRISTINE optical PSFs.
    Random jitter is NOT applied here anymore.
    """
    print(f"📡 Generating Master OPTICAL PSF Library ({num_psfs} PSFs)...")
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
        # Field distortions (Ellipticity increases with radius)
        q_opt = np.random.uniform(0.9, 1.0) - (0.1 * r_norm)
        theta = np.arctan2(fy, fx) + np.random.normal(0, 0.1)
        cos, sin = np.cos(theta), np.sin(theta)
        xp, yp = x * cos + y * sin, -x * sin + y * cos
        
        # Base optical core (represents Airy + simple aberrations)
        s_opt = 0.45 # Roman diffraction core ~0.45-0.5 pixels sigma
        opt_core = np.exp(-(xp**2 / (2 * s_opt**2) + yp**2 / (2 * (s_opt * q_opt)**2)))
        opt_core /= (opt_core.sum() + 1e-9)
        
        if optical_template is not None:
            from scipy.ndimage import rotate
            rotated = rotate(optical_template, np.random.uniform(0, 360), reshape=False, order=3, mode='constant', cval=0.0)
            # Convolve realistic optics with field-dependent core
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
    
    # -------------------------------------------------------------------------
    # 1. GENERATE GLOBAL EXPOSURE JITTER
    # -------------------------------------------------------------------------
    # Spacecraft vibration is global across the whole detector for one exposure.
    # Roman pointing requirement is 14 mas RMS per axis.
    # 14 mas / 110 mas/pixel = 0.127 pixels.
    s_jit = np.random.normal(0.127, 0.01) 
    q_jit = np.random.uniform(0.8, 1.0)
    theta_jit = np.random.uniform(0, np.pi)
    
    # Build the 2D jitter kernel
    k_half = SHAPE_SIZE // 2
    gy, gx = np.meshgrid(np.arange(SHAPE_SIZE) - k_half, np.arange(SHAPE_SIZE) - k_half, indexing='ij')
    cos, sin = np.cos(theta_jit), np.sin(theta_jit)
    gxp, gyp = gx * cos + gy * sin, -gx * sin + gy * cos
    jitter_kernel = np.exp(-(gxp**2 / (2 * s_jit**2) + gyp**2 / (2 * (s_jit * q_jit)**2)))
    jitter_kernel /= (jitter_kernel.sum() + 1e-9)
    # -------------------------------------------------------------------------

    use_jax = HAS_JAX and is_gpu_available()
    if use_jax:
        # Note: render_generate_and_filter_gpu needs to be updated to accept jitter or apply it
        # For simplicity, we apply jitter at the end via CPU FFT or a second JAX call.
        full_image, x_v, y_v, psf_indices, flux_v, mag_v, final_weights_v = render_generate_and_filter_gpu(
            fluxes, mags, psf_weights_lib, mean_psf, eigen_psfs, mosaic_size, mag_limit=mag_limit
        )
    else:
        px, py = np.random.uniform(0, mosaic_size, len(fluxes)), np.random.uniform(0, mosaic_size, len(fluxes))
        all_psf_indices = np.random.randint(0, 100, size=len(fluxes))
        x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
        dx, dy = px - x0, py - y0
        valid = (x0 >= 0) & (x0 < mosaic_size-1) & (y0 >= 0) & (y0 < mosaic_size-1)
        indices = np.concatenate([y0[valid] * mosaic_size + x0[valid], y0[valid] * mosaic_size + x0[valid] + 1, (y0[valid]+1) * mosaic_size + x0[valid], (y0[valid]+1) * mosaic_size + x0[valid] + 1])
        f_v = fluxes[valid]
        vals = np.concatenate([f_v * (1-dx[valid]) * (1-dy[valid]), f_v * dx[valid] * (1-dy[valid]), f_v * (1-dx[valid]) * dy[valid], f_v * dx[valid] * dy[valid]])
        base_grid = scatter_bincount(mosaic_size, indices, vals)
        full_image = fftconvolve(base_grid, mean_psf, mode='same')
        is_bright = mags < mag_limit
        if is_bright.any():
            b_px, b_py, b_f = px[is_bright], py[is_bright], fluxes[is_bright]
            b_weights = psf_weights_lib[all_psf_indices[is_bright]]
            correction_stamps = (b_weights @ eigen_psfs.reshape(N_PCA_COMPONENTS, -1)).reshape(-1, SHAPE_SIZE, SHAPE_SIZE)
            _numpy_paint_hybrid(full_image, b_px, b_py, b_f, correction_stamps)
        v_mask = mags < mag_limit
        x_v, y_v, flux_v, mag_v, psf_indices = px[v_mask], py[v_mask], fluxes[v_mask], mags[v_mask], all_psf_indices[v_mask]
        final_weights_v = psf_weights_lib[psf_indices]

    # --- APPLY GLOBAL JITTER ---
    print(f"🎬 Applying global exposure jitter (s={s_jit:.2f}, q={q_jit:.2f})...")
    full_image = fftconvolve(full_image, jitter_kernel, mode='same')
    full_image = np.maximum(0, full_image)

    # --- Accurately recalculate SNR under Jitter ---
    # We must account for the peak suppression caused by the jitter blur.
    # New effective peak = OpticalPeak convolved with JitterKernel
    lib_peaks_opt = mean_psf[k_half, k_half] + psf_weights_lib @ eigen_psfs[:, k_half, k_half]
    # Estimate peak reduction: approx proportional to AreaOpt / AreaEff
    eff_area_opt = 1.0 / np.sum(mean_psf**2)
    psf_eff_temp = fftconvolve(mean_psf, jitter_kernel, mode='same')
    eff_area_jit = 1.0 / np.sum(psf_eff_temp**2)
    peak_reduction_factor = eff_area_opt / eff_area_jit
    lib_peaks_jit = lib_peaks_opt * peak_reduction_factor

    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (0.11**2) * exp_time
    total_local_light = map_coordinates(full_image, [y_v, x_v], order=1, mode='nearest')
    star_own_peak = flux_v * lib_peaks_jit[psf_indices]
    confusion_bg = np.maximum(0, total_local_light - star_own_peak)
    noise_variance = flux_v + eff_area_jit * (sky_level + confusion_bg + 25.0)
    
    cat_dtype = [('x', 'f4'), ('y', 'f4'), ('flux', 'f4'), ('mag', 'f4'), ('snr', 'f4')] + [(f'w{i}', 'f2') for i in range(N_PCA_COMPONENTS)]
    structured_cat = np.zeros(len(x_v), dtype=cat_dtype)
    structured_cat['x'], structured_cat['y'], structured_cat['flux'], structured_cat['mag'] = x_v, y_v, flux_v, mag_v
    structured_cat['snr'] = flux_v / np.sqrt(np.maximum(1.0, noise_variance))
    for i in range(N_PCA_COMPONENTS): structured_cat[f'w{i}'] = final_weights_v[:, i]
    structured_cat = structured_cat[np.argsort(structured_cat['y'])]

    for suffix, data in [('img', full_image), ('cat', structured_cat), ('meta', np.array([exp_time, zp, sky_mag, s_jit, q_jit, theta_jit])), ('psf_lib', np.concatenate([eigen_psfs.reshape(N_PCA_COMPONENTS, -1), mean_psf.reshape(1, -1)], axis=0))]:
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
