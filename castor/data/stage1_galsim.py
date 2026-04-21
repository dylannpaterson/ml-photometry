import numpy as np
import os
import h5py
import galsim
import yaml
import argparse
import gc
import time
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve
from scipy.special import erf
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

from castor.data.stage0_gaussian import fast_paint_grid, sample_bulge_magnitudes, generate_dust_cirrus
from castor.constants import DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL, SHAPE_SIZE

# 🚀 Relaxed precision for massive speed and memory gains
RELAXED_GSP = galsim.GSParams(
    maximum_fft_size=4096,      # Increased to 4096 to handle mid-sized PSFs without warnings
    folding_threshold=1e-1,    # Very loose to prevent massive padding
    kvalue_accuracy=1e-1,      
    maxk_threshold=1e-1,       
    realspace_relerr=1e-1,
    realspace_abserr=1e-1
)

def calculate_dynamic_magnitude_cutoff(exp_time, zp, sky_mag, pixel_scale, read_noise=5.0, sigma=0.405, snr_cutoff=1.0):
    """Dynamically calculates the confusion limit based on randomized pixel scale and FWHM."""
    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (pixel_scale**2) * exp_time
    n_pix = max(4.0, 4.0 * np.pi * sigma**2) 
    bg_variance = n_pix * (sky_level + read_noise**2)
    a, b, c = 1.0, -(snr_cutoff**2), -(snr_cutoff**2) * bg_variance
    discriminant = max(0.0, b**2 - 4*a*c) 
    min_flux = (-b + np.sqrt(discriminant)) / (2*a)
    return zp - 2.5 * np.log10(min_flux / exp_time)

def generate_random_galsim_psf(bounds):
    """Builds a randomized GalSim PSF with optical aberrations and jitter."""
    z_max = bounds['zernike_max']
    defocus = np.random.uniform(-z_max, z_max)
    astig1 = np.random.uniform(-z_max, z_max)
    astig2 = np.random.uniform(-z_max, z_max)
    coma1 = np.random.uniform(-z_max, z_max)
    coma2 = np.random.uniform(-z_max, z_max)
    
    profile_type = np.random.choice(['moffat', 'airy', 'gaussian', 'kolmogorov'])
    fwhm = np.random.uniform(bounds['fwhm_min'], bounds['fwhm_max'])
    
    obscuration = 0.0
    num_struts = 0
    
    if profile_type == 'moffat':
        beta = np.random.uniform(bounds['moffat_beta_min'], bounds['moffat_beta_max'])
        psf = galsim.Moffat(beta=beta, fwhm=fwhm, gsparams=RELAXED_GSP)
    elif profile_type == 'airy':
        lam_over_diam = fwhm / 1.02 
        obscuration = np.random.uniform(bounds['obscuration_min'], bounds['obscuration_max'])
        num_struts = int(np.random.choice([0, 3, 4]))
        psf = galsim.OpticalPSF(
            lam_over_diam=lam_over_diam,
            defocus=defocus, astig1=astig1, astig2=astig2, coma1=coma1, coma2=coma2,
            obscuration=obscuration, nstruts=num_struts,
            strut_thick=np.random.uniform(0, bounds['strut_thick_max']),
            strut_angle=np.random.uniform(0, 360) * galsim.degrees,
            gsparams=RELAXED_GSP
        )
    elif profile_type == 'gaussian':
        psf = galsim.Gaussian(fwhm=fwhm, gsparams=RELAXED_GSP)
    else: # kolmogorov
        psf = galsim.Kolmogorov(fwhm=fwhm, gsparams=RELAXED_GSP)
        
    max_shear = bounds['max_shear']
    g1, g2 = np.random.uniform(-max_shear, max_shear), np.random.uniform(-max_shear, max_shear)
    psf = psf.shear(g1=g1, g2=g2)
    
    jitter_sigma = np.random.uniform(0, bounds['jitter_sigma_max'])
    if jitter_sigma > 0:
        psf = galsim.Convolve([psf, galsim.Gaussian(sigma=jitter_sigma, gsparams=RELAXED_GSP)], gsparams=RELAXED_GSP)
        
    return psf, {
        'defocus': defocus, 'astig1': astig1, 'astig2': astig2, 'coma1': coma1, 'coma2': coma2,
        'fwhm': fwhm, 'g1': g1, 'g2': g2, 'jitter_sigma': jitter_sigma,
        'obscuration': obscuration, 'num_struts': num_struts
    }

def psf_producer(queue, bounds, pixel_scale_range, count):
    """
    Dedicated process to generate random GalSim PSFs and put them in a queue.
    This serializes the heavy PSF creation (OpticalPSF) to avoid filesystem/resource contention.
    """
    # Seed for each producer to ensure unique PSFs
    np.random.seed(int(time.time() * 1000 + os.getpid()) % (2**32))
    
    produced = 0
    while produced < count:
        try:
            random_pixel_scale = np.random.uniform(pixel_scale_range[0], pixel_scale_range[1])
            random_psf, psf_params = generate_random_galsim_psf(bounds)
            
            # Draw at 2x native scale for high-fidelity interpolation later
            oversample = 2.0
            
            # Use a smaller nx, ny to force GalSim to truncate instead of blowing up FFT
            # This is safe because we only need the core PSF for the training chunks
            cached_psf_image = random_psf.drawImage(
                scale=random_pixel_scale / oversample, 
                nx=SHAPE_SIZE*2, ny=SHAPE_SIZE*2,
                method='auto'
            )
            
            psf_data = {
                'array': cached_psf_image.array.copy(),
                'pixel_scale': random_pixel_scale,
                'oversample': oversample,
                'params': psf_params
            }
            queue.put(psf_data)
            produced += 1
            
            if produced % 100 == 0: gc.collect()
            
        except Exception as e:
            # Skip "Mega PSFs" that violate GSParams or other constraints
            continue

def generate_single_chunk(idx, config, psf_queue):
    """Worker function to generate a single data chunk using a PSF from the queue."""
    s1_cfg = config['curriculum']['stage1']
    d_cfg = config['data_params']
    image_size = d_cfg['image_size']
    cell_size = s1_cfg.get('cell_size', DEFAULT_CELL_SIZE)
    grid_size = image_size // cell_size
    K = d_cfg['max_capacity_per_cell']

    # --- 🛰️ CONSUME PSF FROM QUEUE ---
    psf_data = psf_queue.get()
    psf_array = psf_data['array']
    random_pixel_scale = psf_data['pixel_scale']
    oversample = psf_data['oversample']
    psf_params = psf_data['params']

    cached_psf_image = galsim.ImageF(psf_array, scale=random_pixel_scale / oversample)
    fast_psf = galsim.InterpolatedImage(cached_psf_image, x_interpolant='lanczos3', gsparams=RELAXED_GSP)

    # --- 🚀 PREDICTIVE INITIALIZATION ---
    # ML Target: Average stars per cell
    # Sampling up to K ensures we cover the network's full capacity
    target_lambda = np.random.uniform(0.2, 3.0)
    target_n_active = int(target_lambda * (grid_size ** 2))
    
    # Physics & LF Parameters (Sampled per chunk)
    exp_time = np.random.uniform(d_cfg['physics_params']['exp_time_min'], d_cfg['physics_params']['exp_time_max'])
    zp = d_cfg['physics_params']['zp']
    sky_mag = np.random.uniform(s1_cfg['optical_randomization']['sky_mag_min'], s1_cfg['optical_randomization']['sky_mag_max'])
    read_noise = np.random.uniform(s1_cfg['optical_randomization']['read_noise_min'], s1_cfg['optical_randomization']['read_noise_max'])
    max_extinction = np.random.uniform(0.0, 5.0)
    
    lf = d_cfg['lf_params']
    gamma = np.random.uniform(lf['gamma_min'], lf['gamma_max'])
    rc_loc = np.random.uniform(lf['rc_loc_min'], lf['rc_loc_max'])
    rc_scale = np.random.uniform(lf['rc_scale_min'], lf['rc_scale_max'])
    rc_enhancement = np.random.uniform(lf['rc_enh_min'], lf['rc_enh_max'])
    m_min, m_max = lf['m_min'], lf['m_max']

    mag_limit = calculate_dynamic_magnitude_cutoff(exp_time, zp, sky_mag, random_pixel_scale, read_noise=read_noise, sigma=psf_params['fwhm'] / 2.355)

    # Analytic active fraction: N(<m) ~ 10^(gamma*m)
    def get_lf_frac(m_lim):
        return (10**(gamma * m_lim) - 10**(gamma * m_min)) / (10**(gamma * m_max) - 10**(gamma * m_min))
    
    f_active_base = get_lf_frac(mag_limit)
    f_rc_bin = get_lf_frac(rc_loc + 0.5) - get_lf_frac(rc_loc - 0.5)
    
    # Total expected active stars per base star generated (including RC boost)
    expected_active_per_star = f_active_base + (f_rc_bin * rc_enhancement)
    
    n_pool = int(target_n_active / np.maximum(expected_active_per_star, 1e-6))
    n_pool = np.clip(n_pool, 100, 1_000_000) 

    # --- POPULATION GENERATION ---
    mags = sample_bulge_magnitudes(n_pool, rc_loc, rc_scale, rc_enhancement, m_min=m_min, m_max=m_max, gamma=gamma)
    
    buffer = 10
    chunk_fov_arcsec = (image_size + 2 * buffer) * random_pixel_scale
    px = np.random.uniform(-chunk_fov_arcsec/2, chunk_fov_arcsec/2, len(mags)) / random_pixel_scale + image_size / 2.0
    py = np.random.uniform(-chunk_fov_arcsec/2, chunk_fov_arcsec/2, len(mags)) / random_pixel_scale + image_size / 2.0
    
    mask = (px > -buffer) & (px < image_size + buffer) & (py > -buffer) & (py < image_size + buffer)
    mags, px, py = mags[mask], px[mask], py[mask]
    fluxes = exp_time * (10 ** (-0.4 * (mags - zp)))
    
    if max_extinction > 0.0:
        raw_dust = generate_dust_cirrus(image_size, 1.0)
        transmission_map = 10 ** (-0.4 * raw_dust * max_extinction)
        star_transmissions = map_coordinates(transmission_map, [np.clip(py, 0, image_size-1), np.clip(px, 0, image_size-1)], order=1, mode='nearest')
        apparent_fluxes = fluxes * star_transmissions
        apparent_mags = mags - 2.5 * np.log10(np.maximum(star_transmissions, 1e-9))
        sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (random_pixel_scale**2) * exp_time
        total_sky_map = sky_level * (0.4 + 0.6 * transmission_map) + raw_dust * np.random.uniform(20, 100)
    else:
        apparent_fluxes, apparent_mags = fluxes.copy(), mags.copy()
        sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (random_pixel_scale**2) * exp_time
        total_sky_map = np.full((image_size, image_size), sky_level)

    # --- FINAL RENDERING ---
    v_mask = apparent_mags < mag_limit
    full_image_obj = galsim.ImageF(image_size, image_size, scale=random_pixel_scale)
    # Render Active stars
    for star_idx in np.where(v_mask)[0]:
        star_profile = fast_psf.withFlux(apparent_fluxes[star_idx])
        pos = galsim.PositionD(px[star_idx] + 1.0, py[star_idx] + 1.0)
        star_profile.drawImage(image=full_image_obj, center=pos, add_to_image=True, method='auto')
        
    convolved_bg_array = np.zeros((image_size, image_size), dtype=np.float32)
    if np.any(~v_mask):
        bg_grid = np.zeros((image_size, image_size), dtype=np.float32)
        px_bg, py_bg, f_bg = px[~v_mask], py[~v_mask], apparent_fluxes[~v_mask]
        in_img = (px_bg >= 0) & (px_bg < image_size - 1) & (py_bg >= 0) & (py_bg < image_size - 1)
        px_bg, py_bg, f_bg = px_bg[in_img], py_bg[in_img], f_bg[in_img]
        x0_bg, y0_bg = np.floor(px_bg).astype(int), np.floor(py_bg).astype(int)
        dx_bg, dy_bg = px_bg - x0_bg, py_bg - y0_bg
        w00, w10, w01, w11 = (1 - dx_bg) * (1 - dy_bg), dx_bg * (1 - dy_bg), (1 - dx_bg) * dy_bg, dx_bg * dy_bg
        
        def paint_bg(grid, x, y, w, f):
            mask = (x >= 0) & (x < image_size) & (y >= 0) & (y < image_size)
            flat_idx = y[mask] * image_size + x[mask]
            grid.flat += np.bincount(flat_idx, weights=f[mask] * w[mask], minlength=grid.size)

        paint_bg(bg_grid, x0_bg, y0_bg, w00, f_bg)
        paint_bg(bg_grid, x0_bg + 1, y0_bg, w10, f_bg)
        paint_bg(bg_grid, x0_bg, y0_bg + 1, w01, f_bg)
        paint_bg(bg_grid, x0_bg + 1, y0_bg + 1, w11, f_bg)

        psf_kernel_img = galsim.ImageF(SHAPE_SIZE, SHAPE_SIZE, scale=random_pixel_scale)
        fast_psf.drawImage(image=psf_kernel_img, method='auto')
        convolved_bg_array = fftconvolve(bg_grid, psf_kernel_img.array, mode='same').astype(np.float32)
        full_image_obj.array[:] += convolved_bg_array

    full_image = full_image_obj.array + total_sky_map
    chunk_median = np.median(full_image)
    bg_downsampled = (convolved_bg_array + total_sky_map).reshape(grid_size, cell_size, grid_size, cell_size).mean(axis=(1, 3))
    
    # Calculate SNRs for target grid
    local_bg = map_coordinates(total_sky_map, [np.clip(py, 0, image_size-1), np.clip(px, 0, image_size-1)], order=1, mode='nearest')
    snrs = apparent_fluxes / np.sqrt(np.maximum(1e-9, apparent_fluxes + 12.0 * (local_bg + read_noise**2)))
    
    in_fov = (px >= 0) & (px < image_size) & (py >= 0) & (py < image_size)
    target_grid = fast_paint_grid(px[in_fov], py[in_fov], apparent_fluxes[in_fov], snrs[in_fov], np.argsort(-apparent_fluxes[in_fov]), d_cfg['min_snr'], grid_size, cell_size, K)
    
    g1, g2 = psf_params['g1'], psf_params['g2']
    meta = np.array([exp_time, zp, sky_mag, psf_params['fwhm'], np.sqrt(g1**2 + g2**2), psf_params['obscuration'], float(psf_params['num_struts']), random_pixel_scale, psf_params['defocus'], np.sqrt(psf_params['astig1']**2 + psf_params['astig2']**2), np.sqrt(psf_params['coma1']**2 + psf_params['coma2']**2), psf_params['jitter_sigma'], read_noise, max_extinction, gamma, rc_loc, rc_enhancement], dtype=np.float32)

    # REFACTORED: Save the absolute, linear background expectation (Physical Space)
    target_grid_full = np.concatenate([
        target_grid.reshape(grid_size, grid_size, -1), 
        bg_downsampled[:, :, None]
    ], axis=-1).astype(np.float32)
    
    del full_image_obj, convolved_bg_array, bg_downsampled, total_sky_map
    gc.collect()
    return full_image.astype(np.float32), target_grid_full, chunk_median.astype(np.float32), meta

def run_stage1_generation(config_path="config/config.yaml", num_samples=None, num_workers=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    s1_cfg, d_cfg = config['curriculum']['stage1'], config['data_params']
    total_samples = num_samples if num_samples is not None else (d_cfg['num_train_samples'] + d_cfg['num_val_samples'])
    os.makedirs(s1_cfg['data_dir'], exist_ok=True)
    
    output_path = os.path.join(s1_cfg['data_dir'], "stage1_data.h5")
    
    if num_workers is None: 
        num_workers = mp.cpu_count()
    
    # --- 🏗️ SETUP PRODUCER-CONSUMER MODEL ---
    # We use 4 producers to handle the heavy PSF creation (OpticalPSF)
    n_producers = 4
    # Leave 1 worker for the main process to handle HDF5 writing
    n_consumers = max(1, num_workers - n_producers - 1)
    
    # Cap consumers to 56 to utilize high-core nodes effectively without over-subscription
    n_consumers = min(56, n_consumers)
    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    print(f"🔥 Stage 1 Unified Generation: {n_producers} Producers + {n_consumers} Consumers + 1 HDF5 Writer")

    manager = mp.Manager()
    psf_queue = manager.Queue(maxsize=n_consumers * 4) # Deep buffer to keep workers fed

    # Start PSF Producers
    bounds = s1_cfg['optical_randomization']
    pixel_scale_range = (bounds['pixel_scale_min'], bounds['pixel_scale_max'])
    
    producers = []
    samples_per_producer = (total_samples // n_producers) + 1
    for p_idx in range(n_producers):
        p = mp.Process(target=psf_producer, args=(psf_queue, bounds, pixel_scale_range, samples_per_producer))
        p.start()
        producers.append(p)

    with h5py.File(output_path, 'w') as f:
        dset_img = f.create_dataset("images", (total_samples, d_cfg['image_size'], d_cfg['image_size']), dtype='f4', chunks=(1, d_cfg['image_size'], d_cfg['image_size']), compression="lzf")
        dset_tgt = f.create_dataset("targets", (total_samples, d_cfg['image_size']//s1_cfg.get('cell_size', DEFAULT_CELL_SIZE), d_cfg['image_size']//s1_cfg.get('cell_size', DEFAULT_CELL_SIZE), d_cfg['max_capacity_per_cell']*5 + 1), dtype='f4', chunks=(1, d_cfg['image_size']//s1_cfg.get('cell_size', DEFAULT_CELL_SIZE), d_cfg['image_size']//s1_cfg.get('cell_size', DEFAULT_CELL_SIZE), d_cfg['max_capacity_per_cell']*5 + 1), compression="lzf")
        dset_med = f.create_dataset("chunk_medians", (total_samples,), dtype='f4')
        dset_meta = f.create_dataset("metas", (total_samples, 17), dtype='f4')
        
        worker_func = partial(generate_single_chunk, config=config, psf_queue=psf_queue)
        with mp.Pool(n_consumers) as pool:
            results = pool.imap_unordered(worker_func, range(total_samples), chunksize=1)
            for i, (img, tgt, med, meta) in enumerate(tqdm(results, total=total_samples)):
                dset_img[i], dset_tgt[i], dset_med[i], dset_meta[i] = img, tgt, med, meta
                if i % 100 == 0: gc.collect()

    for p in producers:
        p.join()
    print(f"✅ Stage 1 Generation Complete: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Stage 1 GalSim Data")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()
    run_stage1_generation(config_path=args.config, num_samples=args.num_samples, num_workers=args.num_workers)
