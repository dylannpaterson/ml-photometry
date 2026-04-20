import numpy as np
import os
import h5py
import galsim
import yaml
import argparse
from scipy.ndimage import map_coordinates
from scipy.special import erf
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

from castor.data.stage0_gaussian import fast_paint_grid, sample_bulge_magnitudes, generate_dust_cirrus
from castor.constants import DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL

def calculate_dynamic_magnitude_cutoff(exp_time, zp, sky_mag, pixel_scale, read_noise=5.0, sigma=1.5, snr_cutoff=1.0):
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
        psf = galsim.Moffat(beta=beta, fwhm=fwhm)
    elif profile_type == 'airy':
        lam_over_diam = fwhm / 1.02 
        obscuration = np.random.uniform(bounds['obscuration_min'], bounds['obscuration_max'])
        num_struts = int(np.random.choice([0, 3, 4]))
        psf = galsim.OpticalPSF(
            lam_over_diam=lam_over_diam,
            defocus=defocus, astig1=astig1, astig2=astig2, coma1=coma1, coma2=coma2,
            obscuration=obscuration, nstruts=num_struts,
            strut_thick=np.random.uniform(0, bounds['strut_thick_max']),
            strut_angle=np.random.uniform(0, 360) * galsim.degrees
        )
    elif profile_type == 'gaussian':
        psf = galsim.Gaussian(fwhm=fwhm)
    else: # kolmogorov
        psf = galsim.Kolmogorov(fwhm=fwhm)
        
    max_shear = bounds['max_shear']
    g1, g2 = np.random.uniform(-max_shear, max_shear), np.random.uniform(-max_shear, max_shear)
    psf = psf.shear(g1=g1, g2=g2)
    
    jitter_sigma = np.random.uniform(0, bounds['jitter_sigma_max'])
    if jitter_sigma > 0:
        psf = galsim.Convolve([psf, galsim.Gaussian(sigma=jitter_sigma)])
        
    return psf, {
        'defocus': defocus, 'astig1': astig1, 'astig2': astig2, 'coma1': coma1, 'coma2': coma2,
        'fwhm': fwhm, 'g1': g1, 'g2': g2, 'jitter_sigma': jitter_sigma,
        'obscuration': obscuration, 'num_struts': num_struts
    }

def generate_single_chunk(idx, config):
    """Worker function to generate a single data chunk."""
    s1_cfg = config['curriculum']['stage1']
    d_cfg = config['data_params']
    bounds = s1_cfg['optical_randomization']
    image_size = d_cfg['image_size']
    cell_size = s1_cfg.get('cell_size', DEFAULT_CELL_SIZE)
    grid_size = image_size // cell_size
    K = d_cfg['max_capacity_per_cell']

    random_pixel_scale = np.random.uniform(bounds['pixel_scale_min'], bounds['pixel_scale_max'])
    
    # --- DYNAMIC DENSITY SCALING (Capacity Management) ---
    # target occupancy: average stars per cell brighter than 26
    # Aim for 0.5 to 2.5 stars per cell (model max is 3)
    target_occupancy = np.random.uniform(0.5, 2.5)
    
    cell_area_sq_deg = (cell_size * random_pixel_scale / 3600.0)**2
    # target_density_26 is stars per square degree
    target_density_26 = target_occupancy / cell_area_sq_deg
    
    # Clip to physical bounds from config if provided, but prioritize occupancy safety
    d26_min = d_cfg.get('density_26_min', 500_000)
    d26_max = d_cfg.get('density_26_max', 50_000_000)
    target_density_26 = np.clip(target_density_26, d26_min, d26_max)

    buffer = 10
    chunk_fov_arcsec = (image_size + 2 * buffer) * random_pixel_scale
    chunk_area_sq_deg = (chunk_fov_arcsec / 3600.0)**2
    expected_n_26 = target_density_26 * chunk_area_sq_deg
    
    # LF params
    lf = d_cfg['lf_params']
    gamma = np.random.uniform(lf['gamma_min'], lf['gamma_max'])
    rc_loc = np.random.uniform(lf['rc_loc_min'], lf['rc_loc_max'])
    rc_scale = np.random.uniform(lf['rc_scale_min'], lf['rc_scale_max'])
    rc_enhancement = np.random.uniform(lf['rc_enh_min'], lf['rc_enh_max'])
    m_min, m_max = lf['m_min'], lf['m_max']

    # Analytic LF Fraction calculation for n_base draw
    f_base_26 = (10**(gamma * 26.0) - 10**(gamma * m_min)) / (10**(gamma * m_max) - 10**(gamma * m_min))
    f_base_rc_bin = (10**(gamma * (rc_loc + 0.5)) - 10**(gamma * (rc_loc - 0.5))) / (10**(gamma * m_max) - 10**(gamma * m_min))
    f_rc_26 = 0.5 * (1.0 + erf((26.0 - rc_loc) / (rc_scale * np.sqrt(2.0))))
    
    n_base = int(expected_n_26 / (f_base_26 + f_base_rc_bin * rc_enhancement * f_rc_26))
    n_base = max(100, n_base)

    exp_time = np.random.uniform(d_cfg['physics_params']['exp_time_min'], d_cfg['physics_params']['exp_time_max'])
    zp = d_cfg['physics_params']['zp']
    sky_mag = np.random.uniform(bounds['sky_mag_min'], bounds['sky_mag_max'])
    read_noise = np.random.uniform(bounds['read_noise_min'], bounds['read_noise_max'])
    max_extinction = np.random.uniform(0.0, 5.0)
    
    random_psf, psf_params = generate_random_galsim_psf(bounds)
    
    # --- SPEED FIX: Lighter Interpolation ---
    cached_psf_image = random_psf.drawImage(scale=random_pixel_scale / 4.0, method='auto')
    fast_psf = galsim.InterpolatedImage(cached_psf_image, x_interpolant='lanczos3')
    
    mag_limit = calculate_dynamic_magnitude_cutoff(exp_time, zp, sky_mag, random_pixel_scale, read_noise=read_noise, sigma=psf_params['fwhm'] / 2.355)
    
    mags = sample_bulge_magnitudes(n_base, rc_loc, rc_scale, rc_enhancement, m_min=m_min, m_max=m_max, gamma=gamma)
    px_arcsec = np.random.uniform(-chunk_fov_arcsec/2, chunk_fov_arcsec/2, len(mags))
    py_arcsec = np.random.uniform(-chunk_fov_arcsec/2, chunk_fov_arcsec/2, len(mags))
    px = px_arcsec / random_pixel_scale + image_size / 2.0
    py = py_arcsec / random_pixel_scale + image_size / 2.0
    
    mask = (px > -buffer) & (px < image_size + buffer) & (py > -buffer) & (py < image_size + buffer)
    mags, px, py = mags[mask], px[mask], py[mask]
    fluxes = exp_time * (10 ** (-0.4 * (mags - zp)))
    
    if max_extinction > 0.0:
        raw_dust = generate_dust_cirrus(image_size, 1.0)
        # raw_dust is normalized [0,1] inside generate_dust_cirrus
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

    v_mask = apparent_mags < mag_limit
    full_image_obj = galsim.ImageF(image_size, image_size, scale=random_pixel_scale)
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
        
        flat_indices00 = y0_bg * image_size + x0_bg
        flat_indices10 = y0_bg * image_size + (x0_bg + 1)
        flat_indices01 = (y0_bg + 1) * image_size + x0_bg
        flat_indices11 = (y0_bg + 1) * image_size + (x0_bg + 1)
        
        bg_grid.flat += np.bincount(flat_indices00, weights=f_bg * w00, minlength=bg_grid.size)
        bg_grid.flat += np.bincount(flat_indices10, weights=f_bg * w10, minlength=bg_grid.size)
        bg_grid.flat += np.bincount(flat_indices01, weights=f_bg * w01, minlength=bg_grid.size)
        bg_grid.flat += np.bincount(flat_indices11, weights=f_bg * w11, minlength=bg_grid.size)

        bg_image_obj = galsim.InterpolatedImage(galsim.ImageF(bg_grid, scale=random_pixel_scale), x_interpolant='linear')
        convolved_bg_obj = galsim.Convolve([bg_image_obj, fast_psf])
        convolved_bg_img = galsim.ImageF(image_size, image_size, scale=random_pixel_scale)
        convolved_bg_obj.drawImage(image=convolved_bg_img, method='auto')
        convolved_bg_array = convolved_bg_img.array
        full_image_obj.array[:] += convolved_bg_array

    full_image = full_image_obj.array + total_sky_map
    bg_downsampled = (convolved_bg_array + total_sky_map).reshape(grid_size, cell_size, grid_size, cell_size).mean(axis=(1, 3))
    
    local_bg = map_coordinates(total_sky_map, [np.clip(py, 0, image_size-1), np.clip(px, 0, image_size-1)], order=1, mode='nearest')
    noise_variance = apparent_fluxes + 12.0 * (local_bg + read_noise**2)
    snrs = apparent_fluxes / np.sqrt(np.maximum(1e-9, noise_variance))
    
    in_fov = (px >= 0) & (px < image_size) & (py >= 0) & (py < image_size)
    target_grid = fast_paint_grid(px[in_fov], py[in_fov], apparent_fluxes[in_fov], snrs[in_fov], np.argsort(-apparent_fluxes[in_fov]), d_cfg['min_snr'], grid_size, cell_size, K)
    
    g1, g2 = psf_params['g1'], psf_params['g2']
    meta = np.array([exp_time, zp, sky_mag, psf_params['fwhm'], np.sqrt(g1**2 + g2**2), psf_params['obscuration'], float(psf_params['num_struts']), random_pixel_scale, psf_params['defocus'], np.sqrt(psf_params['astig1']**2 + psf_params['astig2']**2), np.sqrt(psf_params['coma1']**2 + psf_params['coma2']**2), psf_params['jitter_sigma'], read_noise, max_extinction, gamma, rc_loc, rc_enhancement], dtype=np.float32)

    return full_image.astype(np.float32), np.concatenate([target_grid.reshape(grid_size, grid_size, -1), bg_downsampled[:, :, None]], axis=-1).astype(np.float32), np.median(full_image).astype(np.float32), meta

def run_stage1_generation(config_path="config/config.yaml", num_samples=None, num_workers=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    s1_cfg, d_cfg = config['curriculum']['stage1'], config['data_params']
    total_samples = num_samples if num_samples is not None else (d_cfg['num_train_samples'] + d_cfg['num_val_samples'])
    os.makedirs(s1_cfg['data_dir'], exist_ok=True)
    
    output_path = os.path.join(s1_cfg['data_dir'], "stage1_data.h5")
    if num_workers is None: 
        num_workers = mp.cpu_count()
    
    # 1. Leave 1 core entirely free for HDF5 I/O and serialization overhead
    pool_workers = max(1, num_workers - 1)
    print(f"🔥 Parallelizing generation with {pool_workers} workers (1 core reserved for I/O)...")

    with h5py.File(output_path, 'w') as f:
        dset_img = f.create_dataset("images", (total_samples, d_cfg['image_size'], d_cfg['image_size']), dtype='f4')
        dset_tgt = f.create_dataset("targets", (total_samples, d_cfg['image_size']//s1_cfg.get('cell_size', DEFAULT_CELL_SIZE), d_cfg['image_size']//s1_cfg.get('cell_size', DEFAULT_CELL_SIZE), d_cfg['max_capacity_per_cell']*5 + 1), dtype='f4')
        dset_med = f.create_dataset("chunk_medians", (total_samples,), dtype='f4')
        dset_meta = f.create_dataset("metas", (total_samples, 17), dtype='f4')
        
        worker_func = partial(generate_single_chunk, config=config)
        with mp.Pool(pool_workers) as pool:
            # 2. Use imap_unordered to prevent memory buffering
            for i, (img, tgt, med, meta) in enumerate(tqdm(pool.imap_unordered(worker_func, range(total_samples)), total=total_samples)):
                dset_img[i], dset_tgt[i], dset_med[i], dset_meta[i] = img, tgt, med, meta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Stage 1 GalSim Data")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()
    run_stage1_generation(config_path=args.config, num_samples=args.num_samples, num_workers=args.num_workers)
