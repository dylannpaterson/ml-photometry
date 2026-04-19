import numpy as np
import os
import h5py
import galsim
import yaml
import argparse
from scipy.ndimage import map_coordinates
from scipy.special import erf
from tqdm import tqdm

from castor.data.stage0_gaussian import fast_paint_grid, sample_bulge_magnitudes, generate_dust_cirrus
from castor.constants import DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL

def calculate_dynamic_magnitude_cutoff(exp_time, zp, sky_mag, pixel_scale, read_noise=5.0, sigma=1.5, snr_cutoff=1.0):
    """Dynamically calculates the confusion limit based on randomized pixel scale and FWHM."""
    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (pixel_scale**2) * exp_time
    
    # Estimate the number of pixels the core covers based on the random FWHM
    n_pix = max(4.0, 4.0 * np.pi * sigma**2) 
    
    bg_variance = n_pix * (sky_level + read_noise**2)
    a, b, c = 1.0, -(snr_cutoff**2), -(snr_cutoff**2) * bg_variance
    
    # Safeguard against negative discriminant in extreme noise
    discriminant = max(0.0, b**2 - 4*a*c) 
    min_flux = (-b + np.sqrt(discriminant)) / (2*a)
    
    return zp - 2.5 * np.log10(min_flux / exp_time)

def generate_random_galsim_psf(bounds):
    """
    Builds a randomized GalSim PSF with optical aberrations and jitter.
    """
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
        # lam/D for Airy is roughly fwhm / 1.02
        lam_over_diam = fwhm / 1.02 
        obscuration = np.random.uniform(bounds['obscuration_min'], bounds['obscuration_max'])
        num_struts = int(np.random.choice([0, 3, 4]))
        psf = galsim.OpticalPSF(
            lam_over_diam=lam_over_diam,
            defocus=defocus,
            astig1=astig1,
            astig2=astig2,
            coma1=coma1,
            coma2=coma2,
            obscuration=obscuration,
            nstruts=num_struts,
            strut_thick=np.random.uniform(0, bounds['strut_thick_max']),
            strut_angle=np.random.uniform(0, 360) * galsim.degrees
        )
    elif profile_type == 'gaussian':
        psf = galsim.Gaussian(fwhm=fwhm)
    else: # kolmogorov
        psf = galsim.Kolmogorov(fwhm=fwhm)
        
    max_shear = bounds['max_shear']
    g1 = np.random.uniform(-max_shear, max_shear)
    g2 = np.random.uniform(-max_shear, max_shear)
    psf = psf.shear(g1=g1, g2=g2)
    
    jitter_sigma = np.random.uniform(0, bounds['jitter_sigma_max'])
    if jitter_sigma > 0:
        psf = galsim.Convolve([psf, galsim.Gaussian(sigma=jitter_sigma)])
        
    return psf, {
        'defocus': defocus,
        'astig1': astig1,
        'astig2': astig2,
        'coma1': coma1,
        'coma2': coma2,
        'fwhm': fwhm,
        'g1': g1,
        'g2': g2,
        'jitter_sigma': jitter_sigma,
        'obscuration': obscuration,
        'num_struts': num_struts
    }

def run_stage1_generation(config_path="config/config.yaml", num_samples=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    s1_cfg = config['curriculum']['stage1']
    d_cfg = config['data_params']
    bounds = s1_cfg['optical_randomization']
    
    os.makedirs(s1_cfg['data_dir'], exist_ok=True)
    
    if num_samples is None:
        num_train = d_cfg['num_train_samples']
        num_val = d_cfg['num_val_samples']
        total_samples = num_train + num_val
    else:
        total_samples = num_samples
    
    image_size = d_cfg['image_size']
    cell_size = s1_cfg.get('cell_size', DEFAULT_CELL_SIZE)
    grid_size = image_size // cell_size
    K = d_cfg['max_capacity_per_cell']
    
    output_path = os.path.join(s1_cfg['data_dir'], "stage1_data.h5")
    
    with h5py.File(output_path, 'w') as f:
        dset_img = f.create_dataset("images", (total_samples, image_size, image_size), dtype='f4')
        dset_tgt = f.create_dataset("targets", (total_samples, grid_size, grid_size, K*5 + 1), dtype='f4')
        dset_med = f.create_dataset("chunk_medians", (total_samples,), dtype='f4')
        dset_meta = f.create_dataset("metas", (total_samples, 17), dtype='f4')
        
        for i in tqdm(range(total_samples)):
            print(f"\n🚀 Rendering Chunk {i+1}/{total_samples}...")
            # Step 2A: Randomize the Universe
            random_pixel_scale = np.random.uniform(bounds['pixel_scale_min'], bounds['pixel_scale_max'])
            
            # Draw LF params early for density calculation
            lf = d_cfg['lf_params']
            gamma = np.random.uniform(lf['gamma_min'], lf['gamma_max'])
            rc_loc = np.random.uniform(lf['rc_loc_min'], lf['rc_loc_max'])
            rc_scale = np.random.uniform(lf['rc_scale_min'], lf['rc_scale_max'])
            rc_enhancement = np.random.uniform(lf['rc_enh_min'], lf['rc_enh_max'])
            m_min, m_max = lf['m_min'], lf['m_max']

            # Target density: stars / sq deg brighter than 26
            target_density_26 = np.random.uniform(d_cfg['density_26_min'], d_cfg['density_26_max'])
            
            buffer = 10
            chunk_fov_arcsec = (image_size + 2 * buffer) * random_pixel_scale
            chunk_area_sq_deg = (chunk_fov_arcsec / 3600.0)**2
            expected_n_26 = target_density_26 * chunk_area_sq_deg
            
            # LF Fraction Brighter than 26
            f_base_26 = (10**(gamma * 26.0) - 10**(gamma * m_min)) / (10**(gamma * m_max) - 10**(gamma * m_min))
            f_base_rc_bin = (10**(gamma * (rc_loc + 0.5)) - 10**(gamma * (rc_loc - 0.5))) / (10**(gamma * m_max) - 10**(gamma * m_min))
            f_rc_26 = 0.5 * (1.0 + erf((26.0 - rc_loc) / (rc_scale * np.sqrt(2.0))))
            
            n_base = int(expected_n_26 / (f_base_26 + f_base_rc_bin * rc_enhancement * f_rc_26))
            n_base = max(100, n_base)

            # Other params
            exp_time = np.random.uniform(d_cfg['physics_params']['exp_time_min'], d_cfg['physics_params']['exp_time_max'])
            zp = d_cfg['physics_params']['zp']
            sky_mag = np.random.uniform(bounds['sky_mag_min'], bounds['sky_mag_max'])
            read_noise = np.random.uniform(bounds['read_noise_min'], bounds['read_noise_max'])
            max_extinction = np.random.uniform(0.0, 5.0)
            
            # Build Randomized GalSim PSF
            random_psf, psf_params = generate_random_galsim_psf(bounds)
            
            # --- THE FIX: PSF CACHING ---
            cached_psf_image = random_psf.drawImage(scale=random_pixel_scale / 4.0, method='auto')
            fast_psf = galsim.InterpolatedImage(cached_psf_image, x_interpolant='lanczos15')
            
            # Use the new dynamic cutoff!
            mag_limit = calculate_dynamic_magnitude_cutoff(
                exp_time, zp, sky_mag, random_pixel_scale, 
                read_noise=read_noise, 
                sigma=psf_params['fwhm'] / 2.355, 
                snr_cutoff=1.0
            )
            
            # Generate local catalog
            mags = sample_bulge_magnitudes(
                n_base, rc_loc, rc_scale, rc_enhancement,
                m_min=m_min, m_max=m_max, gamma=gamma
            )
            
            px_arcsec = np.random.uniform(-chunk_fov_arcsec/2, chunk_fov_arcsec/2, len(mags))
            py_arcsec = np.random.uniform(-chunk_fov_arcsec/2, chunk_fov_arcsec/2, len(mags))
            
            px = px_arcsec / random_pixel_scale + image_size / 2.0
            py = py_arcsec / random_pixel_scale + image_size / 2.0
            
            mask = (px > -buffer) & (px < image_size + buffer) & (py > -buffer) & (py < image_size + buffer)
            mags = mags[mask]
            px = px[mask]
            py = py[mask]
            
            fluxes = exp_time * (10 ** (-0.4 * (mags - zp)))
            
            # Dust Attenuation
            if max_extinction > 0.0:
                raw_dust = generate_dust_cirrus(image_size, 1.0)
                transmission_map = 10 ** (-0.4 * raw_dust * max_extinction)
                star_transmissions = map_coordinates(transmission_map, [np.clip(py, 0, image_size-1), np.clip(px, 0, image_size-1)], order=1, mode='nearest')
                apparent_fluxes = fluxes * star_transmissions
                apparent_mags = mags - 2.5 * np.log10(np.maximum(star_transmissions, 1e-9))
                
                frac_bg = 0.60
                sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (random_pixel_scale**2) * exp_time
                sky_foreground = sky_level * (1.0 - frac_bg)
                sky_background_attenuated = (sky_level * frac_bg) * transmission_map
                total_sky_map = sky_foreground + sky_background_attenuated
                additive_dust_emission = raw_dust * np.random.uniform(20, 100)
            else:
                apparent_fluxes = fluxes.copy()
                apparent_mags = mags.copy()
                transmission_map = np.ones((image_size, image_size))
                sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (random_pixel_scale**2) * exp_time
                total_sky_map = np.full((image_size, image_size), sky_level)
                additive_dust_emission = np.zeros((image_size, image_size))

            # Step 2F: Hybrid Rendering Optimization
            v_mask = apparent_mags < mag_limit
            full_image_obj = galsim.ImageF(image_size, image_size, scale=random_pixel_scale)
            
            # 1. Individual rendering for bright stars (Now insanely fast)
            for star_idx in np.where(v_mask)[0]:
                star_profile = fast_psf.withFlux(apparent_fluxes[star_idx])
                pos = galsim.PositionD(px[star_idx] + 1.0, py[star_idx] + 1.0)
                star_profile.drawImage(image=full_image_obj, center=pos, add_to_image=True, method='auto')
                
            # 2. Grid-based rendering for ultra-faint background stars
            convolved_bg_array = np.zeros((image_size, image_size), dtype=np.float32)
            if np.any(~v_mask):
                bg_grid = np.zeros((image_size, image_size), dtype=np.float32)
                px_bg, py_bg = px[~v_mask], py[~v_mask]
                f_bg = apparent_fluxes[~v_mask]
                in_img = (px_bg >= 0) & (px_bg < image_size - 1) & (py_bg >= 0) & (py_bg < image_size - 1)
                px_bg, py_bg, f_bg = px_bg[in_img], py_bg[in_img], f_bg[in_img]
                
                x0_bg, y0_bg = np.floor(px_bg).astype(int), np.floor(py_bg).astype(int)
                dx_bg, dy_bg = px_bg - x0_bg, py_bg - y0_bg
                w00, w10, w01, w11 = (1 - dx_bg) * (1 - dy_bg), dx_bg * (1 - dy_bg), (1 - dx_bg) * dy_bg, dx_bg * dy_bg
                
                def paint_flux(grid, x, y, w, f):
                    flat_indices = y * image_size + x
                    grid.flat += np.bincount(flat_indices, weights=f * w, minlength=grid.size)

                paint_flux(bg_grid, x0_bg, y0_bg, w00, f_bg)
                paint_flux(bg_grid, x0_bg + 1, y0_bg, w10, f_bg)
                paint_flux(bg_grid, x0_bg, y0_bg + 1, w01, f_bg)
                paint_flux(bg_grid, x0_bg + 1, y0_bg + 1, w11, f_bg)
                
                bg_image_obj = galsim.InterpolatedImage(galsim.ImageF(bg_grid, scale=random_pixel_scale), x_interpolant='linear')
                convolved_bg_obj = galsim.Convolve([bg_image_obj, fast_psf])
                convolved_bg_img = galsim.ImageF(image_size, image_size, scale=random_pixel_scale)
                convolved_bg_obj.drawImage(image=convolved_bg_img, method='auto')
                convolved_bg_array = convolved_bg_img.array
                full_image_obj.array[:] += convolved_bg_array

            full_image = full_image_obj.array + total_sky_map + additive_dust_emission
            
            # Background Truth Calculation
            truth_bg_map = convolved_bg_array + total_sky_map + additive_dust_emission
            bg_downsampled = truth_bg_map.reshape(grid_size, cell_size, grid_size, cell_size).mean(axis=(1, 3))
            
            # Generate Target Grid
            N_eff = 12.0 
            local_bg = map_coordinates(total_sky_map + additive_dust_emission, [np.clip(py, 0, image_size-1), np.clip(px, 0, image_size-1)], order=1, mode='nearest')
            noise_variance = apparent_fluxes + N_eff * (local_bg + read_noise**2)
            snrs = apparent_fluxes / np.sqrt(np.maximum(1e-9, noise_variance))
            
            in_fov = (px >= 0) & (px < image_size) & (py >= 0) & (py < image_size)
            sort_idx = np.argsort(-apparent_fluxes[in_fov])
            
            target_grid = fast_paint_grid(
                px[in_fov], py[in_fov], apparent_fluxes[in_fov], snrs[in_fov],
                sort_idx, d_cfg['min_snr'], grid_size, cell_size, K
            )
            target_grid_flat = np.concatenate([target_grid.reshape(grid_size, grid_size, -1), bg_downsampled[:, :, None]], axis=-1)
            
            # Step 2D: Track Metadata (17 Elements)
            g1, g2 = psf_params['g1'], psf_params['g2']
            shear_mag = np.sqrt(g1**2 + g2**2)
            astig_mag = np.sqrt(psf_params['astig1']**2 + psf_params['astig2']**2)
            coma_mag = np.sqrt(psf_params['coma1']**2 + psf_params['coma2']**2)
            
            meta = np.array([
                exp_time, zp, sky_mag,           # [0, 1, 2] Photometry
                psf_params['fwhm'], shear_mag,   # [3, 4] Base Optics
                psf_params['obscuration'], float(psf_params['num_struts']),  # [5, 6] Structure
                random_pixel_scale,              # [7] Crowding/Scale
                psf_params['defocus'], astig_mag, coma_mag,    # [8, 9, 10] Zernike Aberrations
                psf_params['jitter_sigma'], read_noise,        # [11, 12] Tracking & Electronics
                max_extinction,                  # [13] Dust Extinction
                gamma, rc_loc, rc_enhancement    # [14, 15, 16] LF / Astrophysics
            ], dtype=np.float32)
            
            # Step 2E: Direct HDF5 Write
            dset_img[i] = full_image.astype(np.float32)
            dset_tgt[i] = target_grid_flat.astype(np.float32)
            dset_med[i] = np.median(full_image).astype(np.float32)
            dset_meta[i] = meta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Stage 1 GalSim Data")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--num_samples", type=int, default=None, help="Override total number of samples to generate")
    args = parser.parse_args()
    
    run_stage1_generation(config_path=args.config, num_samples=args.num_samples)
