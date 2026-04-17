import numpy as np
import torch
from torch.utils.data import Dataset
import os
import h5py
from scipy.signal import fftconvolve
from scipy.ndimage import map_coordinates
from castor.data.transforms import AstroSpaceTransform
from castor.constants import DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL, GLOBAL_STRETCH_SCALE, SHAPE_SIZE

# Ensure STPSF_PATH is set for stpsf tool
if 'STPSF_PATH' not in os.environ:
    # Default location where stpsf downloads data if not found
    default_path = os.path.expanduser('~/data/stpsf-data')
    if os.path.exists(default_path):
        os.environ['STPSF_PATH'] = default_path

def generate_stpsf_roman_psf(grid_size=129, oversample=4):
    """
    Generates a realistic Roman PSF on the fly using the stpsf (Space Telescope PSF) tool.
    Randomizes the detector (SCA) and position for variety.
    """
    try:
        import stpsf
    except ImportError:
        # Fallback to a simplified analytical model if stpsf is not available
        return _generate_fallback_psf(grid_size, oversample)

    inst = stpsf.WFI()
    
    # Randomize Filter (Common Roman NIR filters)
    filters = ['F062', 'F087', 'F106', 'F129', 'F146', 'F158', 'F184', 'F213']
    inst.filter = np.random.choice(filters)
    
    # Randomize Detector (WFI01 to WFI18)
    inst.detector = np.random.choice(inst.detector_list)
    
    # Randomize Position on the 4096x4096detector
    inst.detector_position = (np.random.uniform(4, 4092), np.random.uniform(4, 4092))
    
    # Calculate PSF
    # fov_pixels=grid_size means the FOV will match grid_size detector pixels
    psf_hdu = inst.calc_psf(fov_pixels=grid_size, oversample=oversample)
    psf_data = psf_hdu[0].data.astype(np.float32)
    
    # Ensure exact expected shape (grid_size * oversample)
    expected_s = grid_size * oversample
    if psf_data.shape[0] != expected_s:
        # Crop or pad to match
        s = psf_data.shape[0]
        if s > expected_s:
            start = (s - expected_s) // 2
            psf_data = psf_data[start:start+expected_s, start:start+expected_s]
        else:
            pad = (expected_s - s) // 2
            psf_data = np.pad(psf_data, ((pad, expected_s-s-pad), (pad, expected_s-s-pad)))

    return psf_data / (psf_data.sum() + 1e-9)

def _generate_fallback_psf(grid_size, oversample):
    """Simplified Roman-like PSF with core and diffraction spikes."""
    S = grid_size * oversample
    center = (S - 1) / 2.0
    y, x = np.meshgrid(np.arange(S) - center, np.arange(S) - center, indexing='ij')
    
    sigma = np.random.uniform(1.6, 2.0)
    q = np.random.uniform(0.9, 1.0)
    theta = np.random.uniform(0, np.pi)
    cos, sin = np.cos(theta), np.sin(theta)
    xp, yp = x * cos + y * sin, -x * sin + y * cos
    psf = np.exp(-(xp**2 / (2 * sigma**2) + yp**2 / (2 * (sigma * q)**2)))
    
    spike_angles = [theta + i * np.pi / 3 for i in range(3)]
    for angle in spike_angles:
        dist_to_line = np.abs(x * np.sin(angle) - y * np.cos(angle))
        psf += np.exp(-dist_to_line / 0.8) * 0.05
        
    psf = np.maximum(0, psf)
    return psf / (psf.sum() + 1e-9)

def fast_paint_grid(lx, ly, fluxes, snrs, sort_idx, min_snr, grid_size, cell_size, K):
    """Highly optimized target grid painter."""
    grid_stars = np.zeros((grid_size, grid_size, K, 4), dtype=np.float32)
    counts = np.zeros((grid_size, grid_size), dtype=np.int32)
    
    for idx in range(len(sort_idx)):
        i = sort_idx[idx]
        snr = snrs[i]
        
        if snr <= 1.0:
            target_p = 0.0
        elif snr >= 5.0:
            target_p = 1.0
        else:
            target_p = np.log10(snr) / 0.69897000433

        if target_p <= 0.0:
            continue
            
        cx = int(lx[i] // cell_size)
        cy = int(ly[i] // cell_size)
        
        if 0 <= cx < grid_size and 0 <= cy < grid_size:
            slot = counts[cy, cx]
            if slot < K:
                grid_stars[cy, cx, slot, 0] = target_p
                grid_stars[cy, cx, slot, 1] = lx[i] % cell_size
                grid_stars[cy, cx, slot, 2] = ly[i] % cell_size
                grid_stars[cy, cx, slot, 3] = fluxes[i]
                counts[cy, cx] += 1
                
    return grid_stars

def sample_bulge_magnitudes(n_total, rc_mag, rc_sigma, rc_enhancement=3.0, m_min=12.0, m_max=32.0, gamma=0.3):
    """Samples from a power-law luminosity function with Red Clump enhancement."""
    u = np.random.uniform(0, 1, n_total)
    a = 10**(gamma * m_min)
    b = 10**(gamma * m_max)
    m_base = (1.0 / gamma) * np.log10(u * (b - a) + a)
    
    local_rgb_count = np.sum((m_base >= rc_mag - 0.5) & (m_base <= rc_mag + 0.5))
    n_rc = int(local_rgb_count * rc_enhancement)
    
    if n_rc > 0:
        m_rc = np.random.normal(loc=rc_mag, scale=rc_sigma, size=n_rc)
        m_all = np.concatenate([m_base, m_rc])
    else:
        m_all = m_base
        
    m_all = np.clip(m_all, m_min, m_max)
    np.random.shuffle(m_all)
    return m_all

def calculate_safe_magnitude_cutoff(exp_time, zp, sky_mag, read_noise=5.0, sigma=1.5, snr_cutoff=1.0):
    pixel_scale = 0.11
    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (pixel_scale**2) * exp_time
    n_pix = 12.0 
    bg_variance = n_pix * (sky_level + read_noise**2)
    a, b, c = 1.0, -(snr_cutoff**2), -(snr_cutoff**2) * bg_variance
    min_flux = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    return zp - 2.5 * np.log10(min_flux / exp_time)

def generate_dust_cirrus(img_size, amplitude):
    """Generates fractal dust noise with P(k) ~ k^-3 power spectrum."""
    fx = np.fft.fftfreq(img_size)
    fy = np.fft.fftfreq(img_size)
    kx, ky = np.meshgrid(fx, fy)
    k = np.sqrt(kx**2 + ky**2)
    k[0, 0] = 1e-9 
    
    noise_fft = (np.random.normal(size=(img_size, img_size)) + 
                 1j * np.random.normal(size=(img_size, img_size)))
    
    noise_fft *= k**(-1.5)
    noise_fft[0, 0] = 0.0 
    
    dust_map = np.real(np.fft.ifft2(noise_fft))
    dust_map -= dust_map.min()
    dust_map /= (dust_map.max() + 1e-9)
    return dust_map * amplitude

def generate_mosaic_data(mosaic_size, params):
    """
    Simplified Stage 0 Engine: Uses a random Roman PSF generated on the fly.
    Employs bi-linear interpolation for sub-pixel placement.
    """
    area_ratio = (mosaic_size / params['image_size'])**2
    exp_time, zp, sky_mag = np.random.uniform(30.0, 90.0), 26.5, 22.0
    mag_limit = calculate_safe_magnitude_cutoff(exp_time, zp, sky_mag, snr_cutoff=1.0)
    
    n_stars_total = int(10 ** np.random.uniform(np.log10(params['min_stars'] * area_ratio), np.log10(params['max_stars'] * area_ratio)))
    mags = sample_bulge_magnitudes(n_stars_total, np.random.uniform(14.5, 16.5), np.random.uniform(0.2, 0.5), np.random.uniform(5.0, 15.0), m_min=12.0, m_max=32.0, gamma=np.random.uniform(0.25, 0.35))
    fluxes = exp_time * (10 ** (-0.4 * (mags - zp)))
    
    O = 4 
    
    # 1. Generate a random Roman PSF on the fly using Space Telescope tools (stpsf)
    repr_psf_4x = generate_stpsf_roman_psf(SHAPE_SIZE, oversample=O)
    psf_1x = repr_psf_4x.reshape(SHAPE_SIZE, O, SHAPE_SIZE, O).mean(axis=(1, 3))
    psf_1x /= (np.sum(psf_1x) + 1e-9)

    # 2. Placement
    px, py = np.random.uniform(0.5, mosaic_size-1.5, len(fluxes)), np.random.uniform(0.5, mosaic_size-1.5, len(fluxes))
    
    # 3. Dust Extinction
    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (0.11**2) * exp_time
    if np.random.rand() < 0.7:
        print("🌫️ Adding realistic Interstellar Cirrus (Dust) to mosaic...")
        raw_dust = generate_dust_cirrus(mosaic_size, 1.0)
        max_extinction = np.random.uniform(0.2, 5.0)
        transmission_map = 10 ** (-0.4 * raw_dust * max_extinction)
        
        star_transmissions = map_coordinates(transmission_map, [py, px], order=1, mode='nearest')
        apparent_fluxes = fluxes * star_transmissions
        apparent_mags = mags - 2.5 * np.log10(star_transmissions + 1e-9)
        
        frac_bg = 0.60
        sky_foreground = sky_level * (1.0 - frac_bg)
        sky_background_attenuated = (sky_level * frac_bg) * transmission_map
        total_sky_map = sky_foreground + sky_background_attenuated
        additive_dust_emission = raw_dust * np.random.uniform(20, 100)
    else:
        transmission_map = np.ones((mosaic_size, mosaic_size), dtype=np.float32)
        apparent_fluxes = fluxes.copy()
        apparent_mags = mags.copy()
        total_sky_map = np.full((mosaic_size, mosaic_size), sky_level, dtype=np.float32)
        additive_dust_emission = np.zeros((mosaic_size, mosaic_size), dtype=np.float32)
    
    v_mask = apparent_mags < mag_limit
    
    fg_grid = np.zeros((mosaic_size, mosaic_size), dtype=np.float32)
    bg_grid = np.zeros((mosaic_size, mosaic_size), dtype=np.float32)
    
    x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
    dx, dy = px - x0, py - y0
    w00, w10, w01, w11 = (1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy
    
    def paint_flux(grid, x, y, w, f):
        flat_indices = y * mosaic_size + x
        grid.flat += np.bincount(flat_indices, weights=f * w, minlength=grid.size)

    paint_flux(fg_grid, x0[v_mask], y0[v_mask], w00[v_mask], apparent_fluxes[v_mask])
    paint_flux(fg_grid, x0[v_mask] + 1, y0[v_mask], w10[v_mask], apparent_fluxes[v_mask])
    paint_flux(fg_grid, x0[v_mask], y0[v_mask] + 1, w01[v_mask], apparent_fluxes[v_mask])
    paint_flux(fg_grid, x0[v_mask] + 1, y0[v_mask] + 1, w11[v_mask], apparent_fluxes[v_mask])

    paint_flux(bg_grid, x0[~v_mask], y0[~v_mask], w00[~v_mask], apparent_fluxes[~v_mask])
    paint_flux(bg_grid, x0[~v_mask] + 1, y0[~v_mask], w10[~v_mask], apparent_fluxes[~v_mask])
    paint_flux(bg_grid, x0[~v_mask], y0[~v_mask] + 1, w01[~v_mask], apparent_fluxes[~v_mask])
    paint_flux(bg_grid, x0[~v_mask] + 1, y0[~v_mask] + 1, w11[~v_mask], apparent_fluxes[~v_mask])

    fg_image = fftconvolve(fg_grid, psf_1x, mode='same')
    bg_image = fftconvolve(bg_grid, psf_1x, mode='same')

    # 6. Final Image Composition
    full_image = np.maximum(0, fg_image + bg_image + total_sky_map + additive_dust_emission)
    
    # 7. SNR and Truth
    half = SHAPE_SIZE // 2
    N_eff = 1.0 / (np.sum(psf_1x**2) + 1e-9)
    star_peak_val = psf_1x[half, half]
    
    local_sky = map_coordinates(total_sky_map + additive_dust_emission, [py[v_mask], px[v_mask]], order=1, mode='nearest')
    actual_pixel_values = full_image[np.clip(y0[v_mask], 0, mosaic_size-1), np.clip(x0[v_mask], 0, mosaic_size-1)]
    confusion_light = np.maximum(0.0, actual_pixel_values - (apparent_fluxes[v_mask] * star_peak_val))
    noise_variance = apparent_fluxes[v_mask] + N_eff * (local_sky + confusion_light + 25.0)
    snrs = apparent_fluxes[v_mask] / np.sqrt(noise_variance)
    
    cat_dtype = [('x', 'f4'), ('y', 'f4'), ('flux', 'f4'), ('mag', 'f4'), ('snr', 'f4')]
    sort_y = np.argsort(py[v_mask])
    structured_cat = np.zeros(np.sum(v_mask), dtype=cat_dtype)
    structured_cat['x'] = px[v_mask][sort_y]
    structured_cat['y'] = py[v_mask][sort_y]
    structured_cat['flux'] = apparent_fluxes[v_mask][sort_y]
    structured_cat['mag'] = apparent_mags[v_mask][sort_y]
    structured_cat['snr'] = snrs[sort_y]
    
    meta = np.array([exp_time, zp, sky_mag, 0.0, 0.0, 0.0])
    truth_bg_map = bg_image + total_sky_map + additive_dust_emission

    return full_image, truth_bg_map, structured_cat, meta, psf_1x

class HDF5MosaicDataset(Dataset):
    def __init__(self, h5_path, image_size=256):
        self.h5_path, self.file, self.img_size = h5_path, None, image_size
        self.cell_size, self.grid_size, self.K = DEFAULT_CELL_SIZE, image_size // DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL
        self.transform = AstroSpaceTransform(stretch_scale=GLOBAL_STRETCH_SCALE)
        if not os.path.exists(self.h5_path): raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")
        
        with h5py.File(self.h5_path, 'r') as f:
            self.length = len(f['images'])

    def __len__(self): return self.length

    def __getitem__(self, idx):
        if self.file is None: self.file = h5py.File(self.h5_path, 'r', swmr=True, libver='latest')
        img = torch.from_numpy(self.file['images'][idx]).float()
        target = torch.from_numpy(self.file['targets'][idx]).float()
        median = float(self.file['chunk_medians'][idx])
        meta = self.file['metas'][idx] if 'metas' in self.file else np.zeros(6, dtype=np.float32)
        # Return a single mean PSF or None if we don't need it per-sample here
        return {"image": img, "target": target, "chunk_median": median, "meta": meta}
