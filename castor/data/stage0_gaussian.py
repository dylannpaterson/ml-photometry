import numpy as np
import torch
from torch.utils.data import Dataset
import os
import h5py
from scipy.signal import fftconvolve
from scipy.ndimage import map_coordinates
from castor.data.transforms import AstroSpaceTransform
from castor.constants import DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL, GLOBAL_STRETCH_SCALE, SHAPE_SIZE, N_PCA_COMPONENTS

def generate_field_realistic_psf_library(num_psfs=100, grid_size=127, oversample=4):
    """Generates a master optical library with oversampling."""
    print(f"📡 Generating Master OPTICAL PSF Library ({num_psfs} PSFs, {oversample}x oversampled)...")
    S = grid_size * oversample
    library = np.zeros((num_psfs, S, S), dtype=np.float32)
    center = (S - 1) / 2.0
    optical_template = None
    if os.path.exists("roman_psf_prior_4x.pt"):
        try:
            optical_template = torch.load("roman_psf_prior_4x.pt", map_location='cpu', weights_only=False).numpy()
            if optical_template.shape[0] != S:
                from scipy.ndimage import zoom
                scale = S / optical_template.shape[0]
                optical_template = zoom(optical_template, scale, order=3)
        except Exception as e: print(f"⚠️ Oversampled PSF Load Failed: {e}")

    y, x = np.meshgrid(np.arange(S) - center, np.arange(S) - center, indexing='ij')
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
    n_pix = 12.0 # Matched filter area for Roman
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

def generate_mosaic_data(mosaic_size, params, master_psf_library):
    """
    Simplified Stage 0 Engine: Uses a single physical PSF from the library.
    Employs bi-linear interpolation for sub-pixel placement.
    Includes realistic Dust Extinction (Multiplicative) and Extended Sources.
    """
    area_ratio = (mosaic_size / params['image_size'])**2
    exp_time, zp, sky_mag = np.random.uniform(30.0, 90.0), 26.5, 22.0
    mag_limit = calculate_safe_magnitude_cutoff(exp_time, zp, sky_mag, snr_cutoff=1.0)
    
    n_stars_total = int(10 ** np.random.uniform(np.log10(params['min_stars'] * area_ratio), np.log10(params['max_stars'] * area_ratio)))
    mags = sample_bulge_magnitudes(n_stars_total, np.random.uniform(14.5, 16.5), np.random.uniform(0.2, 0.5), np.random.uniform(5.0, 15.0), m_min=12.0, m_max=32.0, gamma=np.random.uniform(0.25, 0.35))
    fluxes = exp_time * (10 ** (-0.4 * (mags - zp)))
    
    O = 4 # Oversampling
    
    # 1. Select a single Physical PSF from the library
    repr_idx = np.random.randint(0, len(master_psf_library))
    repr_psf_4x = master_psf_library[repr_idx] 
    psf_1x = repr_psf_4x.reshape(SHAPE_SIZE, O, SHAPE_SIZE, O).mean(axis=(1, 3))
    psf_1x /= (np.sum(psf_1x) + 1e-9)

    # 2. Placement
    px, py = np.random.uniform(0.5, mosaic_size-1.5, len(fluxes)), np.random.uniform(0.5, mosaic_size-1.5, len(fluxes))
    
    # 3. NEW: Multiplicative Dust Extinction Logic
    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (0.11**2) * exp_time
    if np.random.rand() < 0.7:
        print("🌫️ Adding realistic Interstellar Cirrus (Dust) to mosaic...")
        raw_dust = generate_dust_cirrus(mosaic_size, 1.0)

        # Realistic Galactic Bulge Extinction: Up to 5.0 mags in Roman NIR filters
        max_extinction = np.random.uniform(0.2, 5.0)
        transmission_map = 10 ** (-0.4 * raw_dust * max_extinction)
        
        star_transmissions = map_coordinates(transmission_map, [py, px], order=1, mode='nearest')
        apparent_fluxes = fluxes * star_transmissions
        apparent_mags = mags - 2.5 * np.log10(star_transmissions + 1e-9)
        
        # Split Sky Background: Only the background component is extincted
        frac_bg = 0.60
        sky_foreground = sky_level * (1.0 - frac_bg)
        sky_background_attenuated = (sky_level * frac_bg) * transmission_map
        total_sky_map = sky_foreground + sky_background_attenuated

        # Additive Emission: Simulates scattering in dense dust (P(k)~k^-3)
        # Scaled up for higher extinction regions
        additive_dust_emission = raw_dust * np.random.uniform(20, 100)
    else:
        transmission_map = np.ones((mosaic_size, mosaic_size), dtype=np.float32)
        apparent_fluxes = fluxes.copy()
        apparent_mags = mags.copy()
        total_sky_map = np.full((mosaic_size, mosaic_size), sky_level, dtype=np.float32)
        additive_dust_emission = np.zeros((mosaic_size, mosaic_size), dtype=np.float32)
    
    # 4. Filter resolved stars using apparent magnitude
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

    # 5. Physics: Convolutions
    fg_image = fftconvolve(fg_grid, psf_1x, mode='same')
    bg_image = fftconvolve(bg_grid, psf_1x, mode='same')

    # 6. Add Extended Sources (Extincted)
    num_extended = np.random.randint(0, 4)
    ext_image = np.zeros((mosaic_size, mosaic_size), dtype=np.float32)
    if num_extended > 0:
        y_idx, x_idx = np.meshgrid(np.arange(mosaic_size), np.arange(mosaic_size), indexing='ij')
        for _ in range(num_extended):
            ex, ey = np.random.uniform(0, mosaic_size), np.random.uniform(0, mosaic_size)
            eflux = np.random.uniform(5000, 30000)
            esigma = np.random.uniform(3.0, 10.0)
            eq = np.random.uniform(0.3, 0.8)
            etheta = np.random.uniform(0, np.pi)
            cos_t, sin_t = np.cos(etheta), np.sin(etheta)
            xp = (x_idx - ex) * cos_t + (y_idx - ey) * sin_t
            yp = -(x_idx - ex) * sin_t + (y_idx - ey) * cos_t
            ext_blob = np.exp(-(xp**2 / (2 * esigma**2) + yp**2 / (2 * (esigma * eq)**2)))
            ext_image += (eflux / (ext_blob.sum() + 1e-9)) * ext_blob
        ext_image *= transmission_map

    # 7. Final Image Composition
    full_image = np.maximum(0, fg_image + bg_image + ext_image + total_sky_map + additive_dust_emission)
    
    # 8. Rigorous SNR using local sky values
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
    
    psf_lib_save = np.zeros((N_PCA_COMPONENTS + 1, SHAPE_SIZE * SHAPE_SIZE), dtype=np.float32)
    psf_lib_save[-1] = psf_1x.flatten()
    meta = np.array([exp_time, zp, sky_mag, 0.0, 0.0, 0.0])

    # For the background truth, we want everything EXCEPT the resolved stars
    truth_bg_map = bg_image + ext_image + total_sky_map + additive_dust_emission

    return full_image, truth_bg_map, structured_cat, meta, psf_lib_save

class HDF5MosaicDataset(Dataset):
    def __init__(self, h5_path, image_size=256):
        self.h5_path, self.file, self.img_size = h5_path, None, image_size
        self.cell_size, self.grid_size, self.K = DEFAULT_CELL_SIZE, image_size // DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL
        self.transform = AstroSpaceTransform(stretch_scale=GLOBAL_STRETCH_SCALE)
        if not os.path.exists(self.h5_path): raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")
        
        with h5py.File(self.h5_path, 'r') as f:
            self.length = len(f['images'])
            self.psf_library = torch.from_numpy(f['psf_libraries'][0]).float() if 'psf_libraries' in f else None

    def __len__(self): return self.length

    def __getitem__(self, idx):
        if self.file is None: self.file = h5py.File(self.h5_path, 'r', swmr=True, libver='latest')
        img = torch.from_numpy(self.file['images'][idx]).float()
        target = torch.from_numpy(self.file['targets'][idx]).float()
        median = float(self.file['chunk_medians'][idx])
        meta = self.file['metas'][idx] if 'metas' in self.file else np.zeros(6, dtype=np.float32)
        return {"image": img, "target": target, "psf_library": self.psf_library, "chunk_median": median, "meta": meta}
