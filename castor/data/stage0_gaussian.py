import numpy as np
import torch
from torch.utils.data import Dataset
import os
import h5py
from scipy.signal import fftconvolve
from castor.data.transforms import AstroSpaceTransform
from castor.constants import DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL, GLOBAL_STRETCH_SCALE, SHAPE_SIZE, N_PCA_COMPONENTS

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

def generate_mosaic_data(mosaic_size, params, master_psf_data):
    """
    The Core Stage 0 Rendering Engine.
    Generates a full mosaic using Area Integration and Representative PSFs.
    """
    area_ratio = (mosaic_size / params['image_size'])**2
    exp_time, zp, sky_mag = np.random.uniform(30.0, 60.0), 26.5, 22.0
    mag_limit = calculate_safe_magnitude_cutoff(exp_time, zp, sky_mag, snr_cutoff=1.0)
    
    n_stars_total = int(10 ** np.random.uniform(np.log10(params['min_stars'] * area_ratio), np.log10(params['max_stars'] * area_ratio)))
    mags = sample_bulge_magnitudes(n_stars_total, np.random.uniform(14.5, 16.5), np.random.uniform(0.2, 0.5), np.random.uniform(5.0, 15.0), m_min=12.0, m_max=32.0, gamma=np.random.uniform(0.25, 0.35))
    fluxes = exp_time * (10 ** (-0.4 * (mags - zp)))
    
    eigen_psfs, psf_weights_lib, mean_psf = master_psf_data
    O = 4 # Oversampling
    
    # Representative PSF Selection
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

    repr_psf_jit_4x = fftconvolve(repr_psf_4x, jitter_kernel_high, mode='same')

    # Proper Binning (Area Integration)
    psf_library = np.zeros((O, O, SHAPE_SIZE, SHAPE_SIZE), dtype=np.float32)
    padded_psf = np.pad(repr_psf_jit_4x, ((0, O), (0, O)))
    for dy_idx in range(O):
        for dx_idx in range(O):
            window = padded_psf[dy_idx : dy_idx + SHAPE_SIZE*O, dx_idx : dx_idx + SHAPE_SIZE*O]
            binned = window.reshape(SHAPE_SIZE, O, SHAPE_SIZE, O).mean(axis=(1, 3))
            psf_library[dy_idx, dx_idx] = binned / (np.sum(binned) + 1e-9)

    px, py = np.random.uniform(0, mosaic_size, len(fluxes)), np.random.uniform(0, mosaic_size, len(fluxes))
    full_image = np.zeros((mosaic_size, mosaic_size), dtype=np.float32)
    x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
    dx_idx = np.clip(np.floor((px - x0) * O).astype(int), 0, O-1)
    dy_idx = np.clip(np.floor((py - y0) * O).astype(int), 0, O-1)
    valid = (x0 >= 0) & (x0 < mosaic_size) & (y0 >= 0) & (y0 < mosaic_size)
    
    for dyi in range(O):
        for dxi in range(O):
            mask = valid & (dx_idx == dxi) & (dy_idx == dyi)
            if not mask.any(): continue
            flat_indices = y0[mask] * mosaic_size + x0[mask]
            grid = np.bincount(flat_indices, weights=fluxes[mask], minlength=mosaic_size*mosaic_size).reshape(mosaic_size, mosaic_size)
            full_image += fftconvolve(grid, psf_library[dyi, dxi], mode='same')

    full_image = np.maximum(0, full_image)
    v_mask = mags < mag_limit
    
    # Rigorous SNR
    half = SHAPE_SIZE // 2
    centered_psf = psf_library[O//2, O//2]
    N_eff = 1.0 / (np.sum(centered_psf**2) + 1e-9)
    actual_pixel_values = full_image[y0[v_mask], x0[v_mask]]
    star_peaks = psf_library[dy_idx[v_mask], dx_idx[v_mask], half, half]
    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (0.11**2) * exp_time
    confusion_light = np.maximum(0.0, actual_pixel_values - (fluxes[v_mask] * star_peaks))
    noise_variance = fluxes[v_mask] + N_eff * (sky_level + confusion_light + 25.0)
    snrs = fluxes[v_mask] / np.sqrt(noise_variance)
    
    cat_dtype = [('x', 'f4'), ('y', 'f4'), ('flux', 'f4'), ('mag', 'f4'), ('snr', 'f4')] + [(f'w{i}', 'f2') for i in range(N_PCA_COMPONENTS)]
    structured_cat = np.zeros(np.sum(v_mask), dtype=cat_dtype)
    structured_cat['x'], structured_cat['y'], structured_cat['flux'], structured_cat['mag'] = px[v_mask], py[v_mask], fluxes[v_mask], mags[v_mask]
    structured_cat['snr'] = snrs
    for i in range(N_PCA_COMPONENTS): structured_cat[f'w{i}'] = repr_weights[i]
    
    # Save the 1x version for visualizer compatibility
    repr_psf_1x = repr_psf_jit_4x.reshape(SHAPE_SIZE, O, SHAPE_SIZE, O).mean(axis=(1, 3))
    norm_val = np.max(repr_psf_1x)
    repr_psf_1x /= (norm_val + 1e-9)
    psf_lib_save = np.zeros((N_PCA_COMPONENTS + 1, SHAPE_SIZE * SHAPE_SIZE), dtype=np.float32)
    psf_lib_save[-1] = repr_psf_1x.flatten()

    meta = np.array([exp_time, zp, sky_mag, s_jit, q_jit, theta_jit])
    
    return full_image, structured_cat, meta, psf_lib_save

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
