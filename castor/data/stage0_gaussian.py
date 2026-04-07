import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
import pandas as pd
import h5py
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve
from castor.data.transforms import AstroSpaceTransform
from castor.constants import DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL, SHAPE_SIZE, GLOBAL_STRETCH_SCALE, N_PCA_COMPONENTS
from numba import njit
import gc

#@njit(boundscheck=False)
def fast_paint_grid(lx, ly, fluxes, snrs, sort_idx, min_snr, grid_size, cell_size, K):
    # CHANGED: Now only 4 channels for targets [p, dx, dy, flux]
    # The network predicts log_vars as latent variables, we don't need targets for them.
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

class GaussianPretrainingProvider(Dataset):
    def __init__(self, num_samples=1000, min_stars=1000000, max_stars=8000000, image_size=256, 
                 max_capacity_per_cell=MAX_CAPACITY_PER_CELL, shape_size=SHAPE_SIZE, 
                 use_fixed_seed=False, global_stretch_scale=GLOBAL_STRETCH_SCALE, min_snr=5.0,
                 psf_library_path=None):
        self.num_samples = num_samples
        self.min_stars = min_stars
        self.max_stars = max_stars
        self.img_size = image_size
        self.K = max_capacity_per_cell
        self.S = shape_size
        self.read_noise = 5.0
        self.use_fixed_seed = use_fixed_seed
        self.min_snr = min_snr
        self.transform = AstroSpaceTransform(stretch_scale=global_stretch_scale)
        self.cell_size = DEFAULT_CELL_SIZE
        self.grid_size = self.img_size // self.cell_size
        self.n_pca = N_PCA_COMPONENTS
        
        if psf_library_path and os.path.exists(psf_library_path):
            print(f"📂 GaussianPretrainingProvider: Loading Master PSF Library from {psf_library_path}")
            master_data = torch.load(psf_library_path, map_location='cpu', weights_only=False)
            if isinstance(master_data, dict):
                self.eigen_psfs = master_data['eigen_psfs']
                self.psf_weights_lib = master_data['weights_lib']
                self.mean_psf = master_data['mean_psf']
            else:
                self.eigen_psfs, self.psf_weights_lib, self.mean_psf = master_data
        else:
            # 1. Generate pristine optical-only library
            raw_library = self._generate_optical_library(100, self.S)
            # 2. Extract Eigen-PSFs
            self.eigen_psfs, self.psf_weights_lib, self.mean_psf = self._compute_eigen_psfs(raw_library, n_components=self.n_pca)
        
        # PCA Weights are no longer used for targets, but we keep them for rendering
        self.psf_library_tensor = torch.cat([
            torch.from_numpy(self.eigen_psfs).view(self.n_pca, -1),
            torch.from_numpy(self.mean_psf).view(1, -1)
        ], dim=0)

    def _generate_optical_library(self, num_psfs, grid_size):
        library = np.zeros((num_psfs, grid_size, grid_size), dtype=np.float32)
        half = grid_size // 2
        optical_template = None
        if os.path.exists("roman_psf_prior.pt"):
            try:
                optical_template = torch.load("roman_psf_prior.pt", map_location='cpu', weights_only=True).numpy()
            except Exception: pass

        y, x = np.meshgrid(np.arange(grid_size) - half, np.arange(grid_size) - half, indexing='ij')
        for i in range(num_psfs):
            fx, fy = np.random.uniform(-2048, 2048), np.random.uniform(-2048, 2048)
            r_norm = np.sqrt(fx**2 + fy**2) / 2896.0
            q_opt = np.random.uniform(0.9, 1.0) - (0.1 * r_norm)
            theta = np.arctan2(fy, fx) + np.random.normal(0, 0.1)
            cos, sin = np.cos(theta), np.sin(theta)
            xp, yp = x * cos + y * sin, -x * sin + y * cos
            s_opt = 0.45
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

    def _compute_eigen_psfs(self, large_library, n_components=10):
        N, H, W = large_library.shape
        data = torch.from_numpy(large_library).float().view(N, H * W)
        mean_psf = data.mean(dim=0)
        centered_data = data - mean_psf
        U, S, V = torch.pca_lowrank(centered_data, q=n_components)
        eigen_psfs = V.t().view(n_components, H, W).numpy()
        psf_weights = (U * S).numpy() 
        return eigen_psfs, psf_weights, mean_psf.view(H, W).numpy()

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        if self.use_fixed_seed: np.random.seed(idx)
        sample = self.generate_chunk()
        return {"image": sample["image"], "target": sample["target"], "chunk_median": sample["chunk_median"], "psf_library": sample["psf_library"]}

    def generate_chunk(self, rc_params=None, exp_params=None):
        if rc_params is None:
            rc_loc, rc_scale, rc_enhancement, lf_gamma = np.random.uniform(14.5, 16.5), np.random.uniform(0.2, 0.5), np.random.uniform(5.0, 15.0), np.random.uniform(0.25, 0.35)
        else: rc_loc, rc_scale, rc_enhancement, lf_gamma = rc_params

        if exp_params is None:
            exp_time, zp, sky_mag = np.random.uniform(30.0, 60.0), 26.5, 22.0
        else: exp_time, zp, sky_mag = exp_params

        pixel_scale, sky_level = 0.11, (10 ** (-0.4 * (sky_mag - zp))) * (0.11**2) * exp_time
        # Adjust star counts for image size
        area_ratio = (self.img_size / 4096.0)**2
        n_stars_base = int(np.random.uniform(self.min_stars, self.max_stars) * area_ratio)
        if n_stars_base < 100: n_stars_base = 10000 # Fallback for small chunks
        
        mags = sample_bulge_magnitudes(n_stars_base, rc_loc, rc_scale, rc_enhancement, m_min=12.0, m_max=32.0, gamma=lf_gamma)
        fluxes = exp_time * (10 ** (-0.4 * (mags - zp)))
        sort_idx = np.argsort(fluxes)[::-1]
        fluxes, mags = fluxes[sort_idx], mags[sort_idx]
        x_centers, y_centers = np.random.uniform(0, self.img_size, len(mags)), np.random.uniform(0, self.img_size, len(mags))
        
        # 4. Apply Jitter Before Pixelation
        s_jit, q_jit, theta_jit = np.random.normal(0.127, 0.01), np.random.uniform(0.8, 1.0), np.random.uniform(0, np.pi)
        O = 4 # Oversampling
        S_jit_high = self.S * O
        k_half_high = S_jit_high // 2
        gy, gx = np.meshgrid(np.arange(S_jit_high) - k_half_high, np.arange(S_jit_high) - k_half_high, indexing='ij')
        cos, sin = np.cos(theta_jit), np.sin(theta_jit)
        s_jit_high = s_jit * O
        gxp, gyp = gx * cos + gy * sin, -gx * sin + gy * cos
        jitter_kernel_high = np.exp(-(gxp**2 / (2 * s_jit_high**2) + gyp**2 / (2 * (s_jit_high * q_jit)**2)))
        jitter_kernel_high /= (jitter_kernel_high.sum() + 1e-9)

        # We assume self.mean_psf is at 1x resolution here
        def get_upsampled(img, scale):
            from scipy.ndimage import zoom
            return zoom(img, scale, order=3)

        if self.mean_psf.shape[0] == self.S:
            mean_psf_4x = get_upsampled(self.mean_psf, O)
        else:
            mean_psf_4x = self.mean_psf

        mean_psf_jit_4x = fftconvolve(mean_psf_4x, jitter_kernel_high, mode='same')

        # Pre-compute 16 shifted 1x PSFs for the mean component
        mean_psf_library = np.zeros((O, O, self.S, self.S), dtype=np.float32)
        for dy_idx in range(O):
            for dx_idx in range(O):
                # Energy conservation: normalize shifted phases to sum to 1
                phase = mean_psf_jit_4x[dy_idx::O, dx_idx::O][:self.S, :self.S]
                mean_psf_library[dy_idx, dx_idx] = phase / (np.sum(phase) + 1e-9)

        # Rendering
        x0, y0 = np.floor(x_centers).astype(int), np.floor(y_centers).astype(int)
        dx_idx = np.clip(np.floor((x_centers - x0) * O).astype(int), 0, O-1)
        dy_idx = np.clip(np.floor((y_centers - y0) * O).astype(int), 0, O-1)
        valid = (x0 >= 0) & (x0 < self.img_size) & (y0 >= 0) & (y0 < self.img_size)
        
        # Base Pass using 16 sub-pixel grids (PCA removed for Stage 0 stability)
        star_signal = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        for dyi in range(O):
            for dxi in range(O):
                mask = valid & (dx_idx == dxi) & (dy_idx == dyi)
                if not mask.any(): continue
                flat_indices = y0[mask] * self.img_size + x0[mask]
                grid = np.bincount(flat_indices, weights=fluxes[mask], minlength=self.img_size*self.img_size).reshape(self.img_size, self.img_size)
                star_signal += fftconvolve(grid, mean_psf_library[dyi, dxi], mode='same')

        star_signal = np.maximum(0, star_signal)

        total_photon_flux = star_signal + sky_level
        chunk_median = np.median(star_signal) + sky_level

        # Calculate Rigorous SNR
        centered_psf = mean_psf_library[O//2, O//2]
        N_eff = 1.0 / (np.sum(centered_psf ** 2) + 1e-9)
        
        x0_idx = np.clip(x0, 0, self.img_size - 1)
        y0_idx = np.clip(y0, 0, self.img_size - 1)
        actual_pixel_values = star_signal[y0_idx, x0_idx]
        
        k_half = self.S // 2
        peaks = mean_psf_library[:, :, k_half, k_half]
        star_peaks = peaks[dy_idx, dx_idx]
        
        # 4. Calculate Confusion Noise
        confusion_light = np.maximum(0.0, actual_pixel_values - (fluxes * star_peaks))
        
        # 5. Calculate Final SNR
        noise_variance = fluxes + N_eff * (sky_level + confusion_light + self.read_noise**2)
        snrs = fluxes / np.sqrt(noise_variance)

        # TARGET GENERATION
        base_grid = fast_paint_grid(x_centers, y_centers, fluxes, snrs, sort_idx, self.min_snr, self.grid_size, self.cell_size, self.K)
        target = torch.cat([torch.from_numpy(base_grid).view(self.grid_size, self.grid_size, -1), 
                            torch.from_numpy(np.full((self.grid_size, self.grid_size, 1), self.transform.target_bg_to_network(sky_level - chunk_median), dtype=np.float32))], dim=-1)

        return {"image": torch.from_numpy(total_photon_flux).unsqueeze(0), "target": target, "chunk_median": float(chunk_median), "psf_library": self.psf_library_tensor.unsqueeze(0)}

class GaussianMosaicDataset(Dataset):
    def __init__(self, data_dir, num_samples=25000, image_size=256, cell_size=DEFAULT_CELL_SIZE, global_stretch_scale=GLOBAL_STRETCH_SCALE):
        self.data_dir, self.num_samples, self.img_size, self.cell_size = data_dir, num_samples, image_size, cell_size
        self.grid_size = image_size // cell_size
        self.transform = AstroSpaceTransform(stretch_scale=global_stretch_scale)
        self.K, self.min_snr = MAX_CAPACITY_PER_CELL, 5.0
        self.target_shape = (self.grid_size, self.grid_size, self.K * 4 + 1)
        self.mosaics = []
        image_files = sorted([f for f in os.listdir(data_dir) if f.endswith("_img.npy")])
        for img_f in image_files:
            base = img_f.replace("_img.npy", "")
            cat_f, meta_f, lib_f = base + "_cat.npy", base + "_meta.npy", base + "_psf_lib.npy"
            if os.path.exists(os.path.join(data_dir, cat_f)) and os.path.exists(os.path.join(data_dir, meta_f)):
                meta = np.load(os.path.join(data_dir, meta_f))
                self.mosaics.append({'img_path': os.path.join(data_dir, img_f), 'cat_path': os.path.join(data_dir, cat_f), 'lib_path': os.path.join(data_dir, lib_f) if os.path.exists(os.path.join(data_dir, lib_f)) else None, 'exp_time': meta[0], 'zp': meta[1], 'sky_mag': meta[2]})
        if not self.mosaics: return
        self.active_mosaic_idx = np.random.randint(0, len(self.mosaics))
        self._load_mosaic_to_ram(self.active_mosaic_idx)
        self.max_samples_per_mosaic = 384 
        self.samples_from_current = np.random.randint(0, self.max_samples_per_mosaic)

    def _load_mosaic_to_ram(self, m_idx):
        mosaic = self.mosaics[m_idx]
        self.active_img, self.active_cat, self.active_library, self.active_mosaic_idx, self.samples_from_current = np.load(mosaic['img_path']), np.load(mosaic['cat_path']), np.load(mosaic['lib_path']) if mosaic['lib_path'] else None, m_idx, 0
        gc.collect()

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        if self.samples_from_current >= self.max_samples_per_mosaic: self._load_mosaic_to_ram(np.random.randint(0, len(self.mosaics)))
        self.samples_from_current += 1
        mosaic, my, mx = self.mosaics[self.active_mosaic_idx], *self.active_img.shape
        py, px = np.random.randint(0, my - self.img_size), np.random.randint(0, mx - self.img_size)
        star_signal_np = self.active_img[py:py+self.img_size, px:px+self.img_size]
        sky_level = (10 ** (-0.4 * (mosaic['sky_mag'] - mosaic['zp']))) * (0.11**2) * mosaic['exp_time']
        signal_tensor = torch.from_numpy(star_signal_np).float().add_(sky_level).clamp_(min=0.0).unsqueeze_(0)
        chunk_median = np.median(star_signal_np) + sky_level
        y_start, y_end = np.searchsorted(self.active_cat['y'], py), np.searchsorted(self.active_cat['y'], py + self.img_size)
        band_cat = self.active_cat[y_start:y_end]
        mask_x = (band_cat['x'] >= px) & (band_cat['x'] < px + self.img_size)
        target_buffer = np.zeros(self.target_shape, dtype=np.float32)
        if mask_x.any():
            local_cat = band_cat[mask_x]
            lx, ly, fluxes, snrs = local_cat['x'] - px, local_cat['y'] - py, local_cat['flux'], local_cat['snr']
            grid_stars_np = fast_paint_grid(lx, ly, fluxes, snrs, np.argsort(fluxes)[::-1], self.min_snr, self.grid_size, self.cell_size, self.K)
            target_buffer[:, :, :-1] = grid_stars_np.reshape(self.grid_size, self.grid_size, -1)
        target_buffer[:, :, -1] = self.transform.target_bg_to_network(sky_level - chunk_median)
        return {"image": signal_tensor, "target": torch.from_numpy(target_buffer), "chunk_median": float(chunk_median), "psf_library": torch.from_numpy(self.active_library).unsqueeze(0)}

class HDF5MosaicDataset(Dataset):
    def __init__(self, h5_path, image_size=256):
        self.h5_path, self.file, self.img_size = h5_path, None, image_size
        self.cell_size, self.grid_size, self.K = DEFAULT_CELL_SIZE, image_size // DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL
        if not os.path.exists(self.h5_path): raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")
        
        with h5py.File(self.h5_path, 'r') as f:
            self.length = len(f['images'])
            if 'psf_libraries' in f and len(f['psf_libraries']) > 0:
                self.psf_library = torch.from_numpy(f['psf_libraries'][0]).float()
            else:
                self.psf_library = None

    def __len__(self): return self.length

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r', swmr=True, libver='latest')
            
        img = torch.from_numpy(self.file['images'][idx]).float()
        target = torch.from_numpy(self.file['targets'][idx]).float()
        median = float(self.file['chunk_medians'][idx])
        meta = self.file['metas'][idx] if 'metas' in self.file else np.zeros(6, dtype=np.float32)

        return {
            "image": img,
            "target": target,
            "psf_library": self.psf_library, 
            "chunk_median": median,
            "meta": meta
        }

    def generate_chunk(self):
        import random
        sample = self[random.randint(0, len(self) - 1)]
        return {"image": sample["image"], "base_grid": sample["target"][:, :, :-1].view(self.grid_size, self.grid_size, self.K, -1), "background_map": sample["target"][:, :, -1:], "chunk_median": sample["chunk_median"]}
