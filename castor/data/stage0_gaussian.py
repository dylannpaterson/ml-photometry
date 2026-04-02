import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
import pandas as pd
import h5py
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import fftconvolve
from scipy.interpolate import UnivariateSpline
from castor.data.transforms import AstroSpaceTransform
from castor.constants import DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL, SHAPE_SIZE, GLOBAL_STRETCH_SCALE, N_PCA_COMPONENTS
from numba import njit
import gc

try:
    import galsim
except ImportError:
    galsim = None

@njit(boundscheck=False)
def fast_paint_grid(lx, ly, fluxes, snrs, psf_weights, sort_idx, min_snr, grid_size, cell_size, K):
    # Size is now exactly 4 + N_PCA (Existence, dx, dy, flux, weights...)
    N_PCA = psf_weights.shape[1]
    grid_stars = np.zeros((grid_size, grid_size, K, 4 + N_PCA), dtype=np.float32)
    counts = np.zeros((grid_size, grid_size), dtype=np.int32)
    
    for idx in range(len(sort_idx)):
        i = sort_idx[idx]
        
        # SNR-based Soft Label for Objectness (Sigmoid Curve)
        # Using the same logic as in stage1_dataset.py
        k = 2.0
        center = 3.0
        snr = snrs[i]
        target_p = 1.0 / (1.0 + np.exp(-k * (snr - center)))
        if snr >= 5.0: target_p = 1.0
        if snr <= 1.0: target_p = 0.0

        # Early Exit: Skip target labels for very faint stars
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
                # Store PCA weights directly in the target
                for w_idx in range(N_PCA):
                    grid_stars[cy, cx, slot, 4 + w_idx] = psf_weights[i, w_idx]
                counts[cy, cx] += 1
                
    return grid_stars

# --- 1. The Dynamic Bulge LF Sampler (RC Prior) ---
def sample_bulge_magnitudes(n_total, rc_mag, rc_sigma, rc_enhancement=3.0, m_min=12.0, m_max=32.0, gamma=0.3):
    """
    Generates a realistic Bulge LF: A continuous exponential RGB/MS 
    with a Red Clump bump anchored to the local background density.
    """
    # 1. Base Population (RGB + Main Sequence)
    u = np.random.uniform(0, 1, n_total)
    a = 10**(gamma * m_min)
    b = 10**(gamma * m_max)
    m_base = (1.0 / gamma) * np.log10(u * (b - a) + a)
    
    # 2. Size the Red Clump proportionally to the local background
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
                 use_fixed_seed=False, global_stretch_scale=GLOBAL_STRETCH_SCALE, min_snr=5.0):
        self.num_samples = num_samples
        self.min_stars = min_stars
        self.max_stars = max_stars
        self.img_size = image_size
        self.K = max_capacity_per_cell
        self.S = shape_size # Now 31 (PCA Basis Resolution)
        self.read_noise = 5.0
        self.use_fixed_seed = use_fixed_seed
        self.min_snr = min_snr
        self.transform = AstroSpaceTransform(stretch_scale=global_stretch_scale)
        self.cell_size = DEFAULT_CELL_SIZE
        self.grid_size = self.img_size // self.cell_size
        self.n_sub = 4
        self.render_kernel_size = 31 # Smooth tails (sigma=1.5)
        self.sigma_fixed = 1.5
        self.n_pca = N_PCA_COMPONENTS
        
        # 1. Generate the standard 100 mathematical Gaussians (31x31)
        raw_library = self._generate_elliptical_library(100, self.render_kernel_size)
        
        # 2. Extract the 20 Eigen-PSFs and the weights for those 100 Gaussians
        self.eigen_psfs, self.psf_weights_lib, self.mean_psf = self._compute_eigen_psfs(raw_library, n_components=self.n_pca)
        
        # 3. Final library for reconstruction: [N_PCA + 1, 961]
        self.psf_library_tensor = torch.cat([
            torch.from_numpy(self.eigen_psfs).view(self.n_pca, -1),
            torch.from_numpy(self.mean_psf).view(1, -1)
        ], dim=0)
        
        self.kernel_bank = self._precompute_kernel_bank(raw_library)
        self.n_pix = 4 * np.pi * (self.sigma_fixed ** 2)
        self.psf_peak = 1.0 / (2 * np.pi * self.sigma_fixed**2)

    def _generate_elliptical_library(self, num_psfs, grid_size):
        library = np.zeros((num_psfs, grid_size, grid_size), dtype=np.float32)
        half = grid_size // 2
        y, x = np.meshgrid(np.arange(grid_size) - half, np.arange(grid_size) - half, indexing='ij')
        
        for i in range(num_psfs):
            q = np.random.uniform(0.7, 1.0)
            theta = np.random.uniform(0, np.pi)
            cos, sin = np.cos(theta), np.sin(theta)
            xp = x * cos + y * sin
            yp = -x * sin + y * cos
            psf = np.exp(-(xp**2 / (2 * self.sigma_fixed**2) + yp**2 / (2 * (self.sigma_fixed * q)**2)))
            psf /= (psf.sum() + 1e-9)
            library[i] = psf
        return library

    def _compute_eigen_psfs(self, large_library, n_components=20):
        """Native PyTorch PCA to extract Eigen-PSFs and their weights."""
        N, H, W = large_library.shape
        data = torch.from_numpy(large_library).float().view(N, H * W)
        mean_psf = data.mean(dim=0)
        centered_data = data - mean_psf
        U, S, V = torch.pca_lowrank(centered_data, q=n_components)
        eigen_psfs = V.t().view(n_components, H, W).numpy()
        psf_weights = (U * S).numpy() 
        return eigen_psfs, psf_weights, mean_psf.view(H, W).numpy()

    def _precompute_kernel_bank(self, raw_library):
        """Precomputes a bank of shifted kernels for high-speed rendering."""
        base_psf = raw_library[0]
        bank = {}
        for i in range(self.n_sub):
            for j in range(self.n_sub):
                bank[(i, j)] = base_psf 
        return bank

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.use_fixed_seed: np.random.seed(idx)
        sample = self.generate_chunk()
        return {"image": sample["image"], "target": sample["target"], "chunk_median": sample["chunk_median"], "psf_library": sample["psf_library"]}

    def generate_chunk(self, rc_params=None, exp_params=None):
        if rc_params is None:
            rc_loc = np.random.uniform(14.5, 16.5)
            rc_scale = np.random.uniform(0.2, 0.5)
            rc_enhancement = np.random.uniform(5.0, 15.0)
            lf_gamma = np.random.uniform(0.25, 0.35)
        else: rc_loc, rc_scale, rc_enhancement, lf_gamma = rc_params

        if exp_params is None:
            exp_time = np.random.uniform(30.0, 60.0)
            zp, sky_mag = 26.5, 22.0
        else: exp_time, zp, sky_mag = exp_params

        pixel_scale = 0.11 
        sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (pixel_scale**2) * exp_time
        
        # Unified massive population
        n_stars_base = int(np.random.uniform(self.min_stars, self.max_stars))
        mags = sample_bulge_magnitudes(n_stars_base, rc_loc, rc_scale, rc_enhancement, m_min=12.0, m_max=32.0, gamma=lf_gamma)
        n_stars = len(mags)
        fluxes = exp_time * (10 ** (-0.4 * (mags - zp)))
        sort_idx = np.argsort(fluxes)[::-1]
        fluxes, mags = fluxes[sort_idx], mags[sort_idx]
        x_centers = np.random.uniform(0, self.img_size, n_stars)
        y_centers = np.random.uniform(0, self.img_size, n_stars)

        # 1. Assign a random PSF index from the elliptical library
        psf_indices = np.random.randint(0, 100, size=n_stars)
        
        # 2. Get the continuous PCA weights for these stars
        psf_weights = self.psf_weights_lib[psf_indices] # [n_stars, N_PCA]

        # Rendering
        star_signal = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        monster_cutoff = min(100, int(n_stars * 0.0005))
        cx, cy, cf = x_centers[monster_cutoff:], y_centers[monster_cutoff:], fluxes[monster_cutoff:]
        x0, y0 = np.floor(cx).astype(int), np.floor(cy).astype(int)
        phase_x = np.clip(np.floor((cx - x0) * self.n_sub).astype(int), 0, self.n_sub - 1)
        phase_y = np.clip(np.floor((cy - y0) * self.n_sub).astype(int), 0, self.n_sub - 1)

        for i in range(self.n_sub):
            for j in range(self.n_sub):
                mask = (phase_x == i) & (phase_y == j)
                if mask.any():
                    phase_map, _, _ = np.histogram2d(
                        y0[mask], x0[mask], 
                        bins=self.img_size, 
                        range=[[0, self.img_size], [0, self.img_size]], 
                        weights=cf[mask]
                    )
                    star_signal += fftconvolve(phase_map, self.kernel_bank[(i, j)], mode='same').astype(np.float32)

        for i in range(monster_cutoff):
            fx, fy, f, p_idx = x_centers[i], y_centers[i], fluxes[i], psf_indices[i]
            half = self.render_kernel_size // 2
            ix, iy = int(fx), int(fy)
            y0_m, y1_m = max(0, iy-half), min(self.img_size, iy+half+1)
            x0_m, x1_m = max(0, ix-half), min(self.img_size, ix+half+1)
            sy0, sy1 = half - (iy - y0_m), half + (y1_m - iy)
            sx0, sx1 = half - (ix - x0_m), half + (x1_m - ix)
            
            # Reconstructing high-res PSF for monster rendering
            psf_reconstructed = (self.eigen_psfs.reshape(self.n_pca, -1).T @ self.psf_weights_lib[p_idx]).reshape(self.render_kernel_size, self.render_kernel_size) + self.mean_psf
            stamp = psf_reconstructed[sy0:sy1, sx0:sx1] * f
            star_signal[y0_m:y1_m, x0_m:x1_m] += stamp

        total_photon_flux = star_signal + sky_level
        chunk_median = np.median(star_signal) + sky_level

        # --- Sub-Pixel Accurate Local Confusion SNR ---
        total_local_light = map_coordinates(star_signal, [y_centers, x_centers], order=1, mode='nearest')
        local_background = np.maximum(0, total_local_light - (fluxes * self.psf_peak))
        noise_variance = fluxes + self.n_pix * (sky_level + local_background + self.read_noise**2)
        snrs = fluxes / np.sqrt(noise_variance)

        # Target Construction (Numba Optimized PCA weights)
        base_grid = fast_paint_grid(
            x_centers, y_centers, fluxes, snrs, psf_weights, sort_idx, 
            self.min_snr, self.grid_size, self.cell_size, self.K
        )

        bg_target = self.transform.target_bg_to_network(sky_level - chunk_median)
        bg_grid = np.full((self.grid_size, self.grid_size, 1), bg_target, dtype=np.float32)
        target = torch.cat([torch.from_numpy(base_grid).view(self.grid_size, self.grid_size, -1), 
                            torch.from_numpy(bg_grid)], dim=-1)

        return {
            "image": torch.from_numpy(total_photon_flux).unsqueeze(0), 
            "target": target, 
            "chunk_median": float(chunk_median), 
            "psf_library": self.psf_library_tensor.unsqueeze(0) # [1, N_PCA + 1, 961]
        }

class GaussianMosaicDataset(Dataset):
    def __init__(self, data_dir, num_samples=25000, image_size=256, cell_size=DEFAULT_CELL_SIZE, global_stretch_scale=GLOBAL_STRETCH_SCALE):
        self.data_dir, self.num_samples, self.img_size, self.cell_size = data_dir, num_samples, image_size, cell_size
        self.grid_size = image_size // cell_size
        self.transform = AstroSpaceTransform(stretch_scale=global_stretch_scale)
        self.K = MAX_CAPACITY_PER_CELL
        self.N_PCA = N_PCA_COMPONENTS
        self.min_snr = 5.0
        
        # Pre-allocate target shape info: K * (4 + N_PCA) + 1
        self.target_shape = (self.grid_size, self.grid_size, self.K * (4 + self.N_PCA) + 1)
        
        # Load mosaic manifests
        self.mosaics = []
        image_files = sorted([f for f in os.listdir(data_dir) if f.endswith("_img.npy")])
        for img_f in image_files:
            base = img_f.replace("_img.npy", "")
            cat_f = base + "_cat.npy"
            meta_f = base + "_meta.npy"
            lib_f = base + "_psf_lib.npy"
            
            if os.path.exists(os.path.join(data_dir, cat_f)) and os.path.exists(os.path.join(data_dir, meta_f)):
                meta = np.load(os.path.join(data_dir, meta_f))
                lib_path = os.path.join(data_dir, lib_f) if os.path.exists(os.path.join(data_dir, lib_f)) else None
                self.mosaics.append({
                    'img_path': os.path.join(data_dir, img_f),
                    'cat_path': os.path.join(data_dir, cat_f),
                    'lib_path': lib_path,
                    'exp_time': meta[0],
                    'zp': meta[1],
                    'sky_mag': meta[2]
                })
        
        if not self.mosaics:
            return

        self.active_mosaic_idx = np.random.randint(0, len(self.mosaics))
        self._load_mosaic_to_ram(self.active_mosaic_idx)
        self.max_samples_per_mosaic = 384 
        self.samples_from_current = np.random.randint(0, self.max_samples_per_mosaic)

    def _load_mosaic_to_ram(self, m_idx):
        self.active_img = None
        self.active_cat = None
        self.active_library = None
        gc.collect()
        
        mosaic = self.mosaics[m_idx]
        self.active_img = np.load(mosaic['img_path'])
        self.active_cat = np.load(mosaic['cat_path']) 
        if mosaic['lib_path']:
            self.active_library = np.load(mosaic['lib_path'])
            
        self.active_mosaic_idx = m_idx
        self.samples_from_current = 0

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        if self.samples_from_current >= self.max_samples_per_mosaic:
            new_idx = np.random.randint(0, len(self.mosaics))
            self._load_mosaic_to_ram(new_idx)
            
        self.samples_from_current += 1
        mosaic = self.mosaics[self.active_mosaic_idx]
        
        my, mx = self.active_img.shape
        py = np.random.randint(0, my - self.img_size)
        px = np.random.randint(0, mx - self.img_size)
        star_signal_np = self.active_img[py:py+self.img_size, px:px+self.img_size]
        
        pixel_scale = 0.11
        sky_level = (10 ** (-0.4 * (mosaic['sky_mag'] - mosaic['zp']))) * (pixel_scale**2) * mosaic['exp_time']
        
        signal_tensor = torch.from_numpy(star_signal_np).float()
        signal_tensor.add_(sky_level).clamp_(min=0.0).unsqueeze_(0)
        chunk_median = np.median(star_signal_np) + sky_level
        
        y_start = np.searchsorted(self.active_cat['y'], py)
        y_end = np.searchsorted(self.active_cat['y'], py + self.img_size)
        band_cat = self.active_cat[y_start:y_end]
        mask_x = (band_cat['x'] >= px) & (band_cat['x'] < px + self.img_size)
        
        target_buffer = np.zeros(self.target_shape, dtype=np.float32)
        
        if mask_x.any():
            local_cat = band_cat[mask_x]
            lx, ly = local_cat['x'] - px, local_cat['y'] - py
            fluxes = local_cat['flux']
            snrs = local_cat['snr']
            
            # Continuous PCA weights must be in the catalog for this to work
            psf_weights = np.column_stack([local_cat[f'w{i}'] for i in range(self.N_PCA)])
            
            sort_idx = np.argsort(fluxes)[::-1]
            grid_stars_np = fast_paint_grid(
                lx, ly, fluxes, snrs, psf_weights, sort_idx, 
                self.min_snr, self.grid_size, self.cell_size, self.K
            )
            target_buffer[:, :, :-1] = grid_stars_np.reshape(self.grid_size, self.grid_size, -1)
        
        bg_val = self.transform.target_bg_to_network(sky_level - chunk_median)
        target_buffer[:, :, -1] = bg_val
        
        return {
            "image": signal_tensor, 
            "target": torch.from_numpy(target_buffer), 
            "chunk_median": float(chunk_median),
            "psf_library": torch.from_numpy(self.active_library).unsqueeze(0)
        }

class HDF5MosaicDataset(Dataset):
    def __init__(self, h5_path, image_size=256):
        self.h5_path = h5_path
        self.file = None
        self.img_size = image_size
        self.cell_size = DEFAULT_CELL_SIZE
        self.grid_size = self.img_size // self.cell_size
        self.K = MAX_CAPACITY_PER_CELL
        
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        with h5py.File(self.h5_path, 'r') as f:
            self.length = len(f['images'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')
            
        img = self.file['images'][idx]
        target = self.file['targets'][idx]
        psf = self.file['psf_libraries'][idx]
        median = self.file['chunk_medians'][idx]

        return {
            "image": torch.from_numpy(img),
            "target": torch.from_numpy(target),
            "psf_library": torch.from_numpy(psf),
            "chunk_median": float(median)
        }

    def generate_chunk(self):
        """Compatibility method for older analysis scripts."""
        import random
        idx = random.randint(0, len(self) - 1)
        sample = self[idx]
        target = sample["target"]
        base_grid = target[:, :, :-1].view(self.grid_size, self.grid_size, self.K, -1)
        
        return {
            "image": sample["image"],
            "base_grid": base_grid,
            "background_map": target[:, :, -1:],
            "chunk_median": sample["chunk_median"]
        }
