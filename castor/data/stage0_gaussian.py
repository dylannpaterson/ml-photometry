import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import fftconvolve
from castor.data.transforms import AstroSpaceTransform
from castor.constants import DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL, SHAPE_SIZE, GLOBAL_STRETCH_SCALE
from numba import njit

try:
    import galsim
except ImportError:
    galsim = None

@njit(fastmath=True)
def fast_paint_grid(lx, ly, fluxes, snrs, shapes, sort_idx, min_snr, grid_size, cell_size, K, S_sq):
    grid_stars = np.zeros((grid_size, grid_size, K, 5 + S_sq), dtype=np.float32)
    counts = np.zeros((grid_size, grid_size), dtype=np.int32)
    
    for idx in range(len(sort_idx)):
        i = sort_idx[idx]
        if snrs[i] >= min_snr:
            cx = int(lx[i] // cell_size)
            cy = int(ly[i] // cell_size)
            
            if 0 <= cx < grid_size and 0 <= cy < grid_size:
                slot = counts[cy, cx]
                if slot < K:
                    comp = 1.0 / (1.0 + np.exp(-2.0 * (snrs[i] - min_snr)))
                    grid_stars[cy, cx, slot, 0] = comp
                    grid_stars[cy, cx, slot, 1] = lx[i] % cell_size
                    grid_stars[cy, cx, slot, 2] = ly[i] % cell_size
                    grid_stars[cy, cx, slot, 3] = fluxes[i]
                    grid_stars[cy, cx, slot, 4] = comp
                    
                    # Numba handles this inner assignment instantly
                    for s in range(S_sq):
                        grid_stars[cy, cx, slot, 5+s] = shapes[i, s]
                        
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
        self.S = shape_size
        self.read_noise = 5.0
        self.use_fixed_seed = use_fixed_seed
        self.min_snr = min_snr
        self.transform = AstroSpaceTransform(stretch_scale=global_stretch_scale)
        self.cell_size = DEFAULT_CELL_SIZE
        self.grid_size = self.img_size // self.cell_size
        self.n_sub = 4
        self.kernel_size = 63
        self.sigma_fixed = 1.5
        self.kernel_bank = self._precompute_kernel_bank(sigma=self.sigma_fixed)
        self.n_pix = 4 * np.pi * (self.sigma_fixed ** 2)
        self.psf_peak = 1.0 / (2 * np.pi * self.sigma_fixed**2)

    def _precompute_kernel_bank(self, sigma=1.5):
        bank = {}
        pixel_scale = 0.11
        half = self.kernel_size // 2
        gy, gx = np.meshgrid(np.arange(self.kernel_size), np.arange(self.kernel_size))
        for i in range(self.n_sub):
            for j in range(self.n_sub):
                dx_shift = (i + 0.5) / self.n_sub
                dy_shift = (j + 0.5) / self.n_sub
                if galsim is not None:
                    base_psf = galsim.Gaussian(sigma=sigma * pixel_scale)
                    shifted_psf = base_psf.shift(dx_shift * pixel_scale, dy_shift * pixel_scale)
                    stamp = galsim.ImageF(self.kernel_size, self.kernel_size, scale=pixel_scale)
                    shifted_psf.drawImage(image=stamp, method='no_pixel')
                    kernel = stamp.array
                else:
                    kernel = np.exp(-((gx - (half + dx_shift))**2 + (gy - (half + dy_shift))**2) / (2 * sigma**2))
                    kernel /= (kernel.sum() + 1e-9)
                bank[(i, j)] = kernel.astype(np.float32)
        return bank

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.use_fixed_seed: np.random.seed(idx)
        sample = self.generate_chunk()
        return {"image": sample["image"], "target": sample["target"], "chunk_median": sample["chunk_median"]}

    def generate_chunk(self, rc_params=None, exp_params=None):
        if rc_params is None:
            rc_loc = np.random.uniform(14.5, 16.5)
            rc_scale = np.random.uniform(0.2, 0.5)
            rc_enhancement = np.random.uniform(5.0, 15.0)
        else: rc_loc, rc_scale, rc_enhancement = rc_params

        if exp_params is None:
            exp_time = np.random.uniform(30.0, 60.0)
            zp, sky_mag = 26.5, 22.0
        else: exp_time, zp, sky_mag = exp_params

        pixel_scale = 0.11 
        sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (pixel_scale**2) * exp_time
        
        # Unified massive population
        n_stars_base = int(np.random.uniform(self.min_stars, self.max_stars))
        mags = sample_bulge_magnitudes(n_stars_base, rc_loc, rc_scale, rc_enhancement, m_min=12.0, m_max=32.0)
        n_stars = len(mags)
        fluxes = exp_time * (10 ** (-0.4 * (mags - zp)))
        sort_idx = np.argsort(fluxes)[::-1]
        fluxes, mags = fluxes[sort_idx], mags[sort_idx]
        x_centers = np.random.uniform(0, self.img_size, n_stars)
        y_centers = np.random.uniform(0, self.img_size, n_stars)

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
            fx, fy, f = x_centers[i], y_centers[i], fluxes[i]
            half = 15
            ix, iy = int(fx), int(fy)
            y0_m, y1_m = max(0, iy-half), min(self.img_size, iy+half+1)
            x0_m, x1_m = max(0, ix-half), min(self.img_size, ix+half+1)
            yy, xx = np.meshgrid(np.arange(y0_m, y1_m), np.arange(x0_m, x1_m), indexing='ij')
            stamp = np.exp(-((xx - fx)**2 + (yy - fy)**2) / (2 * self.sigma_fixed**2))
            stamp *= (f / (2 * np.pi * self.sigma_fixed**2))
            star_signal[y0_m:y1_m, x0_m:x1_m] += stamp

        total_photon_flux = star_signal + sky_level
        raw_image = np.random.normal(loc=total_photon_flux, scale=np.sqrt(total_photon_flux + self.read_noise**2)).astype(np.float32)
        chunk_median = np.median(raw_image)
        normalized_image = self.transform.image_to_network(raw_image, chunk_median)

        # --- Sub-Pixel Accurate Local Confusion SNR ---
        # Sample total light using bilinear interpolation at EXACT centers
        total_local_light = map_coordinates(star_signal, [y_centers, x_centers], order=1, mode='nearest')
        local_background = np.maximum(0, total_local_light - (fluxes * self.psf_peak))
        noise_variance = fluxes + self.n_pix * (sky_level + local_background + self.read_noise**2)
        snrs = fluxes / np.sqrt(noise_variance)

        # Target Construction
        base_grid = np.zeros((self.grid_size, self.grid_size, self.K, 5), dtype=np.float32)
        half = self.S // 2
        grid_range = np.arange(-half, half + 1)
        sy, sx = np.meshgrid(grid_range, grid_range, indexing='ij')
        psf_flat = np.exp(-(sx**2 + sy**2) / (2 * self.sigma_fixed**2))
        psf_flat = (psf_flat / psf_flat.sum()).astype(np.float32).flatten()
        
        shapes, catalog_stars = [], []
        cell_assignments = {}
        for i in range(n_stars):
            if snrs[i] >= self.min_snr:
                tx, ty = x_centers[i], y_centers[i]
                cx, cy = int(tx // self.cell_size), int(ty // self.cell_size)
                if 0 <= cx < self.grid_size and 0 <= cy < self.grid_size:
                    if (cy, cx) not in cell_assignments: cell_assignments[(cy, cx)] = []
                    cell_assignments[(cy, cx)].append([fluxes[i], tx, ty, snrs[i], mags[i]])

        for (cy, cx), cell_stars in cell_assignments.items():
            sorted_stars = sorted(cell_stars, key=lambda x: x[0], reverse=True)
            for slot in range(min(self.K, len(sorted_stars))):
                f, tx, ty, snr, m = sorted_stars[slot]
                comp = 1.0 / (1.0 + np.exp(-2.0 * (snr - self.min_snr)))
                base_grid[cy, cx, slot] = [comp, tx % self.cell_size, ty % self.cell_size, f, comp]
                shapes.append(psf_flat)
                catalog_stars.append({'x': tx, 'y': ty, 'flux': f, 'mag': m, 'shape': psf_flat, 'exp_time': exp_time, 'zp': zp, 'sky_mag': sky_mag})

        bg_target = self.transform.target_bg_to_network(sky_level - chunk_median)
        bg_grid = np.full((self.grid_size, self.grid_size), bg_target, dtype=np.float32)
        target = torch.cat([torch.from_numpy(base_grid).view(self.grid_size, self.grid_size, -1), 
                            torch.from_numpy(bg_grid).unsqueeze(-1)], dim=-1)

        return {"image": torch.from_numpy(normalized_image).unsqueeze(0), "raw_image": torch.from_numpy(raw_image), 
                "physics_image": torch.from_numpy(star_signal), "target": target, "chunk_median": float(chunk_median), 
                "catalog": pd.DataFrame(catalog_stars)}

class GaussianMosaicDataset(Dataset):
    def __init__(self, data_dir, num_samples=25000, image_size=256, cell_size=DEFAULT_CELL_SIZE, global_stretch_scale=GLOBAL_STRETCH_SCALE):
        self.data_dir, self.num_samples, self.img_size, self.cell_size = data_dir, num_samples, image_size, cell_size
        self.grid_size = image_size // cell_size
        self.transform = AstroSpaceTransform(stretch_scale=global_stretch_scale)
        self.K, self.S = MAX_CAPACITY_PER_CELL, SHAPE_SIZE
        self.sigma_fixed = 1.5
        self.n_pix = 4 * np.pi * (self.sigma_fixed ** 2)
        self.psf_peak = 1.0 / (2 * np.pi * self.sigma_fixed**2)
        self.min_snr = 5.0
        
        # Tracking for Worker-Pinned RAM Cache
        self.active_mosaic_idx = -1
        self.active_img = None
        self.active_cat = None
        self.samples_from_current = 0
        self.max_samples_per_mosaic = 200 # Rotate every 200 samples
        
        # Load mosaic manifests
        self.mosaics = []
        image_files = sorted([f for f in os.listdir(data_dir) if f.endswith("_img.npy")])
        for img_f in image_files:
            base = img_f.replace("_img.npy", "")
            cat_f = base + "_cat.npy"
            meta_f = base + "_meta.npy"
            
            if os.path.exists(os.path.join(data_dir, cat_f)) and os.path.exists(os.path.join(data_dir, meta_f)):
                meta = np.load(os.path.join(data_dir, meta_f))
                self.mosaics.append({
                    'img_path': os.path.join(data_dir, img_f),
                    'cat_path': os.path.join(data_dir, cat_f),
                    'exp_time': meta[0],
                    'zp': meta[1],
                    'sky_mag': meta[2]
                })
        
        if not self.mosaics:
            print(f"⚠️ Warning: No optimized mosaics found in {data_dir}. Check generation script.")

    def _load_mosaic_to_ram(self, m_idx):
        """Sequential block read into RAM to bypass choked SSD IOPS."""
        mosaic = self.mosaics[m_idx]
        self.active_img = np.load(mosaic['img_path'])
        
        cat_raw = np.load(mosaic['cat_path'])
        # Sort by Y for binary search optimization
        self.active_cat = cat_raw[np.argsort(cat_raw['y'])]
        
        self.active_mosaic_idx = m_idx
        self.samples_from_current = 0

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        # 1. Manage RAM Cache
        if self.active_mosaic_idx == -1 or self.samples_from_current >= self.max_samples_per_mosaic:
            # Important: Use fresh random for workers
            new_idx = np.random.randint(0, len(self.mosaics))
            self._load_mosaic_to_ram(new_idx)
            
        self.samples_from_current += 1
        mosaic = self.mosaics[self.active_mosaic_idx]
        
        # 2. Slice from RAM
        my, mx = self.active_img.shape
        py = np.random.randint(0, my - self.img_size)
        px = np.random.randint(0, mx - self.img_size)
        star_signal_np = self.active_img[py:py+self.img_size, px:px+self.img_size].copy()
        
        # 3. Add Live Noise using PyTorch
        pixel_scale = 0.11
        sky_level = (10 ** (-0.4 * (mosaic['sky_mag'] - mosaic['zp']))) * (pixel_scale**2) * mosaic['exp_time']
        signal_tensor = torch.from_numpy(star_signal_np + sky_level).clamp(min=0)
        img_noisy_tensor = torch.poisson(signal_tensor) + torch.randn_like(signal_tensor) * 5.0
        
        img_noisy_np = img_noisy_tensor.numpy()
        chunk_median = np.median(img_noisy_np)
        image_tensor = torch.from_numpy(self.transform.image_to_network(img_noisy_np, chunk_median)).unsqueeze(0).float()

        # 4. Binary Search spatial filter in RAM
        y_start = np.searchsorted(self.active_cat['y'], py)
        y_end = np.searchsorted(self.active_cat['y'], py + self.img_size)
        
        band_cat = self.active_cat[y_start:y_end]
        mask_x = (band_cat['x'] >= px) & (band_cat['x'] < px + self.img_size)
        
        if not mask_x.any():
            return {"image": image_tensor, "target": torch.zeros((self.grid_size, self.grid_size, self.K*(5+self.S**2) + 1)), "chunk_median": float(chunk_median)}
        
        local_cat = band_cat[mask_x]
        lx, ly = local_cat['x'] - px, local_cat['y'] - py
        fluxes = local_cat['flux']
        
        # 5. Optimized SNR Update (Nearest Neighbor)
        ly_idx = np.clip(ly.astype(np.int32), 0, self.img_size - 1)
        lx_idx = np.clip(lx.astype(np.int32), 0, self.img_size - 1)
        total_local_light = star_signal_np[ly_idx, lx_idx]
        
        local_background = np.maximum(0, total_local_light - (fluxes * self.psf_peak))
        noise_variance = fluxes + self.n_pix * (sky_level + local_background + 25.0)
        snrs = fluxes / np.sqrt(np.maximum(1.0, noise_variance))
        
        # 6. Grid Painting (Numba Optimized)
        sort_idx = np.argsort(fluxes)[::-1]
        grid_stars_np = fast_paint_grid(
            lx, ly, fluxes, snrs, local_cat['shape'], sort_idx, 
            self.min_snr, self.grid_size, self.cell_size, self.K, self.S**2
        )
        
        bg_grid = np.full((self.grid_size, self.grid_size, 1), self.transform.target_bg_to_network(sky_level - chunk_median), dtype=np.float32)
        target = torch.cat([torch.from_numpy(grid_stars_np).view(self.grid_size, self.grid_size, -1), torch.from_numpy(bg_grid)], dim=-1)
        
        return {"image": image_tensor, "target": target, "chunk_median": float(chunk_median)}
