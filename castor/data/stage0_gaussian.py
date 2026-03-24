import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from castor.data.transforms import AstroSpaceTransform
from castor.constants import DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL, SHAPE_SIZE, GLOBAL_STRETCH_SCALE

try:
    import galsim
except ImportError:
    galsim = None

# --- 1. The Dynamic Bulge LF Sampler (RC Prior) ---
def sample_bulge_magnitudes(n_stars, rc_mag, rc_sigma, rc_fraction, m_min=12.0, m_max=26.0, gamma=0.3):
    """Samples apparent magnitudes (m). Smaller m = brighter star."""
    n_rc = int(n_stars * rc_fraction)
    n_bg = n_stars - n_rc

    # 1. True Exponential Background via Inverse Transform Sampling
    # N(m) proportional to 10^(gamma * m)
    # gamma ~ 0.3 is a reasonable generic slope for star counts
    u = np.random.uniform(0, 1, n_bg)
    
    # Inverse CDF of the exponential distribution bounded by m_min and m_max
    a = 10**(gamma * m_min)
    b = 10**(gamma * m_max)
    m_bg = (1.0 / gamma) * np.log10(u * (b - a) + a)

    # 2. Red Clump (Gaussian in magnitude space)
    m_rc = np.random.normal(loc=rc_mag, scale=rc_sigma, size=n_rc)

    # Combine and clip (just in case)
    m_all = np.concatenate([m_bg, m_rc])
    m_all = np.clip(m_all, m_min, m_max)

    return m_all 

class GaussianPretrainingProvider(Dataset):
    def __init__(self, num_samples=1000, min_stars=100, max_stars=1500, image_size=256, 
                 max_capacity_per_cell=MAX_CAPACITY_PER_CELL, shape_size=SHAPE_SIZE, 
                 use_fixed_seed=False, global_stretch_scale=GLOBAL_STRETCH_SCALE, min_snr=5.0):
        """
        Generates realistic synthetic data for the Roman Bulge Time Domain Survey.
        Uses Phase-Banked FFT Kernels for perfect sub-pixel accuracy.
        """
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

        # Grid parameters
        self.cell_size = DEFAULT_CELL_SIZE
        self.grid_size = self.img_size // self.cell_size
        
        # Phase-Banked FFT Kernel Configuration (4x4 sub-pixel grid)
        self.n_sub = 4
        self.kernel_size = 63
        self.sigma_fixed = 1.5
        self.kernel_bank = self._precompute_kernel_bank(sigma=self.sigma_fixed)
        
        # Effective PSF area for SNR calculation (FIX 1)
        self.n_pix = 4 * np.pi * (self.sigma_fixed ** 2)
        
        # Pre-allocate coordinate grids for vectorization
        self.xx, self.yy = np.meshgrid(np.arange(self.img_size), np.arange(self.img_size))

    def _precompute_kernel_bank(self, sigma=1.5):
        """Pre-renders 16 distinct sub-pixel shifted kernels for perfect convolution."""
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
        if self.use_fixed_seed:
            np.random.seed(idx)
        sparse_sample = self.generate_chunk()
        image = sparse_sample["image"]
        
        base_grid = sparse_sample["base_grid"]
        bg_map = sparse_sample["background_map"]
        shapes = sparse_sample["shapes"]
        indices = sparse_sample["indices"]
        
        S2 = self.S * self.S
        grid_size = self.grid_size
        
        star_targets = torch.zeros((grid_size, grid_size, self.K, 5 + S2), dtype=torch.float32)
        star_targets[..., :5] = base_grid
        
        if len(indices) > 0:
            for i in range(len(indices)):
                y, x, k = indices[i]
                star_targets[y, x, k, 5:5+S2] = shapes[i]
        
        flattened_stars = star_targets.view(grid_size, grid_size, -1)
        target = torch.cat([flattened_stars, bg_map.unsqueeze(-1)], dim=-1)
        
        return {
            "image": image,
            "target": target,
            "chunk_median": sparse_sample["chunk_median"]
        }

    def generate_chunk(self, rc_params=None, exp_params=None):
        """Generates a realistic Roman-like chunk using the Phase-Banked FFT Engine."""
        if rc_params is None:
            rc_loc = np.random.uniform(14.5, 16.5)
            rc_scale = np.random.uniform(0.2, 0.5)
            rc_fraction = np.random.uniform(0.05, 0.20)
        else:
            rc_loc, rc_scale, rc_fraction = rc_params

        if exp_params is None:
            exp_time = np.random.uniform(30.0, 60.0)
            zp = 26.5
            sky_mag = 22.0
        else:
            exp_time, zp, sky_mag = exp_params

        pixel_scale = 0.11 
        sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (pixel_scale**2) * exp_time
        
        # NEW: Unified population rendering (deep exponential LF)
        n_stars = 100000 
        mags = sample_bulge_magnitudes(n_stars, rc_loc, rc_scale, rc_fraction, m_min=12.0, m_max=32.0)
        fluxes = exp_time * (10 ** (-0.4 * (mags - zp)))
        
        # Sort by flux
        sort_idx = np.argsort(fluxes)[::-1]
        fluxes = fluxes[sort_idx]
        mags = mags[sort_idx]
        
        x_centers = np.random.uniform(0, self.img_size, n_stars)
        y_centers = np.random.uniform(0, self.img_size, n_stars)

        # --- Phase-Banked Two-Tier Rendering ---
        star_signal = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        monster_cutoff = int(n_stars * 0.001)
        
        cx, cy, cf = x_centers[monster_cutoff:], y_centers[monster_cutoff:], fluxes[monster_cutoff:]
        x0, y0 = np.floor(cx).astype(int), np.floor(cy).astype(int)
        phase_x = np.clip(np.floor((cx - x0) * self.n_sub).astype(int), 0, self.n_sub - 1)
        phase_y = np.clip(np.floor((cy - y0) * self.n_sub).astype(int), 0, self.n_sub - 1)

        for i in range(self.n_sub):
            for j in range(self.n_sub):
                phase_map = np.zeros((self.img_size, self.img_size), dtype=np.float32)
                mask = (phase_x == i) & (phase_y == j)
                if mask.any():
                    valid = (x0[mask] >= 0) & (x0[mask] < self.img_size) & (y0[mask] >= 0) & (y0[mask] < self.img_size)
                    np.add.at(phase_map, (y0[mask][valid], x0[mask][valid]), cf[mask][valid])
                
                if phase_map.any():
                    star_signal += fftconvolve(phase_map, self.kernel_bank[(i, j)], mode='same')

        for i in range(monster_cutoff):
            fx, fy, flux = x_centers[i], y_centers[i], fluxes[i]
            half = 15
            ix, iy = int(fx), int(fy)
            y0_m, y1_m = max(0, iy-half), min(self.img_size, iy+half+1)
            x0_m, x1_m = max(0, ix-half), min(self.img_size, ix+half+1)
            yy, xx = np.meshgrid(np.arange(y0_m, y1_m), np.arange(x0_m, x1_m), indexing='ij')
            stamp = np.exp(-((xx - fx)**2 + (yy - fy)**2) / (2 * self.sigma_fixed**2))
            stamp *= (flux / (stamp.sum() + 1e-9))
            star_signal[y0_m:y1_m, x0_m:x1_m] += stamp

        gt_background = np.full((self.img_size, self.img_size), sky_level, dtype=np.float32)
        total_photon_flux = gt_background + star_signal
        noise_std = np.sqrt(total_photon_flux + self.read_noise**2)
        raw_image = np.random.normal(loc=total_photon_flux, scale=noise_std).astype(np.float32)

        chunk_median = np.median(raw_image)
        normalized_image = self.transform.image_to_network(raw_image, chunk_median)

        # --- Target Grid Construction ---
        base_grid = np.zeros((self.grid_size, self.grid_size, self.K, 5), dtype=np.float32)
        half = self.S // 2
        grid_range = np.arange(-half, half + 1)
        sy, sx = np.meshgrid(grid_range, grid_range, indexing='ij')
        psf_shape = np.exp(-(sx**2 + sy**2) / (2 * self.sigma_fixed**2))
        psf_shape /= (psf_shape.sum() + 1e-9)
        psf_shape_flat = psf_shape.astype(np.float32).flatten()
        
        shapes, indices, catalog_stars = [], [], []
        cell_assignments = {}
        for i in range(n_stars):
            tx, ty, flux = x_centers[i], y_centers[i], fluxes[i]
            cx, cy = int(tx // self.cell_size), int(ty // self.cell_size)
            if cx < 0 or cx >= self.grid_size or cy < 0 or cy >= self.grid_size: continue
            
            noise_variance = flux + self.n_pix * (sky_level + self.read_noise**2)
            snr = flux / np.sqrt(noise_variance)
            
            if snr >= self.min_snr:
                if (cy, cx) not in cell_assignments: cell_assignments[(cy, cx)] = []
                cell_assignments[(cy, cx)].append([flux, tx, ty, snr, mags[i]])

        for (cy, cx), cell_stars in cell_assignments.items():
            sorted_stars = sorted(cell_stars, key=lambda x: x[0], reverse=True)
            for slot in range(min(self.K, len(sorted_stars))):
                flux, tx, ty, snr, mag = sorted_stars[slot]
                completeness = 1.0 / (1.0 + np.exp(-2.0 * (snr - self.min_snr)))
                base_grid[cy, cx, slot] = [completeness, tx % self.cell_size, ty % self.cell_size, flux, completeness]
                shapes.append(psf_shape_flat)
                indices.append([cy, cx, slot])
                catalog_stars.append({'x': tx, 'y': ty, 'flux': flux, 'mag': mag, 'shape': psf_shape_flat,
                                      'exp_time': exp_time, 'zp': zp, 'sky_mag': sky_mag})

        residual_bg_linear = gt_background - chunk_median
        bg_grid_stretched = self.transform.target_bg_to_network(residual_bg_linear).reshape(self.grid_size, self.cell_size, self.grid_size, self.cell_size).mean(axis=(1, 3))

        return {
            "image": torch.from_numpy(normalized_image).unsqueeze(0),
            "raw_image": torch.from_numpy(raw_image),
            "physics_image": torch.from_numpy(star_signal),
            "base_grid": torch.from_numpy(base_grid),
            "background_map": torch.from_numpy(bg_grid_stretched),
            "shapes": torch.from_numpy(np.array(shapes)) if shapes else torch.tensor([]),
            "indices": torch.from_numpy(np.array(indices)) if indices else torch.tensor([]),
            "chunk_median": float(chunk_median),
            "catalog": pd.DataFrame(catalog_stars),
            "exp_time": exp_time, "zp": zp, "sky_mag": sky_mag
        }

class GaussianMosaicDataset(Dataset):
    # Class-level cache to share index between train and val
    _STAR_INDEX_CACHE = {}

    def __init__(self, data_dir, num_samples=25000, image_size=256, cell_size=DEFAULT_CELL_SIZE, global_stretch_scale=GLOBAL_STRETCH_SCALE):
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.img_size = image_size
        self.cell_size = cell_size
        self.grid_size = image_size // cell_size
        self.transform = AstroSpaceTransform(stretch_scale=global_stretch_scale)
        self.K, self.S = MAX_CAPACITY_PER_CELL, SHAPE_SIZE
        
        # Effective PSF area for SNR calculation
        sigma_fixed = 1.5
        self.n_pix = 4 * np.pi * (sigma_fixed ** 2)
        self.min_snr = 5.0
        
        self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith("_img.npy")])
        if not self.image_files:
            raise FileNotFoundError(f"No mosaics found in {data_dir}.")
            
        cache_key = os.path.abspath(data_dir)
        if cache_key in self._STAR_INDEX_CACHE:
            print(f"🚀 Reusing cached Stage 0 Star Index for {data_dir}...")
            cached_data = self._STAR_INDEX_CACHE[cache_key]
            self.mosaics = cached_data['mosaics']
            self.all_stars = cached_data['all_stars']
            self.star_offsets = cached_data['star_offsets']
        else:
            print(f"🚀 Optimizing Stage 0: Indexing {len(self.image_files)} mosaics for High-Speed training...")
            
            self.mosaics = []
            all_stars = []
            star_offsets = [0]
            
            for i, img_f in enumerate(self.image_files):
                base = img_f.replace("_img.npy", "")
                cat_f = [f for f in os.listdir(data_dir) if f.startswith(base) and f.endswith(".parquet")]
                if not cat_f: continue
                
                img_path = os.path.join(data_dir, img_f)
                cat_path = os.path.join(data_dir, cat_f[0])
                
                cat = pd.read_parquet(cat_path)
                
                self.mosaics.append({
                    'img_path': img_path,
                    'exp_time': cat['exp_time'].iloc[0] if 'exp_time' in cat.columns else 54.0,
                    'zp': cat['zp'].iloc[0] if 'zp' in cat.columns else 26.5,
                    'sky_mag': cat['sky_mag'].iloc[0] if 'sky_mag' in cat.columns else 22.0
                })
                
                stars = np.zeros((len(cat), 4 + self.S**2), dtype=np.float32)
                stars[:, 0] = cat['x'].values
                stars[:, 1] = cat['y'].values
                stars[:, 2] = cat['flux'].values
                stars[:, 3] = cat['mag'].values
                
                shapes = np.stack(cat['shape'].values)
                stars[:, 4:] = shapes
                
                all_stars.append(stars)
                star_offsets.append(star_offsets[-1] + len(stars))
                
            self.all_stars = np.concatenate(all_stars, axis=0)
            self.star_offsets = np.array(star_offsets)
            print(f"✅ Indexed {len(self.all_stars)} stars into compact memory ({(self.all_stars.nbytes / 1e6):.1f} MB)")
            
            self._STAR_INDEX_CACHE[cache_key] = {
                'mosaics': self.mosaics,
                'all_stars': self.all_stars,
                'star_offsets': self.star_offsets
            }

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        m_idx = np.random.randint(0, len(self.mosaics))
        mosaic = self.mosaics[m_idx]
        
        img_mmap = np.load(mosaic['img_path'], mmap_mode='r')
        
        full_h, full_w = img_mmap.shape
        py, px = np.random.randint(0, full_h - self.img_size), np.random.randint(0, full_w - self.img_size)
        clean_physics = img_mmap[py:py+self.img_size, px:px+self.img_size].copy()
        
        exp_time, zp, sky_mag = mosaic['exp_time'], mosaic['zp'], mosaic['sky_mag']
        scale = 0.11
        sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (scale**2) * exp_time
        
        img_with_sky = clean_physics + sky_level
        img_noisy = np.random.poisson(np.maximum(img_with_sky, 0)).astype(np.float32)
        img_noisy += np.random.normal(0, 5.0, size=img_noisy.shape)
        
        chunk_median = np.median(img_noisy)
        normalized_image = self.transform.image_to_network(img_noisy, chunk_median)
        image_tensor = torch.from_numpy(normalized_image).unsqueeze(0).float()

        s_start, s_end = self.star_offsets[m_idx], self.star_offsets[m_idx+1]
        m_stars = self.all_stars[s_start:s_end]
        
        mask = (m_stars[:, 0] >= px) & (m_stars[:, 0] < px + self.img_size) & \
               (m_stars[:, 1] >= py) & (m_stars[:, 1] < py + self.img_size)
        
        local_stars = m_stars[mask]
        if len(local_stars) == 0:
            grid_stars = torch.zeros((self.grid_size, self.grid_size, self.K, 5 + self.S**2), dtype=torch.float32)
        else:
            sort_idx = np.argsort(local_stars[:, 2])[::-1]
            local_stars = local_stars[sort_idx]
            
            lx = local_stars[:, 0] - px
            ly = local_stars[:, 1] - py
            fluxes = local_stars[:, 2]
            
            noise_variance = fluxes + self.n_pix * (sky_level + 25.0)
            snrs = fluxes / np.sqrt(noise_variance)
            comps = 1.0 / (1.0 + np.exp(-2.0 * (snrs - self.min_snr)))
            
            valid_mask = snrs >= self.min_snr
            local_stars = local_stars[valid_mask]
            lx = lx[valid_mask]
            ly = ly[valid_mask]
            fluxes = fluxes[valid_mask]
            comps = comps[valid_mask]
            
            cxs = (lx // self.cell_size).astype(int)
            cys = (ly // self.cell_size).astype(int)
            
            grid_stars_np = np.zeros((self.grid_size, self.grid_size, self.K, 5 + self.S**2), dtype=np.float32)
            counts = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
            
            for i in range(len(local_stars)):
                cx, cy = cxs[i], cys[i]
                if cx < 0 or cx >= self.grid_size or cy < 0 or cy >= self.grid_size: continue
                
                slot = counts[cy, cx]
                if slot < self.K:
                    grid_stars_np[cy, cx, slot, 0] = float(comps[i])
                    grid_stars_np[cy, cx, slot, 1] = float(lx[i] % self.cell_size)
                    grid_stars_np[cy, cx, slot, 2] = float(ly[i] % self.cell_size)
                    grid_stars_np[cy, cx, slot, 3] = float(fluxes[i])
                    grid_stars_np[cy, cx, slot, 4] = float(comps[i])
                    grid_stars_np[cy, cx, slot, 5:] = local_stars[i, 4:]
                    counts[cy, cx] += 1
            
            grid_stars = torch.from_numpy(grid_stars_np)

        bg_target_linear = sky_level - chunk_median
        bg_grid_stretched = self.transform.target_bg_to_network(np.full((self.grid_size, self.grid_size), bg_target_linear, dtype=np.float32))
        target = torch.cat([grid_stars.view(self.grid_size, self.grid_size, -1), torch.from_numpy(bg_grid_stretched).unsqueeze(-1).float()], dim=-1)
        
        return {"image": image_tensor, "target": target, "chunk_median": float(chunk_median)}
