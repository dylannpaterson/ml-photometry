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

# --- 1. The Dynamic Bulge LF Sampler ---
def sample_bulge_magnitudes(n_stars, rc_mag, rc_sigma, rc_fraction, m_min=10.0, m_max=26.0):
    """Samples apparent magnitudes (m). Smaller m = brighter star."""
    n_rc = int(n_stars * rc_fraction)
    n_bg = n_stars - n_rc

    # 1. Background (Power law in linear space becomes linear in magnitude space)
    u = np.random.uniform(0, 1, n_bg)
    m_bg = np.interp(u, [0.0, 0.95, 1.0], [m_max, 18.0, m_min])

    # 2. Red Clump (Gaussian in magnitude space)
    m_rc = np.random.normal(loc=rc_mag, scale=rc_sigma, size=n_rc)

    # 3. Combine and clip
    m_all = np.concatenate([m_bg, m_rc])
    m_all = np.clip(m_all, m_min, m_max)

    return m_all 

class GaussianPretrainingProvider(Dataset):
    def __init__(self, num_samples=1000, min_stars=100, max_stars=1500, image_size=256, 
                 max_capacity_per_cell=MAX_CAPACITY_PER_CELL, shape_size=SHAPE_SIZE, 
                 use_fixed_seed=False, global_stretch_scale=GLOBAL_STRETCH_SCALE, min_snr=5.0):
        """
        Generates realistic synthetic data for the Roman Bulge Time Domain Survey.
        Vectorized for speed.
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

        # Grid parameters: DEFAULT_CELL_SIZE x DEFAULT_CELL_SIZE cells
        self.cell_size = DEFAULT_CELL_SIZE
        self.grid_size = self.img_size // self.cell_size
        
        # Pre-allocate coordinate grids for vectorization
        self.xx, self.yy = np.meshgrid(np.arange(self.img_size), np.arange(self.img_size))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.use_fixed_seed:
            np.random.seed(idx)
        sparse_sample = self.generate_chunk()
        image = sparse_sample["image"]
        
        # Redensify exactly like PregeneratedDataset (including shapes)
        base_grid = sparse_sample["base_grid"]
        bg_map = sparse_sample["background_map"]
        shapes = sparse_sample["shapes"]
        indices = sparse_sample["indices"]
        
        # Target shape: [H, W, (K * (5 + S^2)) + 1]
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

    def _generate_psf_kernel(self, sigma=1.5):
        """Generates a normalized local PSF kernel for convolution."""
        half = 15 # Large enough kernel to avoid edge artifacts
        x = np.arange(-half, half + 1)
        y = np.arange(-half, half + 1)
        xx, yy = np.meshgrid(x, y)
        psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        psf /= (psf.sum() + 1e-9)
        return psf.astype(np.float32)

    def generate_chunk(self, rc_params=None, exp_params=None):
        """Generates a realistic Roman-like chunk using the Speed-Hack rendering engine."""
        # 1. Setup Dynamic Bulge & Instrument Parameters
        if rc_params is None:
            rc_loc = np.random.uniform(14.5, 16.5)
            rc_scale = np.random.uniform(0.2, 0.5)
            rc_fraction = np.random.uniform(0.05, 0.20)
        else:
            rc_loc, rc_scale, rc_fraction = rc_params

        if exp_params is None:
            exp_time = np.random.uniform(20.0, 100.0)
            zp = 26.5
            sky_mag = 22.0
        else:
            exp_time, zp, sky_mag = exp_params

        # Physical counts for sky background
        pixel_scale = 0.11 # Roman
        sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (pixel_scale**2) * exp_time
        
        num_stars = int(np.random.randint(self.min_stars, self.max_stars + 1))
        sigma = 1.5
        
        # 2. Sample Magnitudes & Convert to Flux
        mags = sample_bulge_magnitudes(num_stars, rc_loc, rc_scale, rc_fraction)
        fluxes = exp_time * (10 ** (-0.4 * (mags - zp)))
        
        # Sort bright to faint
        sort_idx = np.argsort(fluxes)[::-1]
        fluxes = fluxes[sort_idx]
        mags = mags[sort_idx]
        
        x_centers = np.random.uniform(0, self.img_size, num_stars)
        y_centers = np.random.uniform(0, self.img_size, num_stars)

        # 3. Two-Tier Rendering Speed Hack
        monster_cutoff = int(num_stars * 0.02)
        
        # Tier A: The Crowd (FFT Convolution)
        crowd_flux_map = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        cx, cy, cf = x_centers[monster_cutoff:], y_centers[monster_cutoff:], fluxes[monster_cutoff:]
        x0, y0 = np.floor(cx).astype(int), np.floor(cy).astype(int)
        dx, dy = cx - x0, cy - y0
        mask = (x0 >= 0) & (x0 < self.img_size - 1) & (y0 >= 0) & (y0 < self.img_size - 1)
        x0, y0, dx, dy, cf = x0[mask], y0[mask], dx[mask], dy[mask], cf[mask]
        
        np.add.at(crowd_flux_map, (y0, x0), cf * (1-dx) * (1-dy))
        np.add.at(crowd_flux_map, (y0, x0+1), cf * dx * (1-dy))
        np.add.at(crowd_flux_map, (y0+1, x0), cf * (1-dx) * dy)
        np.add.at(crowd_flux_map, (y0+1, x0+1), cf * dx * dy)
        
        # Tier B: The Stellar Sea (Unresolved mottled background)
        n_unresolved = 50000 # Dense sea
        sea_mags = sample_bulge_magnitudes(n_unresolved, rc_loc, rc_scale, rc_fraction, m_min=26.0, m_max=32.0)
        sea_fluxes = exp_time * (10 ** (-0.4 * (sea_mags - zp)))
        ux = np.random.uniform(0, self.img_size-1, n_unresolved)
        uy = np.random.uniform(0, self.img_size-1, n_unresolved)
        ux0, uy0 = np.floor(ux).astype(int), np.floor(uy).astype(int)
        umask = (ux0 >= 0) & (ux0 < self.img_size-1) & (uy0 >= 0) & (uy0 < self.img_size-1)
        np.add.at(crowd_flux_map, (uy0[umask], ux0[umask]), sea_fluxes[umask])

        psf_kernel = self._generate_psf_kernel(sigma=sigma)
        star_signal = fftconvolve(crowd_flux_map, psf_kernel, mode='same')

        # Tier C: Monsters (Individual renders)
        for i in range(monster_cutoff):
            fx, fy, flux = x_centers[i], y_centers[i], fluxes[i]
            half = 10
            ix, iy = int(fx), int(fy)
            y0, y1 = max(0, iy-half), min(self.img_size, iy+half+1)
            x0, x1 = max(0, ix-half), min(self.img_size, ix+half+1)
            
            yy, xx = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing='ij')
            stamp = np.exp(-((xx - fx)**2 + (yy - fy)**2) / (2 * sigma**2))
            stamp *= (flux / (stamp.sum() + 1e-9))
            star_signal[y0:y1, x0:x1] += stamp

        # 4. Noise & Background
        gt_background = np.full((self.img_size, self.img_size), sky_level, dtype=np.float32)
        total_photon_flux = gt_background + star_signal
        noise_std = np.sqrt(total_photon_flux + self.read_noise**2)
        raw_image = np.random.normal(loc=total_photon_flux, scale=noise_std).astype(np.float32)

        chunk_median = np.median(raw_image)
        normalized_image = self.transform.image_to_network(raw_image, chunk_median)

        # 5. Build Target Grid (JIT)
        base_grid = np.zeros((self.grid_size, self.grid_size, self.K, 5), dtype=np.float32)
        
        half = self.S // 2
        grid = np.arange(-half, half + 1)
        sy, sx = np.meshgrid(grid, grid, indexing='ij')
        psf_shape = np.exp(-(sx**2 + sy**2) / (2 * sigma**2))
        psf_shape /= (psf_shape.sum() + 1e-9)
        psf_shape_flat = psf_shape.astype(np.float32).flatten()
        
        shapes, indices = [], []
        
        cell_assignments = {}
        for i in range(num_stars):
            tx, ty, flux = x_centers[i], y_centers[i], fluxes[i]
            cx, cy = int(tx // self.cell_size), int(ty // self.cell_size)
            if cx < 0 or cx >= self.grid_size or cy < 0 or cy >= self.grid_size: continue
            
            snr = flux / np.sqrt(flux + sky_level + self.read_noise**2)
            if snr >= self.min_snr:
                if (cy, cx) not in cell_assignments: cell_assignments[(cy, cx)] = []
                cell_assignments[(cy, cx)].append([flux, tx, ty, snr, mags[i]])

        catalog_stars = []
        for (cy, cx), cell_stars in cell_assignments.items():
            sorted_stars = sorted(cell_stars, key=lambda x: x[0], reverse=True)
            for slot in range(min(self.K, len(sorted_stars))):
                flux, tx, ty, snr, mag = sorted_stars[slot]
                completeness = 1.0 / (1.0 + np.exp(-2.0 * (snr - self.min_snr)))
                
                base_grid[cy, cx, slot] = [1.0, tx % self.cell_size, ty % self.cell_size, flux, completeness]
                shapes.append(psf_shape_flat)
                indices.append([cy, cx, slot])
                
                catalog_stars.append({
                    'x': tx, 'y': ty, 'flux': flux, 'mag': mag, 'shape': psf_shape_flat
                })

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
            "sky_level": sky_level,
            "catalog": pd.DataFrame(catalog_stars),
            "exp_time": exp_time,
            "zp": zp,
            "sky_mag": sky_mag
        }

class GaussianMosaicDataset(Dataset):
    def __init__(self, data_dir, num_samples=25000, image_size=256, cell_size=DEFAULT_CELL_SIZE, global_stretch_scale=GLOBAL_STRETCH_SCALE):
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.img_size = image_size
        self.cell_size = cell_size
        self.grid_size = image_size // cell_size
        self.transform = AstroSpaceTransform(stretch_scale=global_stretch_scale)
        self.K = MAX_CAPACITY_PER_CELL
        self.S = SHAPE_SIZE
        
        self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith("_img.npy")])
        self.catalog_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".parquet")])
        
        if not self.image_files:
            raise FileNotFoundError(f"No mosaics found in {data_dir}.")
            
        print(f"🔗 Stage 0 Macro-Sparse: Memory-mapping {len(self.image_files)} physics mosaics...")
        self.mosaics = []
        for img_f, cat_f in zip(self.image_files, self.catalog_files):
            img_mmap = np.load(os.path.join(data_dir, img_f), mmap_mode='r')
            catalog = pd.read_parquet(os.path.join(data_dir, cat_f))
            self.mosaics.append({'image': img_mmap, 'catalog': catalog})

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        m_idx = np.random.randint(0, len(self.mosaics))
        mosaic = self.mosaics[m_idx]
        
        full_h, full_w = mosaic['image'].shape
        py = np.random.randint(0, full_h - self.img_size)
        px = np.random.randint(0, full_w - self.img_size)
        
        clean_physics = mosaic['image'][py:py+self.img_size, px:px+self.img_size].copy()
        
        # Metadata-driven sky injection
        cat_full = mosaic['catalog']
        exp_time = cat_full['exp_time'].iloc[0] if 'exp_time' in cat_full.columns else 54.0
        zp = cat_full['zp'].iloc[0] if 'zp' in cat_full.columns else 26.5
        sky_mag = cat_full['sky_mag'].iloc[0] if 'sky_mag' in cat_full.columns else 22.0
        scale = 0.11
        sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (scale**2) * exp_time
        
        img_with_sky = clean_physics + sky_level
        img_noisy = np.random.poisson(np.maximum(img_with_sky, 0)).astype(np.float32)
        img_noisy += np.random.normal(0, 5.0, size=img_noisy.shape)
        
        chunk_median = np.median(img_noisy)
        normalized_image = self.transform.image_to_network(img_noisy, chunk_median)
        image_tensor = torch.from_numpy(normalized_image).unsqueeze(0).float()

        # JIT Target Grid Construction
        mask = (cat_full['x'] >= px) & (cat_full['x'] < px + self.img_size) & \
               (cat_full['y'] >= py) & (cat_full['y'] < py + self.img_size)
        local_stars = cat_full[mask].copy()
        local_stars['lx'] = local_stars['x'] - px
        local_stars['ly'] = local_stars['y'] - py
        
        grid_stars = torch.zeros((self.grid_size, self.grid_size, self.K, 5 + self.S**2), dtype=torch.float32)
        cell_assignments = {}
        for _, star in local_stars.iterrows():
            cx, cy = int(star['lx'] // self.cell_size), int(star['ly'] // self.cell_size)
            if (cy, cx) not in cell_assignments: cell_assignments[(cy, cx)] = []
            cell_assignments[(cy, cx)].append(star)
            
        for (cy, cx), stars in cell_assignments.items():
            if cy >= self.grid_size or cx >= self.grid_size: continue
            sorted_stars = sorted(stars, key=lambda s: s['flux'], reverse=True)
            for slot in range(min(self.K, len(sorted_stars))):
                star = sorted_stars[slot]
                snr = star['flux'] / np.sqrt(star['flux'] + sky_level + 25.0)
                comp = 1.0 / (1.0 + np.exp(-2.0 * (snr - 5.0)))
                
                grid_stars[cy, cx, slot, 0] = 1.0
                grid_stars[cy, cx, slot, 1] = star['lx'] % self.cell_size
                grid_stars[cy, cx, slot, 2] = star['ly'] % self.cell_size
                grid_stars[cy, cx, slot, 3] = star['flux']
                grid_stars[cy, cx, slot, 4] = comp
                grid_stars[cy, cx, slot, 5:] = torch.from_numpy(star['shape'])

        bg_target_linear = sky_level - chunk_median
        bg_grid_stretched = self.transform.target_bg_to_network(np.full((self.grid_size, self.grid_size), bg_target_linear))
        
        target = torch.cat([grid_stars.view(self.grid_size, self.grid_size, -1), torch.from_numpy(bg_grid_stretched).unsqueeze(-1)], dim=-1)
        
        return {
            "image": image_tensor,
            "target": target,
            "chunk_median": float(chunk_median)
        }
