import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
from src.data.transforms import AstroSpaceTransform

class GaussianPretrainingProvider(Dataset):
    def __init__(self, num_samples=1000, min_stars=100, max_stars=1500, image_size=256, max_capacity_per_cell=3, shape_size=7, use_fixed_seed=False, global_stretch_scale=10.0):
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
        self.transform = AstroSpaceTransform(stretch_scale=global_stretch_scale)

        # Grid parameters: 4x4 cells for 256x256 image = 64x64 grid
        self.cell_size = 4
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
        
        # Target shape: [H, W, (K * (5 + S^2)) + 1] -> 259 channels for S=9, K=3
        S2 = self.S * self.S
        grid_size = self.grid_size
        
        # 1. Build K-slot star targets
        star_targets = torch.zeros((grid_size, grid_size, self.K, 5 + S2), dtype=torch.float32)
        
        # Fill base metadata (p, dx, dy, m, c)
        star_targets[..., :5] = base_grid
        
        # Fill shapes at correct indices
        if len(indices) > 0:
            for i in range(len(indices)):
                y, x, k = indices[i]
                star_targets[y, x, k, 5:5+S2] = shapes[i]
        
        # 2. Flatten K dimension and Append Background
        flattened_stars = star_targets.view(grid_size, grid_size, -1)
        target = torch.cat([flattened_stars, bg_map.unsqueeze(-1)], dim=-1)
        
        return {
            "image": image,
            "target": target,
            "chunk_median": sparse_sample["chunk_median"]
        }

    def _sample_luminosity_function(self, n_stars, alpha=2.0, f_min=30, f_max=10000):
        """Samples fluxes from a power-law distribution."""
        u = np.random.uniform(0, 1, n_stars)
        if alpha == 1.0:
            return f_min * np.exp(u * np.log(f_max / f_min))
        else:
            return ( (f_max**(1-alpha) - f_min**(1-alpha)) * u + f_min**(1-alpha) )**(1/(1-alpha))

    def _generate_psf_shape(self, sigma=1.5):
        """Generates a normalized local PSF cutout exactly centered in the window."""
        half = self.S // 2
        x = np.arange(-half, half + 1)
        y = np.arange(-half, half + 1)
        xx, yy = np.meshgrid(x, y)
        psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        psf /= (psf.sum() + 1e-9)
        return psf.astype(np.float32).flatten()

    def _get_local_density(self, l, b):
        """Calculates expected detectable stars by integrating Coleman model."""
        try:
            from coleman_bulge_density import bulge_density_model
            r_vals = np.linspace(4.0, 12.0, 50)
            dr_pc = (r_vals[1] - r_vals[0]) * 1000.0
            l_array = np.full_like(r_vals, l)
            b_array = np.full_like(r_vals, b)
            density_3d, _ = bulge_density_model(r=r_vals, lat=b_array, lon=l_array)
            patch_deg2 = (256 * 0.11 / 3600.0)**2
            solid_angle_sr = patch_deg2 * (np.pi / 180.0)**2
            volume_elements = (r_vals * 1000.0)**2 * dr_pc * solid_angle_sr
            total_stars = np.sum(np.array(density_3d) * volume_elements)
            return np.clip(total_stars * 0.05, 100, 5000)
        except:
            scale_l, scale_b = 5.0, 2.0
            relative_density = np.exp(-np.sqrt((l/scale_l)**2 + (b/scale_b)**2))
            return 100 + 900 * relative_density

    def generate_chunk(self):
        """Generates a realistic Roman-like chunk using AstroSpaceTransform."""
        center_l = np.random.uniform(-5.0, 5.0)
        center_b = np.random.uniform(-3.0, 3.0)
        expected_stars = self._get_local_density(center_l, center_b)
        num_stars = int(np.random.poisson(expected_stars))

        zodiacal_level = np.random.uniform(20.0, 40.0)
        smooth_bg = np.full((self.img_size, self.img_size), zodiacal_level, dtype=np.float32)
        
        sigma = 1.5
        u_patch = 3
        star_patch = int(round(5 * sigma))
        
        unresolved_img = np.zeros_like(smooth_bg)
        num_tail = num_stars * 2 
        u_tail = np.random.uniform(0, 1, num_tail)
        f_min_t, f_max_t = 1.0, 10.0
        tail_fluxes = ((f_max_t**(1-2.0) - f_min_t**(1-2.0)) * u_tail + f_min_t**(1-2.0))**(1/(1-2.0))
        tail_x = np.random.uniform(0, self.img_size, num_tail)
        tail_y = np.random.uniform(0, self.img_size, num_tail)
        
        for i in range(num_tail):
            ix, iy = int(round(tail_x[i])), int(round(tail_y[i]))
            x0, x1 = max(0, ix - u_patch), min(self.img_size, ix + u_patch + 1)
            y0, y1 = max(0, iy - u_patch), min(self.img_size, iy + u_patch + 1)
            xx = np.arange(x0, x1)
            yy = np.arange(y0, y1)[:, np.newaxis]
            patch = (tail_fluxes[i] / (2 * np.pi * sigma**2)) * np.exp(-((xx - tail_x[i])**2 + (yy - tail_y[i])**2) / (2 * sigma**2))
            unresolved_img[y0:y1, x0:x1] += patch

        noise_floor = np.sqrt(smooth_bg + unresolved_img + self.read_noise**2)

        fluxes = self._sample_luminosity_function(num_stars * 2) 
        x_centers = np.random.uniform(0, self.img_size, num_stars * 2)
        y_centers = np.random.uniform(0, self.img_size, num_stars * 2)
        
        base_grid = np.zeros((self.grid_size, self.grid_size, self.K, 5), dtype=np.float32)
        star_signal = np.zeros_like(smooth_bg)
        psf_shape = self._generate_psf_shape(sigma=1.5)
        shapes, indices = [], []
        
        cell_assignments = {}
        for i in range(len(fluxes)):
            px, py = int(np.clip(x_centers[i], 0, self.img_size-1)), int(np.clip(y_centers[i], 0, self.img_size-1))
            local_noise = noise_floor[py, px]
            snr = fluxes[i] / (local_noise * 4.0)
            
            if snr >= 2.0:
                cx, cy = int(x_centers[i] // self.cell_size), int(y_centers[i] // self.cell_size)
                cx, cy = min(cx, self.grid_size - 1), min(cy, self.grid_size - 1)
                if (cy, cx) not in cell_assignments: cell_assignments[(cy, cx)] = []
                completeness = float(np.clip((snr - 3.5) / 4.5, 0.0, 1.0))
                cell_assignments[(cy, cx)].append([fluxes[i], x_centers[i], y_centers[i], snr, completeness])

        for (cy, cx), cell_stars in cell_assignments.items():
            sorted_stars = sorted(cell_stars, key=lambda x: x[0], reverse=True)
            for slot in range(min(self.K, len(sorted_stars))):
                flux, tx, ty, snr, tcomp = sorted_stars[slot]
                
                # CENTRALIZED: Convert linear flux to network target
                m_target = self.transform.target_flux_to_network(flux)
                
                base_grid[cy, cx, slot] = [1.0, tx % self.cell_size, ty % self.cell_size, m_target, tcomp]
                shapes.append(psf_shape)
                indices.append([cy, cx, slot])
                ix, iy = int(np.clip(tx, 0, self.img_size-1)), int(np.clip(ty, 0, self.img_size-1))
                x0, x1 = max(0, ix - star_patch), min(self.img_size, ix + star_patch + 1)
                y0, y1 = max(0, iy - star_patch), min(self.img_size, iy + star_patch + 1)
                xx = np.arange(x0, x1)
                yy = np.arange(y0, y1)[:, np.newaxis]
                patch = (flux * (2 * np.pi * sigma**2)**-1) * np.exp(-((xx - tx)**2 + (yy - ty)**2) / (2 * sigma**2))
                star_signal[y0:y1, x0:x1] += patch
            
            if len(sorted_stars) > self.K:
                for i in range(self.K, len(sorted_stars)):
                    flux, tx, ty, _, _ = sorted_stars[i]
                    ix, iy = int(np.clip(tx, 0, self.img_size-1)), int(np.clip(ty, 0, self.img_size-1))
                    x0, x1 = max(0, ix - u_patch), min(self.img_size, ix + u_patch + 1)
                    y0, y1 = max(0, iy - u_patch), min(self.img_size, iy + u_patch + 1)
                    xx = np.arange(x0, x1)
                    yy = np.arange(y0, y1)[:, np.newaxis]
                    patch = (flux / (2 * np.pi * sigma**2)) * np.exp(-((xx - tx)**2 + (yy - ty)**2) / (2 * sigma**2))
                    unresolved_img[y0:y1, x0:x1] += patch

        gt_background = (smooth_bg + unresolved_img).astype(np.float32)
        total_photon_flux = gt_background + star_signal
        noise_std = np.sqrt(total_photon_flux + self.read_noise**2)
        raw_image = np.random.normal(loc=total_photon_flux, scale=noise_std).astype(np.float32)

        chunk_median = np.median(raw_image)
        # CENTRALIZED: Linear image -> Network Space
        normalized_image = self.transform.image_to_network(raw_image, chunk_median)

        # CENTRALIZED: Linear BG residual -> Network Space
        residual_bg_linear = gt_background - chunk_median
        residual_bg_stretched = self.transform.target_bg_to_network(residual_bg_linear)
        
        bg_grid = residual_bg_stretched.reshape(self.grid_size, self.cell_size, self.grid_size, self.cell_size).mean(axis=(1, 3))

        return {
            "image": torch.from_numpy(normalized_image).unsqueeze(0),
            "raw_image": torch.from_numpy(raw_image),
            "base_grid": torch.from_numpy(base_grid),
            "background_map": torch.from_numpy(bg_grid),
            "shapes": torch.from_numpy(np.array(shapes)) if shapes else torch.tensor([]),
            "indices": torch.from_numpy(np.array(indices)) if indices else torch.tensor([]),
            "chunk_median": float(chunk_median)
        }

    def visualize_chunk(self, image_tensor, true_catalogue, output_path="visualization_bulge.png"):
        img = image_tensor.squeeze().numpy()
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(img, cmap='inferno', origin='lower')
        fig.colorbar(im, ax=ax, label='Arcsinh Intensity')
        sample_size = min(200, len(true_catalogue))
        indices = np.random.choice(len(true_catalogue), sample_size, replace=False)
        for idx in indices:
            x, y, flux, comp = true_catalogue[idx]
            ax.plot(x, y, 'g+', markersize=5, alpha=comp)
        plt.savefig(output_path)

class GaussianMosaicDataset(Dataset):
    def __init__(self, data_dir, num_samples=25000, image_size=256, cell_size=4, global_stretch_scale=10.0):
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.img_size = image_size
        self.cell_size = cell_size
        self.grid_size = image_size // cell_size
        self.transform = AstroSpaceTransform(stretch_scale=global_stretch_scale)
        
        self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith("_img.npy")])
        self.target_files = sorted([f for f in os.listdir(data_dir) if f.endswith("_target.npy")])
        
        if not self.image_files:
            raise FileNotFoundError(f"No mosaics found in {data_dir}. Run scripts/generate_mosaics.py first.")
            
        print(f"🔗 Memory-mapping {len(self.image_files)} SCA Mosaics (Dual-mmap)...")
        self.mosaics = []
        for img_f, tgt_f in zip(self.image_files, self.target_files):
            img_mmap = np.load(os.path.join(data_dir, img_f), mmap_mode='r')
            tgt_mmap = np.load(os.path.join(data_dir, tgt_f), mmap_mode='r')
            self.mosaics.append((img_mmap, tgt_mmap))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        mosaic_idx = np.random.randint(0, len(self.mosaics))
        full_img, full_tgt = self.mosaics[mosaic_idx]
        
        full_grid_size = full_img.shape[0] 
        max_grid = (full_grid_size // self.cell_size) - self.grid_size
        
        gy = np.random.randint(0, max_grid)
        gx = np.random.randint(0, max_grid)
        py, px = gy * self.cell_size, gx * self.cell_size
        
        raw_image = full_img[py:py+self.img_size, px:px+self.img_size]
        target_raw = full_tgt[gy:gy+self.grid_size, gx:gx+self.grid_size]
        
        chunk_median = np.median(raw_image)
        # CENTRALIZED: Linear -> Network
        normalized_image = self.transform.image_to_network(raw_image, chunk_median)
        image_tensor = torch.from_numpy(normalized_image).unsqueeze(0).float()

        if len(target_raw.shape) == 4:
            star_grid = target_raw[..., :-1]
            bg_map = target_raw[..., 0, -1] 
            # CENTRALIZED: Linear -> Network
            bg_residual_stretched = self.transform.target_bg_to_network(bg_map - chunk_median)
            flattened_stars = star_grid.reshape(self.grid_size, self.grid_size, -1)
            target = torch.cat([torch.from_numpy(flattened_stars), torch.from_numpy(bg_residual_stretched).unsqueeze(-1)], dim=-1)
        else:
            target = torch.from_numpy(target_raw.copy())
            bg_linear_abs = target[..., -1]
            # CENTRALIZED: Linear -> Network
            bg_residual_stretched = self.transform.target_bg_to_network(bg_linear_abs - chunk_median)
            target[..., -1] = bg_residual_stretched
            
        return {
            "image": image_tensor,
            "target": target,
            "chunk_median": float(chunk_median)
        }
