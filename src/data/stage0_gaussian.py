import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os

class GaussianPretrainingProvider(Dataset):
    def __init__(self, num_samples=1000, min_stars=100, max_stars=1500, image_size=256, max_capacity_per_cell=3, shape_size=7, use_fixed_seed=False):
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
        
        # Target shape: [H, W, K, 5 + S^2 + 1] -> 87 channels for S=9, K=3
        S2 = self.S * self.S
        grid_size = self.grid_size
        target = torch.zeros((grid_size, grid_size, self.K, 5 + S2 + 1), dtype=torch.float32)
        
        # Fill base metadata (p, dx, dy, m, c)
        target[..., :5] = base_grid
        
        # Fill shapes at correct indices
        if len(indices) > 0:
            for i in range(len(indices)):
                y, x, k = indices[i]
                target[y, x, k, 5:5+S2] = shapes[i]
        
        # Fill background in last channel
        target[..., -1] = bg_map.unsqueeze(-1)
        
        return image, target

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

    def generate_chunk(self):
        """Generates a sparse chunk with a realistic Bulge background (Vectorized)."""
        # 1. Generate Realistic Background Surface
        # Galactic Bulge has high background from unresolved stars and zodiacal light
        x_lin = np.linspace(-1, 1, self.img_size)
        y_lin = np.linspace(-1, 1, self.img_size)
        xx_bg, yy_bg = np.meshgrid(x_lin, y_lin)
        
        # Mean background level approx 100.0 e- (typical for Bulge F146)
        c = np.random.uniform(80.0, 120.0)
        coeffs = np.random.uniform(-10.0, 10.0, 5)
        base_bg = np.maximum(10.0, (coeffs[0]*xx_bg + coeffs[1]*yy_bg + 
                                    coeffs[2]*xx_bg**2 + coeffs[3]*yy_bg**2 + 
                                    coeffs[4]*xx_bg*yy_bg + c))
        
        # Add high-frequency "unresolved star" noise (correlated noise)
        unresolved_noise = np.random.gamma(shape=2.0, scale=5.0, size=base_bg.shape)
        gt_background = (base_bg + unresolved_noise).astype(np.float32)
        
        # 2. Base image with Poisson-like noise
        # Variance = Mean + ReadNoise^2
        noise_std = np.sqrt(gt_background + self.read_noise**2)
        image = np.random.normal(loc=gt_background, scale=noise_std).astype(np.float32)
        
        # 3. Add detectable stars (Vectorized)
        num_stars = np.random.randint(self.min_stars, self.max_stars)
        fluxes = self._sample_luminosity_function(num_stars)
        x_centers = np.random.uniform(0, self.img_size, num_stars)
        y_centers = np.random.uniform(0, self.img_size, num_stars)
        
        sigma = 1.5
        patch_size = int(round(5 * sigma))
        for i in range(num_stars):
            ix, iy = int(round(x_centers[i])), int(round(y_centers[i]))
            x0, x1 = max(0, ix - patch_size), min(self.img_size, ix + patch_size + 1)
            y0, y1 = max(0, iy - patch_size), min(self.img_size, iy + patch_size + 1)
            x = np.arange(x0, x1)
            y = np.arange(y0, y1)[:, np.newaxis]
            patch = (fluxes[i] / (2 * np.pi * sigma**2)) * np.exp(-((x - x_centers[i])**2 + (y - y_centers[i])**2) / (2 * sigma**2))
            image[y0:y1, x0:x1] += patch

        # 4. Grid Assignment
        base_grid = np.zeros((self.grid_size, self.grid_size, self.K, 5), dtype=np.float32)
        psf_shape = self._generate_psf_shape(sigma)
        shapes = []
        indices = []

        for i in range(num_stars):
            px, py = int(np.clip(x_centers[i], 0, self.img_size-1)), int(np.clip(y_centers[i], 0, self.img_size-1))
            local_bg = gt_background[py, px]
            snr = fluxes[i] / np.sqrt(fluxes[i] + local_bg + self.read_noise**2)
            completeness = float(np.clip((snr - 3.0) / 7.0, 0.0, 1.0))
            cell_x, cell_y = int(x_centers[i] // self.cell_size), int(y_centers[i] // self.cell_size)
            cell_x, cell_y = min(cell_x, self.grid_size - 1), min(cell_y, self.grid_size - 1)
            for slot in range(self.K):
                if base_grid[cell_y, cell_x, slot, 0] == 0.0:
                    base_grid[cell_y, cell_x, slot] = [1.0, x_centers[i] % self.cell_size, y_centers[i] % self.cell_size, np.log10(fluxes[i] + 1e-9), completeness]
                    shapes.append(psf_shape)
                    indices.append([cell_y, cell_x, slot])
                    break

        bg_grid = gt_background.reshape(self.grid_size, self.cell_size, self.grid_size, self.cell_size).mean(axis=(1, 3))

        return {
            "image": torch.from_numpy(image).unsqueeze(0),
            "base_grid": torch.from_numpy(base_grid),
            "background_map": torch.from_numpy(bg_grid),
            "shapes": torch.from_numpy(np.array(shapes)) if shapes else torch.tensor([]),
            "indices": torch.from_numpy(np.array(indices)) if indices else torch.tensor([])
        }

    def visualize_chunk(self, image_tensor, true_catalogue, output_path="visualization_bulge.png"):
        from matplotlib.colors import LogNorm
        img = image_tensor.squeeze().numpy()
        img_min = img.min()
        if img_min <= 0: img = img - img_min + 1e-3
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(img, cmap='inferno', origin='lower', norm=LogNorm())
        fig.colorbar(im, ax=ax, label='Flux')
        sample_size = min(200, len(true_catalogue))
        indices = np.random.choice(len(true_catalogue), sample_size, replace=False)
        for idx in indices:
            x, y, flux, comp = true_catalogue[idx]
            ax.plot(x, y, 'g+', markersize=5, alpha=comp)
        ax.set_title(f"Synthetic Chunk (256x256): {len(true_catalogue)} stars")
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")

class GaussianMosaicDataset(Dataset):
    def __init__(self, data_dir, num_samples=25000, image_size=256, cell_size=4):
        """Loads large mosaics into memory and samples random chunks."""
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.img_size = image_size
        self.cell_size = cell_size
        self.grid_size = image_size // cell_size
        
        self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith("_img.pt")])
        self.target_files = sorted([f for f in os.listdir(data_dir) if f.endswith("_tgt.pt")])
        
        if not self.image_files:
            raise FileNotFoundError(f"No mosaic files found in {data_dir}. Run scripts/generate_mosaics.py first.")
            
        print(f"📦 Loading {len(self.image_files)} SCA Mosaics into RAM...")
        self.mosaics = []
        for img_f, tgt_f in zip(self.image_files, self.target_files):
            img = torch.load(os.path.join(data_dir, img_f), map_location='cpu')
            tgt = torch.load(os.path.join(data_dir, tgt_f), map_location='cpu')
            self.mosaics.append((img, tgt))
        print(f"✅ Ready to sample {num_samples} chunks.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        mosaic_idx = np.random.randint(0, len(self.mosaics))
        full_img, full_tgt = self.mosaics[mosaic_idx]
        
        max_y_grid = full_tgt.shape[0] - self.grid_size
        max_x_grid = full_tgt.shape[1] - self.grid_size
        
        gy = np.random.randint(0, max_y_grid)
        gx = np.random.randint(0, max_x_grid)
        
        py, px = gy * self.cell_size, gx * self.cell_size
        
        img_chunk = full_img[py : py + self.img_size, px : px + self.img_size].unsqueeze(0).float()
        tgt_chunk = full_tgt[gy : gy + self.grid_size, gx : gx + self.grid_size].float()
        
        return img_chunk, tgt_chunk
