import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class GaussianPretrainingProvider(Dataset):
    def __init__(self, num_samples=1000, min_stars=100, max_stars=1500, image_size=256, max_capacity_per_cell=3, shape_size=7, use_fixed_seed=False):
        """
        Generates realistic synthetic data for the Roman Bulge Time Domain Survey.
        Edge-to-Edge prediction on 256x256 image with 64x64 grid.
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

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.use_fixed_seed:
            np.random.seed(idx)
        sparse_sample = self.generate_chunk()
        image = sparse_sample["image"]
        
        # Redensify exactly like PregeneratedDataset
        base_grid = sparse_sample["base_grid"]
        bg_map = sparse_sample["background_map"]
        
        # Target shape: [H, W, K, 6]
        grid_size = self.grid_size
        target = torch.zeros((grid_size, grid_size, self.K, 6), dtype=torch.float32)
        target[..., :-1] = base_grid
        target[..., -1] = bg_map.unsqueeze(-1)
        
        return image, target

    def _add_star_to_image(self, image, x_center, y_center, flux, sigma=1.5):
        """Adds a normalized 2D Gaussian profile to the image."""
        # Standard normalization constant so 'flux' is total volume
        norm_const = flux / (2 * np.pi * sigma**2)
        
        patch_half_size = int(round(5 * sigma)) 
        ix, iy = int(round(x_center)), int(round(y_center))
        x0, x1 = max(0, ix - patch_half_size), min(self.img_size, ix + patch_half_size + 1)
        y0, y1 = max(0, iy - patch_half_size), min(self.img_size, iy + patch_half_size + 1)
        x = np.arange(x0, x1, 1, float)
        y = np.arange(y0, y1, 1, float)
        y = y[:, np.newaxis]
        
        patch = norm_const * np.exp(-((x - x_center)**2 + (y - y_center)**2) / (2 * sigma**2))
        image[y0:y1, x0:x1] += patch

    def _sample_luminosity_function(self, n_stars, alpha=2.0, f_min=30, f_max=10000):
        """Samples fluxes from a power-law distribution."""
        u = np.random.uniform(0, 1, n_stars)
        if alpha == 1.0:
            return f_min * np.exp(u * np.log(f_max / f_min))
        else:
            return ( (f_max**(1-alpha) - f_min**(1-alpha)) * u + f_min**(1-alpha) )**(1/(1-alpha))

    def _generate_psf_shape(self, sigma=1.5):
        """Generates a normalized local PSF cutout exactly centered in the window."""
        # window centered on (0,0)
        half = self.S // 2
        x = np.arange(-half, half + 1)
        y = np.arange(-half, half + 1)
        xx, yy = np.meshgrid(x, y)
        
        # Evaluate relative to 0.0 (the center pixel)
        psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        psf /= (psf.sum() + 1e-9)
        return psf.astype(np.float32).flatten()

    def _generate_polynomial_background(self):
        """Generates a smoothly varying 2D polynomial background surface."""
        x = np.linspace(-1, 1, self.img_size)
        y = np.linspace(-1, 1, self.img_size)
        xx, yy = np.meshgrid(x, y)
        
        # 2nd order coefficients: tilt, bowl, ridge
        # Baseline bg around 10.0, variation of +/- 5.0
        c = np.random.uniform(8.0, 12.0) # Offset
        coeffs = np.random.uniform(-2.0, 2.0, 5) # x, y, x2, y2, xy
        
        bg = (coeffs[0]*xx + coeffs[1]*yy + 
              coeffs[2]*xx**2 + coeffs[3]*yy**2 + 
              coeffs[4]*xx*yy + c)
        
        # Ensure it stays positive
        return np.maximum(0.1, bg).astype(np.float32)

    def generate_chunk(self):
        """Generates a sparse chunk with a polynomial background."""
        # 1. Generate Background Surface
        gt_background = self._generate_polynomial_background()
        
        # 2. Add Poisson-like Noise (Normal approximation)
        image = np.random.normal(loc=gt_background, scale=2.0)
        
        # 3. Base grid and containers
        base_grid = np.zeros((self.grid_size, self.grid_size, self.K, 5), dtype=np.float32)
        psf_shape = self._generate_psf_shape()
        shapes = []
        indices = []

        num_stars = np.random.randint(self.min_stars, self.max_stars)
        fluxes = self._sample_luminosity_function(num_stars)

        for i in range(num_stars):
            true_x = np.random.uniform(0, self.img_size)
            true_y = np.random.uniform(0, self.img_size)
            flux = fluxes[i]

            # Use local background from surface for SNR
            local_bg = gt_background[int(true_y), int(true_x)]
            snr = flux / np.sqrt(flux + local_bg + self.read_noise**2)
            completeness = float(np.clip((snr - 3.0) / 7.0, 0.0, 1.0))

            self._add_star_to_image(image, true_x, true_y, flux)
            
            # Grid Assignment
            cell_x, cell_y = int(true_x // self.cell_size), int(true_y // self.cell_size)
            cell_x = min(cell_x, self.grid_size - 1)
            cell_y = min(cell_y, self.grid_size - 1)
            dx, dy = true_x % self.cell_size, true_y % self.cell_size

            for slot in range(self.K):
                if base_grid[cell_y, cell_x, slot, 0] == 0.0:
                    base_grid[cell_y, cell_x, slot] = [1.0, dx, dy, np.log10(flux + 1e-9), completeness]
                    shapes.append(psf_shape)
                    indices.append([cell_y, cell_x, slot])
                    break

        # Downsample ground truth background to grid size for target
        # We take the average of each cell
        bg_grid = gt_background.reshape(self.grid_size, self.cell_size, self.grid_size, self.cell_size).mean(axis=(1, 3))

        sparse_target = {
            "image": torch.tensor(image, dtype=torch.float32).unsqueeze(0),
            "base_grid": torch.tensor(base_grid, dtype=torch.float32),
            "background_map": torch.tensor(bg_grid, dtype=torch.float32), # [grid_size, grid_size]
            "shapes": torch.tensor(np.array(shapes), dtype=torch.float32) if shapes else torch.tensor([]),
            "indices": torch.tensor(np.array(indices), dtype=torch.long) if indices else torch.tensor([])
        }

        return sparse_target

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
