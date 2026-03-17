import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class GaussianPretrainingProvider(Dataset):
    def __init__(self, num_samples=1000, min_stars=100, max_stars=1500, image_size=256, max_capacity_per_cell=3, shape_size=7):
        """
        Generates realistic synthetic data for the Roman Bulge Time Domain Survey.
        Edge-to-Edge prediction on 256x256 image with 128x128 grid.
        """
        self.num_samples = num_samples
        self.min_stars = min_stars
        self.max_stars = max_stars
        self.img_size = image_size
        self.K = max_capacity_per_cell
        self.S = shape_size
        self.read_noise = 5.0

        # Grid parameters: 2x2 cells for 256x256 image = 128x128 grid
        self.cell_size = 2
        self.grid_size = self.img_size // self.cell_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image_tensor, target_tensor, _ = self.generate_chunk()
        return image_tensor, target_tensor

    def _add_star_to_image(self, image, x_center, y_center, flux, sigma=1.5):
        """Adds a 2D Gaussian profile to the image."""
        patch_half_size = int(round(5 * sigma)) 
        ix, iy = int(round(x_center)), int(round(y_center))
        x0, x1 = max(0, ix - patch_half_size), min(self.img_size, ix + patch_half_size + 1)
        y0, y1 = max(0, iy - patch_half_size), min(self.img_size, iy + patch_half_size + 1)
        x = np.arange(x0, x1, 1, float)
        y = np.arange(y0, y1, 1, float)
        y = y[:, np.newaxis]
        patch = flux * np.exp(-((x - x_center)**2 + (y - y_center)**2) / (2 * sigma**2))
        image[y0:y1, x0:x1] += patch

    def _sample_luminosity_function(self, n_stars, alpha=2.0, f_min=30, f_max=1000):
        """Samples fluxes from a power-law distribution."""
        u = np.random.uniform(0, 1, n_stars)
        if alpha == 1.0:
            return f_min * np.exp(u * np.log(f_max / f_min))
        else:
            return ( (f_max**(1-alpha) - f_min**(1-alpha)) * u + f_min**(1-alpha) )**(1/(1-alpha))

    def _generate_psf_7x7(self, x_center, y_center, sigma=1.5):
        """Generates a normalized 7x7 local PSF cutout exactly centered in the window."""
        # 7x7 grid centered on (0,0) relative to the star's integer pixel
        half = self.S // 2
        # Creating a grid from -3 to 3
        x = np.arange(-half, half + 1)
        y = np.arange(-half, half + 1)
        xx, yy = np.meshgrid(x, y)
        
        # Evaluate relative to 0.0 (the center pixel) instead of the sub-pixel center
        psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        psf /= (psf.sum() + 1e-9)
        return psf.astype(np.float32).flatten()

    def generate_chunk(self):
        """Generates a sparse 256x256 image chunk and target data."""
        image = np.random.normal(loc=10.0, scale=2.0, size=(self.img_size, self.img_size))
        
        # Base grid: [128, 128, K, 5] (p, dx, dy, m, c)
        base_grid = np.zeros((self.grid_size, self.grid_size, self.K, 5), dtype=np.float32)
        
        # Sparse shapes list: [[shape_49], ...] and their indices [[y, x, slot], ...]
        shapes = []
        indices = []

        num_stars = np.random.randint(self.min_stars, self.max_stars)
        fluxes = self._sample_luminosity_function(num_stars)

        for i in range(num_stars):
            true_x = np.random.uniform(0, self.img_size)
            true_y = np.random.uniform(0, self.img_size)
            flux = fluxes[i]

            # SNR / Completeness
            ix, iy = int(true_x), int(true_y)
            y_start, y_end = max(0, iy-1), min(self.img_size, iy+2)
            x_start, x_end = max(0, ix-1), min(self.img_size, ix+2)
            local_bg = np.sum(image[y_start:y_end, x_start:x_end])

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
                    base_grid[cell_y, cell_x, slot] = [1.0, dx, dy, flux, completeness]
                    
                    # Store sparse shape and its mapping
                    psf_shape = self._generate_psf_7x7(true_x, true_y)
                    shapes.append(psf_shape)
                    indices.append([cell_y, cell_x, slot])
                    break

        # Package as sparse dictionary for efficient storage
        sparse_target = {
            "image": torch.tensor(image, dtype=torch.float32).unsqueeze(0),
            "base_grid": torch.tensor(base_grid, dtype=torch.float32),
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
