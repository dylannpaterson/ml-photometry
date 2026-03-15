import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class GaussianStarDataset(Dataset):
    def __init__(self, num_samples=1000, min_stars=100, max_stars=1500, image_size=256, max_capacity_per_cell=5):
        """
        Generates realistic synthetic data for the Roman Bulge Time Domain Survey.
        Edge-to-Edge prediction on 256x256 image with 128x128 grid.
        """
        self.num_samples = num_samples
        self.min_stars = min_stars
        self.max_stars = max_stars
        self.img_size = image_size
        self.K = max_capacity_per_cell
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

    def generate_chunk(self):
        """Generates a 256x256 image chunk."""
        # 1. Background + Noise
        image = np.random.normal(loc=10.0, scale=2.0, size=(self.img_size, self.img_size))
        target_grid = np.zeros((self.grid_size, self.grid_size, self.K, 5), dtype=np.float32)

        num_stars = np.random.randint(self.min_stars, self.max_stars)
        fluxes = self._sample_luminosity_function(num_stars)

        true_catalogue = []

        for i in range(num_stars):
            # Coordinates anywhere in the image
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
            true_catalogue.append((true_x, true_y, flux, completeness))

            # Grid Assignment
            cell_x, cell_y = int(true_x // self.cell_size), int(true_y // self.cell_size)
            # Clip to grid bounds
            cell_x = min(cell_x, self.grid_size - 1)
            cell_y = min(cell_y, self.grid_size - 1)
            
            dx, dy = true_x % self.cell_size, true_y % self.cell_size

            for slot in range(self.K):
                if target_grid[cell_y, cell_x, slot, 0] == 0.0:
                    target_grid[cell_y, cell_x, slot] = [1.0, dx, dy, flux, completeness]
                    break

        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        target_tensor = torch.tensor(target_grid, dtype=torch.float32)

        return image_tensor, target_tensor, true_catalogue

    def visualize_chunk(self, image_tensor, true_catalogue):
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
        plt.savefig("visualization_bulge.png")
        print(f"Visualization saved to visualization_bulge.png")

if __name__ == "__main__":
    dataset = GaussianStarDataset(min_stars=500, max_stars=1500)
    image, target, catalogue = dataset.generate_chunk()
    print(f"Generated chunk with {len(catalogue)} stars.")
    dataset.visualize_chunk(image, catalogue)
