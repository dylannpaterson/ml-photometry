import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
from scipy.ndimage import zoom
from astropy.io import fits
from src.data.transforms import AstroSpaceTransform

def upsample_background(bg_map, target_size):
    """
    Upsamples a grid-based background map to full image resolution.
    Uses bilinear interpolation with correct physical centering (cell centers).
    """
    from scipy.interpolate import RegularGridInterpolator
    h, w = bg_map.shape
    H, W = target_size
    cell_size = H // h
    
    # Grid coordinates (cell centers)
    x = np.arange(w) * cell_size + (cell_size - 1) / 2.0
    y = np.arange(h) * cell_size + (cell_size - 1) / 2.0
    
    interp = RegularGridInterpolator((y, x), bg_map, method='linear', bounds_error=False, fill_value=None)
    
    # Target coordinates (all pixels)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    return interp((yy, xx))

class InferenceEngine:
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.img_size = config["data_params"]["image_size"]
        self.stretch_scale = config["data_params"].get("GLOBAL_STRETCH_SCALE", 10.0)
        self.transform = AstroSpaceTransform(stretch_scale=self.stretch_scale)

    def predict(self, image_tensor, threshold=0.5):
        """Runs inference on a single 2D image tensor [1, H, W]."""
        self.model.eval()
        with torch.no_grad():
            input_tensor = image_tensor.unsqueeze(0).to(self.device)
            prediction_dict = self.model(input_tensor)
            prediction = prediction_dict["stars"].squeeze(0).cpu().numpy()
            bg_map = prediction_dict["background"].squeeze(0).cpu().numpy()
            
        predicted_stars, predicted_shapes = [], []
        grid_h, grid_w, K, _ = prediction.shape
        cell_size = self.img_size // grid_h
        
        for y in range(grid_h):
            for x in range(grid_w):
                for k in range(K):
                    p, dx, dy, m_network, c = prediction[y, x, k, :5]
                    if p > threshold:
                        # FIX: m_network is the network's estimate of the STRETCHED flux.
                        # We must map it back to physical photons.
                        linear_flux = self.transform.network_to_flux(m_network)
                        predicted_stars.append(((x * cell_size) + dx, (y * cell_size) + dy, float(linear_flux), c, p))
                        shape_vector = prediction[y, x, k, 5:]
                        S = int(np.sqrt(len(shape_vector)))
                        predicted_shapes.append(shape_vector.reshape(S, S))
        return predicted_stars, predicted_shapes, bg_map

    def visualize(self, image_tensor, true_catalogue, predicted_stars, predicted_shapes, bg_map, gt_bg_map, threshold, chunk_median=0.0, output_path="inference_comparison.png"):
        from src.engine.evaluator import match_stars
        img_stretched = image_tensor.squeeze().numpy()
        H, W = img_stretched.shape
        
        # 1. Component Preparation (Network Space: Stretched)
        full_residual_bg_stretched = upsample_background(bg_map.squeeze(), (H, W))
        full_gt_residual_bg_stretched = upsample_background(gt_bg_map.squeeze(), (H, W))
        
        reconstruction_stars_linear = np.zeros_like(img_stretched)
        for (x, y, flux, c, p), shape in zip(predicted_stars, predicted_shapes):
            ix, iy, S = int(round(x)), int(round(y)), shape.shape[0]
            half = S // 2
            y0, y1 = max(0, iy - half), min(H, iy + half + 1)
            x0, x1 = max(0, ix - half), min(W, ix + half + 1)
            sy0, sy1, sx0, sx1 = half - (iy - y0), half + (y1 - iy), half - (ix - x0), half + (x1 - ix)
            reconstruction_stars_linear[y0:y1, x0:x1] += flux * shape[sy0:sy1, sx0:sx1]

        # 2. Linear Reconstruction (Residual Space)
        full_residual_bg_linear = self.transform.network_to_bg(full_residual_bg_stretched)
        full_reconstruction_linear = reconstruction_stars_linear + full_residual_bg_linear
        
        # 3. Map back to Stretched Space for units-matching comparison
        full_reconstruction_stretched = self.transform.target_bg_to_network(full_reconstruction_linear)
        residual_stretched = img_stretched - full_reconstruction_stretched

        # 4. Absolute Space Conversion (Raw Physical Photons)
        img_linear_abs = self.transform.network_to_image(img_stretched, chunk_median)
        full_reconstruction_linear_abs = full_reconstruction_linear + chunk_median
        full_bg_abs = full_residual_bg_linear + chunk_median
        full_gt_bg_abs = self.transform.network_to_bg(full_gt_residual_bg_stretched) + chunk_median

        # --- FITS OUTPUT ---
        hdul = fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(img_stretched, name="INPUT_STRETCHED"),
            fits.ImageHDU(full_reconstruction_stretched, name="MODEL_STRETCHED"),
            fits.ImageHDU(residual_stretched, name="RESIDUAL_STRETCHED"),
            fits.ImageHDU(img_linear_abs, name="INPUT_LINEAR_ABS"),
            fits.ImageHDU(full_reconstruction_linear_abs, name="MODEL_LINEAR_ABS"),
            fits.ImageHDU(full_bg_abs, name="BG_PRED_ABS"),
            fits.ImageHDU(full_gt_bg_abs, name="BG_TRUE_ABS")
        ])
        fits_path = output_path.replace(".png", ".fits")
        hdul.writeto(fits_path, overwrite=True)
        print(f"FITS data saved to {fits_path}")

        # Statistics
        matches, _, _ = match_stars(true_catalogue, predicted_stars)
        true_mags, pred_mags = [], []
        for t_idx, p_idx, _ in matches:
            true_mags.append(np.log10(true_catalogue[t_idx][2] + 1e-9))
            pred_mags.append(np.log10(predicted_stars[p_idx][2] + 1e-9))

        # 6. Figure Layout
        fig = plt.figure(figsize=(30, 24))
        gs = fig.add_gridspec(5, 4)
        
        # Row 1-2: Stretched Comparisons (Network Space)
        vmin, vmax = np.percentile(img_stretched, [1, 99.9])
        
        ax1 = fig.add_subplot(gs[0:2, 0])
        ax1.imshow(img_stretched, cmap='inferno', origin='lower', vmin=vmin, vmax=vmax)
        ax1.set_title("Input (Stretched)")
        for s in true_catalogue: ax1.plot(s[0], s[1], 'g+', markersize=8, alpha=0.4)
        
        ax2 = fig.add_subplot(gs[0:2, 1])
        ax2.imshow(full_reconstruction_stretched, cmap='inferno', origin='lower', vmin=vmin, vmax=vmax)
        ax2.set_title("Model (Stretched)")
        
        ax3 = fig.add_subplot(gs[0:2, 2])
        rmax = max(0.1, np.percentile(np.abs(residual_stretched), 99))
        im3 = ax3.imshow(residual_stretched, cmap='bwr', origin='lower', vmin=-rmax, vmax=rmax)
        ax3.set_title("Residual (Stretched)")
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # Row 3: Absolute Linear Comparisons (Physical Space)
        ax4 = fig.add_subplot(gs[2, 0])
        l_vmin, l_vmax = np.percentile(img_linear_abs, [10, 99.9])
        ax4.imshow(img_linear_abs, cmap='inferno', origin='lower', norm=LogNorm(vmin=max(1.0, l_vmin), vmax=l_vmax))
        ax4.set_title("Input (Absolute Linear)")
        
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.imshow(full_reconstruction_linear_abs, cmap='inferno', origin='lower', norm=LogNorm(vmin=max(1.0, l_vmin), vmax=l_vmax))
        ax5.set_title("Model (Absolute Linear)")

        # Background Side-by-Side (Residuals in Stretched Space)
        bg_vmin = min(full_residual_bg_stretched.min(), full_gt_residual_bg_stretched.min())
        bg_vmax = max(full_residual_bg_stretched.max(), full_gt_residual_bg_stretched.max())
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.imshow(full_residual_bg_stretched, cmap='viridis', origin='lower', vmin=bg_vmin, vmax=bg_vmax)
        ax6.set_title("Pred Residual BG (Stretched)")
        ax7 = fig.add_subplot(gs[2, 3])
        im7 = ax7.imshow(full_gt_residual_bg_stretched, cmap='viridis', origin='lower', vmin=bg_vmin, vmax=bg_vmax)
        ax7.set_title("Truth Residual BG (Stretched)")
        plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)

        # Row 4-5: PSF & Mag Plots
        if true_mags:
            ax8 = fig.add_subplot(gs[3, 0])
            ax8.scatter(true_mags, pred_mags, alpha=0.5)
            ax8.plot([min(true_mags), max(true_mags)], [min(true_mags), max(true_mags)], 'r--')
            ax8.set_title("Mag Recovery"); ax8.set_aspect('equal')

        plt.suptitle(f"Generative Diagnostic (Scale={self.stretch_scale}) | Predicted Stars: {len(predicted_stars)}", fontsize=24)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_path); print(f"Comparison saved to {output_path}")
