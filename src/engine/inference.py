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
                    p, dx, dy, physical_flux, c = prediction[y, x, k, :5]
                    if p > threshold:
                        # NEW: The model now outputs raw physical photons directly!
                        predicted_stars.append(((x * cell_size) + dx, (y * cell_size) + dy, float(physical_flux), c, p))
                        shape_vector = prediction[y, x, k, 5:]
                        S = int(np.sqrt(len(shape_vector)))
                        predicted_shapes.append(shape_vector.reshape(S, S))
        return predicted_stars, predicted_shapes, bg_map

    def visualize(self, image_tensor, true_catalogue, predicted_stars, predicted_shapes, bg_map, gt_bg_map, threshold, chunk_median=0.0, output_path="inference_comparison.png"):
        from src.engine.evaluator import match_stars
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
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
        
        # 3. Absolute Space Conversion (Raw Physical Photons)
        img_linear_abs = self.transform.network_to_image(img_stretched, chunk_median)
        full_reconstruction_linear_abs = full_reconstruction_linear + chunk_median
        
        # NEW: Linear Residual
        residual_linear = img_linear_abs - full_reconstruction_linear_abs
        
        full_bg_abs = full_residual_bg_linear + chunk_median
        full_gt_bg_abs = self.transform.network_to_bg(full_gt_residual_bg_stretched) + chunk_median

        # --- FITS OUTPUT ---
        hdul = fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(img_linear_abs, name="INPUT_LINEAR"),
            fits.ImageHDU(full_reconstruction_linear_abs, name="MODEL_LINEAR"),
            fits.ImageHDU(residual_linear, name="RESIDUAL_LINEAR"),
            fits.ImageHDU(full_bg_abs, name="BG_PRED_LINEAR"),
            fits.ImageHDU(full_gt_bg_abs, name="BG_TRUE_LINEAR")
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
        gs = fig.add_gridspec(5, 4, hspace=0.3, wspace=0.3)
        
        def add_colorbar(im, ax):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)

        # Row 1-2: Primary Linear Comparisons
        l_vmin, l_vmax = np.percentile(img_linear_abs, [10, 99.9])
        norm = LogNorm(vmin=max(1.0, l_vmin), vmax=l_vmax)
        
        ax1 = fig.add_subplot(gs[0:2, 0])
        ax1.imshow(img_linear_abs, cmap='inferno', origin='lower', norm=norm, aspect='equal')
        ax1.set_title("Input (Linear Photons)")
        for s in true_catalogue: ax1.plot(s[0], s[1], 'g+', markersize=8, alpha=0.4)
        
        ax2 = fig.add_subplot(gs[0:2, 1], sharex=ax1, sharey=ax1)
        im2 = ax2.imshow(full_reconstruction_linear_abs, cmap='inferno', origin='lower', norm=norm, aspect='equal')
        ax2.set_title("Model (Linear Photons)")
        add_colorbar(im2, ax2)
        
        ax3 = fig.add_subplot(gs[0:2, 2], sharex=ax1, sharey=ax1)
        # Linear residual typically has wide range, center on 0 with symlog or robust limits
        r_limit = np.percentile(np.abs(residual_linear), 99)
        im3 = ax3.imshow(residual_linear, cmap='bwr', origin='lower', vmin=-r_limit, vmax=r_limit, aspect='equal')
        ax3.set_title("Linear Residual (Data - Model)")
        add_colorbar(im3, ax3)

        # Row 3: Background Comparisons (Linear)
        bg_vmin = min(full_bg_abs.min(), full_gt_bg_abs.min())
        bg_vmax = max(full_bg_abs.max(), full_gt_bg_abs.max())
        
        ax4 = fig.add_subplot(gs[2, 0], sharex=ax1, sharey=ax1)
        ax4.imshow(full_bg_abs, cmap='viridis', origin='lower', vmin=bg_vmin, vmax=bg_vmax, aspect='equal')
        ax4.set_title("Predicted Background (Linear)")
        
        ax5 = fig.add_subplot(gs[2, 1], sharex=ax1, sharey=ax1)
        im5 = ax5.imshow(full_gt_bg_abs, cmap='viridis', origin='lower', vmin=bg_vmin, vmax=bg_vmax, aspect='equal')
        ax5.set_title("Truth Background (Linear)")
        add_colorbar(im5, ax5)

        # Row 4-5: PSF & Mag Plots
        if true_mags:
            ax8 = fig.add_subplot(gs[3:, 0:2])
            ax8.scatter(true_mags, pred_mags, alpha=0.5, s=10)
            all_mags = true_mags + pred_mags
            mmin, mmax = min(all_mags), max(all_mags)
            ax8.plot([mmin, mmax], [mmin, mmax], 'r--', alpha=0.8)
            ax8.set_xlabel("True log10(Flux)")
            ax8.set_ylabel("Predicted log10(Flux)")
            ax8.set_title("Magnitude Recovery Accuracy")
            ax8.set_aspect('equal')
            ax8.grid(True, alpha=0.3)

        # PSF Profile Plots
        if predicted_shapes:
            ax_psf_x = fig.add_subplot(gs[3:, 2])
            ax_psf_y = fig.add_subplot(gs[3:, 3])
            
            num_to_plot = min(100, len(predicted_shapes))
            for i in range(num_to_plot):
                shape = predicted_shapes[i]
                prof_x = np.mean(shape, axis=0)
                prof_y = np.mean(shape, axis=1)
                ax_psf_x.plot(prof_x, color='C0', alpha=0.1, linewidth=1)
                ax_psf_y.plot(prof_y, color='C1', alpha=0.1, linewidth=1)
            
            all_shapes = np.stack(predicted_shapes[:100])
            ax_psf_x.plot(np.mean(all_shapes, axis=(0, 1)), color='black', linewidth=2, label='Mean')
            ax_psf_y.plot(np.mean(all_shapes, axis=(0, 2)), color='black', linewidth=2, label='Mean')
            
            ax_psf_x.set_title("PSF X-Profiles (Y-avg)")
            ax_psf_y.set_title("PSF Y-Profiles (X-avg)")
            ax_psf_x.set_xlabel("Pixels"); ax_psf_y.set_xlabel("Pixels")
            ax_psf_x.grid(True, alpha=0.2); ax_psf_y.grid(True, alpha=0.2)

        plt.suptitle(f"Generative Diagnostic (Scale={self.stretch_scale}) | Predicted Stars: {len(predicted_stars)}", fontsize=24)
        plt.savefig(output_path); print(f"Comparison saved to {output_path}")
