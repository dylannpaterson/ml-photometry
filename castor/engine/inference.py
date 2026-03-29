import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
from scipy.ndimage import zoom
from astropy.io import fits
from castor.data.transforms import AstroSpaceTransform
from castor.constants import GLOBAL_STRETCH_SCALE, N_PCA_COMPONENTS

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
        self.stretch_scale = config["data_params"].get("GLOBAL_STRETCH_SCALE", GLOBAL_STRETCH_SCALE)
        self.transform = AstroSpaceTransform(stretch_scale=self.stretch_scale)

    def predict(self, image_tensor, threshold=0.5, psf_basis=None, mean_psf=None):
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
        
        # Determine S from basis if available, else fallback
        S = 31 if psf_basis is not None else 9 
        
        for y in range(grid_h):
            for x in range(grid_w):
                for k in range(K):
                    p, dx, dy, physical_flux, c = prediction[y, x, k, :5]
                    if p > threshold:
                        predicted_stars.append(((x * cell_size) + dx, (y * cell_size) + dy, float(physical_flux), c, p))
                        
                        # Reconstruct PSF from PCA weights
                        weights = prediction[y, x, k, 5:]
                        if psf_basis is not None and mean_psf is not None:
                            # weights: [20], basis: [20, 961], mean: [961]
                            shape_flat = (weights @ psf_basis) + mean_psf
                            predicted_shapes.append(shape_flat.reshape(S, S))
                        else:
                            # Fallback or raw weight visualization
                            S_fallback = int(np.sqrt(len(weights)))
                            predicted_shapes.append(weights.reshape(S_fallback, S_fallback))
                            
        return predicted_stars, predicted_shapes, bg_map

    def visualize(self, image_tensor, true_catalogue, predicted_stars, predicted_shapes, bg_map, gt_bg_map, threshold, chunk_median=0.0, output_path="inference_comparison.png"):
        from castor.engine.evaluator import match_stars
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
            sy0, sy1 = half - (iy - y0), half + (y1 - iy)
            sx0, sx1 = half - (ix - x0), half + (x1 - ix)
            reconstruction_stars_linear[y0:y1, x0:x1] += flux * shape[sy0:sy1, sx0:sx1]

        # 2. Linear Reconstruction (Residual Space)
        full_residual_bg_linear = self.transform.network_to_bg(full_residual_bg_stretched)
        full_reconstruction_linear = reconstruction_stars_linear + full_residual_bg_linear
        
        # 3. Absolute Space Conversion (Raw Physical Photons)
        img_linear_abs = self.transform.network_to_image(img_stretched, chunk_median)
        full_reconstruction_linear_abs = full_reconstruction_linear + chunk_median
        
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

        # Statistics & Matching for Visualization
        matches, unmatched_true, unmatched_pred = match_stars(true_catalogue, predicted_stars, distance_threshold=2.0)
        
        matched_true_mags, matched_pred_mags = [], []
        for t_idx, p_idx, _ in matches:
            matched_true_mags.append(np.log10(true_catalogue[t_idx][2] + 1e-9))
            matched_pred_mags.append(np.log10(predicted_stars[p_idx][2] + 1e-9))

        all_true_mags = [np.log10(s[2] + 1e-9) for s in true_catalogue]
        all_pred_mags = [np.log10(s[2] + 1e-9) for s in predicted_stars]

        # 6. Figure Layout
        fig = plt.figure(figsize=(30, 24))
        gs = fig.add_gridspec(5, 4, hspace=0.3, wspace=0.3)
        
        def add_colorbar(im, ax):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)

        def sanitize_for_plot(data, fill_val=0.0):
            d = data.copy()
            mask = ~np.isfinite(d)
            d[mask] = fill_val
            return np.clip(d, -1e15, 1e15)

        # Row 1-2: Primary Linear Comparisons
        img_linear_abs = sanitize_for_plot(img_linear_abs, fill_val=chunk_median)
        full_reconstruction_linear_abs = sanitize_for_plot(full_reconstruction_linear_abs, fill_val=chunk_median)
        residual_linear = sanitize_for_plot(residual_linear, fill_val=0.0)

        l_vmin, l_vmax = np.percentile(img_linear_abs, [10, 99.9])
        l_vmin = max(1.0, l_vmin)
        l_vmax = max(l_vmin + 1.0, l_vmax)
        norm = LogNorm(vmin=l_vmin, vmax=l_vmax, clip=True)
        
        ax1 = fig.add_subplot(gs[0:2, 0])
        ax1.imshow(img_linear_abs, cmap='inferno', origin='lower', norm=norm, aspect='equal')
        ax1.set_title("Input (Missed Sources = Cyan)")
        
        ax2 = fig.add_subplot(gs[0:2, 1], sharex=ax1, sharey=ax1)
        im2 = ax2.imshow(full_reconstruction_linear_abs, cmap='inferno', origin='lower', norm=norm, aspect='equal')
        ax2.set_title("Model (Matched Sources = Lime)")
        
        matched_true_indices = [m[0] for m in matches]
        for i, s in enumerate(true_catalogue):
            if i in matched_true_indices:
                ax2.plot(s[0], s[1], color='lime', marker='+', linestyle='None', markersize=10, alpha=0.8)
            else:
                ax1.plot(s[0], s[1], color='cyan', marker='+', linestyle='None', markersize=10, alpha=0.8)
        
        add_colorbar(im2, ax2)
        
        ax3 = fig.add_subplot(gs[0:2, 2], sharex=ax1, sharey=ax1)
        r_limit = np.percentile(np.abs(residual_linear), 99)
        if r_limit <= 0 or not np.isfinite(r_limit): r_limit = 1.0
        im3 = ax3.imshow(residual_linear, cmap='bwr', origin='lower', vmin=-r_limit, vmax=r_limit, aspect='equal')
        ax3.set_title("Linear Residual (Missed = Black)")
        
        for i, s in enumerate(true_catalogue):
            if i not in matched_true_indices:
                ax3.plot(s[0], s[1], 'k+', markersize=10, alpha=0.8)
        
        add_colorbar(im3, ax3)

        # Row 3: Background Comparisons (Linear)
        full_bg_abs = sanitize_for_plot(full_bg_abs, fill_val=chunk_median)
        full_gt_bg_abs = sanitize_for_plot(full_gt_bg_abs, fill_val=chunk_median)
        bg_vmin = min(full_bg_abs.min(), full_gt_bg_abs.min())
        bg_vmax = max(full_bg_abs.max(), full_gt_bg_abs.max())
        if bg_vmax <= bg_vmin: bg_vmax = bg_vmin + 1.0
        
        ax4 = fig.add_subplot(gs[2, 0], sharex=ax1, sharey=ax1)
        ax4.imshow(full_bg_abs, cmap='viridis', origin='lower', vmin=bg_vmin, vmax=bg_vmax, aspect='equal')
        ax4.set_title("Predicted Background (Linear)")
        
        ax5 = fig.add_subplot(gs[2, 1], sharex=ax1, sharey=ax1)
        im5 = ax5.imshow(full_gt_bg_abs, cmap='viridis', origin='lower', vmin=bg_vmin, vmax=bg_vmax, aspect='equal')
        ax5.set_title("Truth Background (Linear)")
        add_colorbar(im5, ax5)

        # Row 4-5: PSF & Mag Plots & Missed Stats
        if matched_true_mags:
            ax8 = fig.add_subplot(gs[3:, 0])
            m_true_clean = [m for m in matched_true_mags if np.isfinite(m)]
            m_pred_clean = [m for m in matched_pred_mags if np.isfinite(m)]
            if m_true_clean and m_pred_clean:
                ax8.scatter(matched_true_mags, matched_pred_mags, alpha=0.5, s=10)
                all_plot_mags = m_true_clean + m_pred_clean
                mmin, mmax = min(all_plot_mags), max(all_plot_mags)
                ax8.plot([mmin, mmax], [mmin, mmax], 'r--', alpha=0.8)
                ax8.set_xlabel("True log10(Flux)")
                ax8.set_ylabel("Predicted log10(Flux)")
                ax8.set_title("Magnitude Recovery Accuracy (Matched Pairs)")
                ax8.set_aspect('equal')
                ax8.grid(True, alpha=0.3)

        if all_true_mags:
            ax_hist = fig.add_subplot(gs[3, 1])
            true_m_clean = [m for m in all_true_mags if np.isfinite(m)]
            pred_m_clean = [m for m in all_pred_mags if np.isfinite(m)]
            
            if true_m_clean:
                mmin_h, mmax_h = min(true_m_clean), max(true_m_clean)
                if pred_m_clean:
                    mmin_h = min(mmin_h, min(pred_m_clean))
                    mmax_h = max(mmax_h, max(pred_m_clean))
                
                bins = np.linspace(mmin_h, mmax_h, 30)
                ax_hist.hist(true_m_clean, bins=bins, alpha=0.3, label='True (Full)', color='gray')
                if pred_m_clean:
                    ax_hist.hist(pred_m_clean, bins=bins, alpha=0.5, label='Predicted (Full)', color='C0', histtype='step', linewidth=2)
                ax_hist.set_xlabel("log10(Flux)")
                ax_hist.set_ylabel("Count")
                ax_hist.set_title("Luminosity Function Comparison (All Stars)")
                ax_hist.legend()
                ax_hist.grid(True, alpha=0.2)

        ax9 = fig.add_subplot(gs[4, 1])
        missed_comps = [true_catalogue[i][3] for i in range(len(true_catalogue)) if i not in matched_true_indices]
        matched_comps = [true_catalogue[i][3] for i in matched_true_indices]
        
        if missed_comps or matched_comps:
            mc_clean = [c for c in matched_comps if np.isfinite(c)]
            misc_clean = [c for c in missed_comps if np.isfinite(c)]
            if mc_clean or misc_clean:
                ax9.hist([mc_clean, misc_clean], bins=20, stacked=True, 
                        label=['Detected', 'Missed'], color=['g', 'r'], alpha=0.7)
                ax9.set_xlabel("Target Completeness Score (SNR Proxy)")
                ax9.set_ylabel("Star Count")
                ax9.set_title("Detection Success vs. Target Completeness")
                ax9.legend()
                ax9.grid(True, alpha=0.2)

        # PSF Profile Plots
        if predicted_shapes:
            ax_psf_x = fig.add_subplot(gs[3:, 2])
            ax_psf_y = fig.add_subplot(gs[3:, 3])
            
            shapes_clean = [s for s in predicted_shapes if np.all(np.isfinite(s))]
            if shapes_clean:
                num_to_plot = min(100, len(shapes_clean))
                for i in range(num_to_plot):
                    shape = shapes_clean[i]
                    prof_x = np.mean(shape, axis=0)
                    prof_y = np.mean(shape, axis=1)
                    ax_psf_x.plot(prof_x, color='C0', alpha=0.1, linewidth=1)
                    ax_psf_y.plot(prof_y, color='C1', alpha=0.1, linewidth=1)
                
                all_shapes = np.stack(shapes_clean[:100])
                ax_psf_x.plot(np.mean(all_shapes, axis=(0, 1)), color='black', linewidth=2, label='Mean')
                ax_psf_y.plot(np.mean(all_shapes, axis=(0, 2)), color='black', linewidth=2, label='Mean')
                
                ax_psf_x.set_title("PSF X-Profiles (Y-avg)")
                ax_psf_y.set_title("PSF Y-Profiles (X-avg)")
                ax_psf_x.set_xlabel("Pixels"); ax_psf_y.set_xlabel("Pixels")
                ax_psf_x.grid(True, alpha=0.2); ax_psf_y.grid(True, alpha=0.2)

        plt.suptitle(f"Generative Diagnostic (Scale={self.stretch_scale}) | Predicted Stars: {len(predicted_stars)}", fontsize=24)
        plt.savefig(output_path); print(f"Comparison saved to {output_path}")
