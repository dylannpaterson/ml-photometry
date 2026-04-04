import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
from scipy.ndimage import zoom
from astropy.io import fits
from castor.data.transforms import AstroSpaceTransform
from castor.constants import GLOBAL_STRETCH_SCALE, N_PCA_COMPONENTS
from scipy.signal import fftconvolve

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

    def predict(self, image_tensor, threshold=0.1, psf_basis=None, mean_psf=None):
        """Runs inference on a single 2D image tensor [H, W]."""
        self.model.eval()
        with torch.no_grad():
            # 1. FIX: Apply Median Subtraction and Arcsinh Stretch (Pre-processing)
            # This ensures raw linear data is correctly scaled for the network
            chunk_median = image_tensor.median().item()
            stretched_tensor = torch.arcsinh((image_tensor - chunk_median) / self.stretch_scale)
            
            # 2. FIX: Ensure [Batch, Channel, H, W] dimensionality
            if stretched_tensor.dim() == 2:
                input_tensor = stretched_tensor.unsqueeze(0).unsqueeze(0)
            elif stretched_tensor.dim() == 3:
                input_tensor = stretched_tensor.unsqueeze(0)
            else:
                input_tensor = stretched_tensor
                
            input_tensor = input_tensor.to(self.device)

            # 3. FIX: Match Training Mixed Precision Context
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                prediction_dict = self.model(input_tensor)
            
            # Note: Unpack predictions and convert to float32 for stable CPU processing
            prediction = prediction_dict["stars"].squeeze(0).float().cpu().numpy()
            bg_map = prediction_dict["background"].squeeze(0).float().cpu().numpy()
            
        predicted_stars, predicted_shapes = [], []
        grid_h, grid_w, K, _ = prediction.shape
        cell_size = self.img_size // grid_h
        
        # Determine S from basis if available, else fallback
        if psf_basis is not None:
            S = int(psf_basis.shape[1]**0.5)
        else:
            S = 9
        
        for y in range(grid_h):
            for x in range(grid_w):
                for k in range(K):
                    p, dx, dy, physical_flux = prediction[y, x, k, :4]
                    if p > threshold:
                        predicted_stars.append(((x * cell_size) + dx, (y * cell_size) + dy, float(physical_flux), p))
                        
                        # Reconstruct PSF from PCA weights
                        weights = prediction[y, x, k, 4:]
                        if psf_basis is not None and mean_psf is not None:
                            # weights: [20], basis: [20, 961], mean: [961]
                            shape_flat = (weights @ psf_basis) + mean_psf
                            predicted_shapes.append(shape_flat.reshape(S, S))
                        else:
                            # FIX: Safe fallback - Create a simple 9x9 Gaussian instead of crashing on weights.reshape
                            sy, sx = np.meshgrid(np.arange(9)-4, np.arange(9)-4)
                            fallback_psf = np.exp(-(sx**2 + sy**2) / (2 * 1.5**2))
                            fallback_psf /= fallback_psf.sum()
                            predicted_shapes.append(fallback_psf)
                            
        return predicted_stars, predicted_shapes, bg_map

    def visualize(self, image_tensor, true_catalogue, predicted_stars, predicted_shapes, bg_map, gt_bg_map, threshold, chunk_median=0.0, jitter_params=None, output_path="inference_comparison.png"):
        """
        Visualizes inference results.
        jitter_params: (s_jit, q_jit, theta_jit) to match input image smear.
        """
        from castor.engine.evaluator import match_stars
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        img_stretched = image_tensor.squeeze().numpy()
        H, W = img_stretched.shape
        
        # 1. Component Preparation (Network Space: Stretched)
        full_residual_bg_stretched = upsample_background(bg_map.squeeze(), (H, W))
        full_gt_residual_bg_stretched = upsample_background(gt_bg_map.squeeze(), (H, W))
        
        # --- DEFINITION: DETECTED = p >= 0.5 ---
        detected_stars = []
        detected_shapes = []
        for s, shp in zip(predicted_stars, predicted_shapes):
            if s[3] >= 0.5:
                detected_stars.append(s)
                detected_shapes.append(shp)

        # --- SUB-PIXEL ACCURATE RECONSTRUCTION ---
        reconstruction_stars_linear = np.zeros_like(img_stretched)
        for (x, y, flux, p), shape in zip(detected_stars, detected_shapes):
            S = shape.shape[0]
            half = S // 2
            x0, y0 = int(np.floor(x)), int(np.floor(y))
            dx, dy = x - x0, y - y0
            w00, w10, w01, w11 = (1.0-dx)*(1.0-dy), dx*(1.0-dy), (1.0-dx)*dy, dx*dy
            
            for j, i, w in [(0, 0, w00), (1, 0, w10), (0, 1, w01), (1, 1, w11)]:
                if w <= 0: continue
                iy, ix = y0 + i, x0 + j
                y_min, y_max = max(0, iy-half), min(H, iy+half+1)
                x_min, x_max = max(0, ix-half), min(W, ix+half+1)
                sy0, sy1 = half - (iy-y_min), half + (y_max-iy)
                sx0, sx1 = half - (ix-x_min), half + (x_max-ix)
                if sy1 > sy0 and sx1 > sx0:
                    reconstruction_stars_linear[y_min:y_max, x_min:x_max] += (flux * w) * shape[sy0:sy1, sx0:sx1]

        # --- APPLY GLOBAL JITTER TO RECONSTRUCTION ---
        if jitter_params is not None:
            s_j, q_j, t_j = jitter_params
            kh = S // 2
            gy, gx = np.meshgrid(np.arange(S) - kh, np.arange(S) - kh, indexing='ij')
            cos, sin = np.cos(t_j), np.sin(t_j)
            gxp, gyp = gx * cos + gy * sin, -gx * sin + gy * cos
            j_kernel = np.exp(-(gxp**2 / (2 * s_j**2) + gyp**2 / (2 * (s_j * q_j)**2)))
            j_kernel /= (j_kernel.sum() + 1e-9)
            reconstruction_stars_linear = fftconvolve(reconstruction_stars_linear, j_kernel, mode='same')

        # 2. Linear Reconstruction (Residual Space)
        full_residual_bg_linear = self.transform.network_to_bg(full_residual_bg_stretched)
        full_reconstruction_linear = reconstruction_stars_linear + full_residual_bg_linear
        
        # 3. Absolute Space Conversion (Raw Physical Photons)
        img_linear_abs = self.transform.network_to_image(img_stretched, chunk_median)
        full_reconstruction_linear_abs = full_reconstruction_linear + chunk_median
        residual_linear = img_linear_abs - full_reconstruction_linear_abs
        full_bg_abs = full_residual_bg_linear + chunk_median
        full_gt_bg_abs = self.transform.network_to_bg(full_gt_residual_bg_stretched) + chunk_median

        # FITS Output
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

        # Matching
        match_true = [(s[1], s[2], s[3]) for s in true_catalogue]
        match_pred = [(s[0], s[1], s[2]) for s in detected_stars]
        matches, unmatched_true, unmatched_pred = match_stars(match_true, match_pred, distance_threshold=2.0)
        
        matched_true_mags = [np.log10(true_catalogue[m[0]][3] + 1e-9) for m in matches]
        matched_pred_mags = [np.log10(detected_stars[m[1]][2] + 1e-9) for m in matches]
        all_true_mags = [np.log10(s[3] + 1e-9) for s in true_catalogue]
        
        # Extract missed true mags
        matched_true_indices = [m[0] for m in matches]
        missed_true_mags = [np.log10(true_catalogue[i][3] + 1e-9) for i in range(len(true_catalogue)) if i not in matched_true_indices]

        # Figure Layout
        fig = plt.figure(figsize=(30, 24))
        gs = fig.add_gridspec(5, 4, hspace=0.3, wspace=0.3)
        
        def add_colorbar(im, ax):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)

        def sanitize_for_plot(data, fill_val=0.0):
            d = data.copy()
            mask = ~np.isfinite(d); d[mask] = fill_val
            return np.clip(d, -1e15, 1e15)

        img_linear_abs = sanitize_for_plot(img_linear_abs, fill_val=chunk_median)
        full_reconstruction_linear_abs = sanitize_for_plot(full_reconstruction_linear_abs, fill_val=chunk_median)
        residual_linear = sanitize_for_plot(residual_linear, fill_val=0.0)

        l_vmin, l_vmax = np.percentile(img_linear_abs, [10, 99.9])
        l_vmin, l_vmax = max(1.0, l_vmin), max(l_vmin + 1.0, l_vmax)
        norm = LogNorm(vmin=l_vmin, vmax=l_vmax, clip=True)
        
        ax1 = fig.add_subplot(gs[0:2, 0])
        ax1.imshow(img_linear_abs, cmap='inferno', origin='lower', norm=norm, aspect='equal')
        ax1.set_title("Input (Missed Sources = Cyan)")
        
        ax2 = fig.add_subplot(gs[0:2, 1], sharex=ax1, sharey=ax1)
        im2 = ax2.imshow(full_reconstruction_linear_abs, cmap='inferno', origin='lower', norm=norm, aspect='equal')
        ax2.set_title("Model (Matched Sources = Lime)")
        
        visible_true_indices = [i for i, s in enumerate(true_catalogue) if s[0] >= 0.1]
        for i in visible_true_indices:
            s = true_catalogue[i]
            if i in matched_true_indices:
                ax2.plot(s[1], s[2], color='lime', marker='+', linestyle='None', markersize=10, alpha=0.8)
            else:
                ax1.plot(s[1], s[2], color='cyan', marker='+', linestyle='None', markersize=10, alpha=0.8)
        
        pred_x, pred_y = [s[0] for s in detected_stars], [s[1] for s in detected_stars]
        ax2.scatter(pred_x, pred_y, color='red', s=1, alpha=0.5, label='Detected')
        add_colorbar(im2, ax2)
        
        ax3 = fig.add_subplot(gs[0:2, 2], sharex=ax1, sharey=ax1)
        r_limit = np.percentile(np.abs(residual_linear), 99)
        if r_limit <= 0 or not np.isfinite(r_limit): r_limit = 1.0
        im3 = ax3.imshow(residual_linear, cmap='bwr', origin='lower', vmin=-r_limit, vmax=r_limit, aspect='equal')
        ax3.set_title("Linear Residual (Missed = Black)")
        for i in visible_true_indices:
            if i not in matched_true_indices:
                s = true_catalogue[i]; ax3.plot(s[1], s[2], 'k+', markersize=10, alpha=0.8)
        add_colorbar(im3, ax3)

        # Background Row
        full_bg_abs, full_gt_bg_abs = sanitize_for_plot(full_bg_abs, fill_val=chunk_median), sanitize_for_plot(full_gt_bg_abs, fill_val=chunk_median)
        bg_vmin, bg_vmax = min(full_bg_abs.min(), full_gt_bg_abs.min()), max(full_bg_abs.max(), full_gt_bg_abs.max())
        if bg_vmax <= bg_vmin: bg_vmax = bg_vmin + 1.0
        ax4 = fig.add_subplot(gs[2, 0], sharex=ax1, sharey=ax1)
        ax4.imshow(full_bg_abs, cmap='viridis', origin='lower', vmin=bg_vmin, vmax=bg_vmax, aspect='equal')
        ax4.set_title("Predicted Background (Linear)")
        ax5 = fig.add_subplot(gs[2, 1], sharex=ax1, sharey=ax1)
        im5 = ax5.imshow(full_gt_bg_abs, cmap='viridis', origin='lower', vmin=bg_vmin, vmax=bg_vmax, aspect='equal')
        ax5.set_title("Truth Background (Linear)")
        add_colorbar(im5, ax5)

        # Plots Row
        if matched_true_mags:
            ax8 = fig.add_subplot(gs[3:, 0])
            ax8.scatter(matched_true_mags, matched_pred_mags, alpha=0.5, s=10)
            mmin, mmax = min(matched_true_mags+matched_pred_mags), max(matched_true_mags+matched_pred_mags)
            ax8.plot([mmin, mmax], [mmin, mmax], 'r--', alpha=0.8)
            ax8.set_xlabel("True log10(Flux)"); ax8.set_ylabel("Predicted log10(Flux)")
            ax8.set_title("Magnitude Recovery Accuracy"); ax8.set_aspect('equal'); ax8.grid(True, alpha=0.3)

        if all_true_mags:
            ax_hist = fig.add_subplot(gs[3, 1])
            m_p90 = [np.log10(s[2] + 1e-9) for s in predicted_stars if s[3] >= 0.9]
            m_p50 = [np.log10(s[2] + 1e-9) for s in predicted_stars if s[3] >= 0.5]
            m_p10 = [np.log10(s[2] + 1e-9) for s in predicted_stars if s[3] >= 0.1]
            mmin_h, mmax_h = min(all_true_mags+m_p10), max(all_true_mags+m_p10)
            bins = np.linspace(mmin_h, mmax_h, 30)
            ax_hist.hist(all_true_mags, bins=bins, alpha=0.2, label='Truth', color='black')
            ax_hist.hist(m_p10, bins=bins, alpha=0.4, label='p >= 0.1', histtype='step', linestyle=':')
            ax_hist.hist(m_p50, bins=bins, alpha=0.7, label='p >= 0.5', histtype='step', linestyle='--')
            ax_hist.hist(m_p90, bins=bins, alpha=1.0, label='p >= 0.9', histtype='step')
            ax_hist.set_xlabel("log10(Flux)"); ax_hist.set_title("LF vs. Confidence"); ax_hist.legend(); ax_hist.grid(True, alpha=0.2)

        # NEW: Error Rates (FP/FN %) vs. Signal Strength
        if true_catalogue and detected_stars:
            ax_err = fig.add_subplot(gs[4, 0])
            matched_pred_indices = [m[1] for m in matches]
            matched_true_indices = [m[0] for m in matches]
            
            # 1. Define Bins for Objectness/Confidence (0.0 to 1.0)
            err_bins = np.linspace(0.0, 1.0, 21)
            bin_centers = (err_bins[:-1] + err_bins[1:]) / 2.0
            
            # 2. Calculate False Negative Rate (relative to Truth Objectness)
            true_obj = np.array([s[0] for s in true_catalogue])
            fn_rates = []
            for j in range(len(err_bins)-1):
                in_bin = (true_obj >= err_bins[j]) & (true_obj < err_bins[j+1])
                if in_bin.any():
                    total_in_bin = in_bin.sum()
                    missed_in_bin = sum(1 for i in np.where(in_bin)[0] if i not in matched_true_indices)
                    fn_rates.append(100.0 * missed_in_bin / total_in_bin)
                else:
                    fn_rates.append(0.0)
            
            # 3. Calculate False Positive Rate (relative to Predicted Confidence)
            pred_conf = np.array([s[3] for s in detected_stars])
            fp_rates = []
            for j in range(len(err_bins)-1):
                in_bin = (pred_conf >= err_bins[j]) & (pred_conf < err_bins[j+1])
                if in_bin.any():
                    total_in_bin = in_bin.sum()
                    spurious_in_bin = sum(1 for i in np.where(in_bin)[0] if i not in matched_pred_indices)
                    fp_rates.append(100.0 * spurious_in_bin / total_in_bin)
                else:
                    fp_rates.append(0.0)
            
            ax_err.plot(bin_centers, fn_rates, 'r-o', label='False Negative Rate (%)', linewidth=2)
            ax_err.plot(bin_centers, fp_rates, 'o-', color='orange', label='False Positive Rate (%)', linewidth=2)
            ax_err.set_xlabel("Strength (Target Objectness / Pred Confidence)")
            ax_err.set_ylabel("Error Rate (%)")
            ax_err.set_title("Classification Reliability")
            ax_err.set_ylim(-5, 105)
            ax_err.grid(True, alpha=0.3)
            ax_err.legend()

        # NEW: Matched vs Missed Histogram (Detection Completeness)
        if all_true_mags:
            ax_comp = fig.add_subplot(gs[4, 1])
            ax_comp.hist([matched_true_mags, missed_true_mags], bins=bins, stacked=True, 
                         label=['Detected', 'Missed'], color=['green', 'red'], alpha=0.7)
            ax_comp.set_xlabel("True log10(Flux)")
            ax_comp.set_ylabel("Count")
            ax_comp.set_title("Detection Completeness (Matched vs Missed)")
            ax_comp.legend()
            ax_comp.grid(True, alpha=0.2)

        if detected_shapes:
            ax_psf_x, ax_psf_y = fig.add_subplot(gs[3:, 2]), fig.add_subplot(gs[3:, 3])
            for i in range(min(100, len(detected_shapes))):
                shape = detected_shapes[i]
                ax_psf_x.plot(np.mean(shape, axis=0), color='C0', alpha=0.1)
                ax_psf_y.plot(np.mean(shape, axis=1), color='C1', alpha=0.1)
            ax_psf_x.set_title("PSF X-Profiles"); ax_psf_y.set_title("PSF Y-Profiles")

        plt.suptitle(f"Generative Diagnostic (Scale={self.stretch_scale}) | Predicted Stars (p>=0.5): {len(detected_stars)}", fontsize=24)
        plt.savefig(output_path); print(f"Comparison saved to {output_path}")
