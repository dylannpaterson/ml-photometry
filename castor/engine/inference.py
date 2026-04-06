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
            # Star shape: [Batch, H, W, K, 7]
            prediction_tensor = prediction_dict["stars"].squeeze(0).float()
            
            prediction = prediction_tensor.cpu().numpy()
            bg_map = prediction_dict["background"].squeeze(0).float().cpu().numpy()
            
        predicted_stars, predicted_shapes = [], []
        grid_h, grid_w, K, _ = prediction.shape
        cell_size = self.img_size // grid_h
        
        # Determine S from basis if available, else fallback
        if mean_psf is not None:
            S = int(mean_psf.shape[0]**0.5)
        else:
            S = 9
        
        for y in range(grid_h):
            for x in range(grid_w):
                for k in range(K):
                    p, dx, dy, physical_flux = prediction[y, x, k, :4]
                    if p > threshold:
                        # NEW: Extract log_vars and convert to sigma (standard deviation)
                        # log_var_x (4), log_var_y (5), log_var_m (6)
                        log_vars = prediction[y, x, k, 4:7]
                        sigmas = np.exp(0.5 * log_vars)
                        
                        predicted_stars.append(((x * cell_size) + dx, (y * cell_size) + dy, float(physical_flux), p, sigmas))
                        
                        # Use Mean PSF for reconstruction (Shape recovery dropped in favor of uncertainty)
                        if mean_psf is not None:
                            predicted_shapes.append(mean_psf.reshape(S, S))
                        else:
                            # Safe fallback - Create a simple 9x9 Gaussian
                            sy, sx = np.meshgrid(np.arange(9)-4, np.arange(9)-4)
                            fallback_psf = np.exp(-(sx**2 + sy**2) / (2 * 1.5**2))
                            fallback_psf /= (fallback_psf.sum() + 1e-9)
                            predicted_shapes.append(fallback_psf)
                            
        return predicted_stars, predicted_shapes, bg_map

    def visualize(self, image_tensor, true_catalogue, predicted_stars, predicted_shapes, bg_map, gt_bg_map, threshold, chunk_median=0.0, jitter_params=None, output_path="inference_comparison.png", psf_basis=None, mean_psf=None):
        """
        Visualizes inference results with Aleatoric Uncertainty.
        jitter_params: (s_jit, q_jit, theta_jit) to match input image smear.
        """
        from castor.engine.evaluator import match_stars
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        img_stretched = image_tensor.squeeze().numpy()
        H, W = img_stretched.shape
        
        # 1. MATCHING
        detected_stars = [s for s in predicted_stars if s[3] >= 0.5]
        detected_shapes = [shp for s, shp in zip(predicted_stars, predicted_shapes) if s[3] >= 0.5]
        
        match_true = [(s[1], s[2], s[3]) for s in true_catalogue]
        match_pred = [(s[0], s[1], s[2]) for s in detected_stars]
        matches, unmatched_true, unmatched_pred = match_stars(match_true, match_pred, distance_threshold=2.0)
        matched_true_indices = [m[0] for m in matches]
        matched_pred_indices = [m[1] for m in matches]

        # 2. COMPONENT PREPARATION
        full_residual_bg_stretched = upsample_background(bg_map.squeeze(), (H, W))
        full_gt_residual_bg_stretched = upsample_background(gt_bg_map.squeeze(), (H, W))
        
        # --- SUB-PIXEL ACCURATE RECONSTRUCTION ---
        reconstruction_stars_linear = np.zeros_like(img_stretched)
        for (x, y, flux, p, sigmas), shape in zip(detected_stars, detected_shapes):
            S_s = shape.shape[0]; half_s = S_s // 2
            x0, y0 = int(np.floor(x)), int(np.floor(y))
            dx, dy = x - x0, y - y0
            w00, w10, w01, w11 = (1.0-dx)*(1.0-dy), dx*(1.0-dy), (1.0-dx)*dy, dx*dy
            for j, i, w in [(0, 0, w00), (1, 0, w10), (0, 1, w01), (1, 1, w11)]:
                if w <= 0: continue
                iy, ix = y0 + i, x0 + j
                y_min, y_max = max(0, iy-half_s), min(H, iy+half_s+1)
                x_min, x_max = max(0, ix-half_s), min(W, ix+half_s+1)
                sy0, sy1 = half_s - (iy-y_min), half_s + (y_max-iy)
                sx0, sx1 = half_s - (ix-x_min), half_s + (x_max-ix)
                if sy1 > sy0 and sx1 > sx0:
                    reconstruction_stars_linear[y_min:y_max, x_min:x_max] += (flux * w) * shape[sy0:sy1, sx0:sx1]

        # --- RECONSTRUCT MISSED SOURCES ---
        reconstruction_missed_linear = np.zeros_like(img_stretched)
        if mean_psf is not None:
            S_m = int(len(mean_psf)**0.5); half_m = S_m // 2
            mean_psf_2d = mean_psf.reshape(S_m, S_m)
            for i in range(len(true_catalogue)):
                if i not in matched_true_indices:
                    p_t, x_t, y_t, flux_t = true_catalogue[i][:4]
                    if p_t < 0.1: continue 
                    x0, y0 = int(np.floor(x_t)), int(np.floor(y_t))
                    dx, dy = x_t - x0, y_t - y0
                    w00, w10, w01, w11 = (1.0-dx)*(1.0-dy), dx*(1.0-dy), (1.0-dx)*dy, dx*dy
                    for j, i_off, w in [(0, 0, w00), (1, 0, w10), (0, 1, w01), (1, 1, w11)]:
                        if w <= 0: continue
                        iy, ix = y0 + i_off, x0 + j
                        ym, yM = max(0, iy-half_m), min(H, iy+half_m+1)
                        xm, xM = max(0, ix-half_m), min(W, ix+half_m+1)
                        sy0, sy1 = half_m - (iy-ym), half_m + (yM-iy)
                        sx0, sx1 = half_m - (ix-xm), half_m + (xM-ix)
                        if sy1 > sy0 and sx1 > sx0:
                            reconstruction_missed_linear[ym:yM, xm:xM] += (flux_t * w) * mean_psf_2d[sy0:sy1, sx0:sx1]

        # --- APPLY GLOBAL JITTER ---
        if jitter_params is not None:
            s_j, q_j, t_j = jitter_params
            kj = 63; gy, gx = np.meshgrid(np.arange(127) - kj, np.arange(127) - kj, indexing='ij')
            cos, sin = np.cos(t_j), np.sin(t_j)
            gxp, gyp = gx * cos + gy * sin, -gx * sin + gy * cos
            j_kernel = np.exp(-(gxp**2 / (2 * s_j**2) + gyp**2 / (2 * (s_j * q_j)**2)))
            j_kernel /= (j_kernel.sum() + 1e-9)
            reconstruction_stars_linear = fftconvolve(reconstruction_stars_linear, j_kernel, mode='same')
            reconstruction_missed_linear = fftconvolve(reconstruction_missed_linear, j_kernel, mode='same')

        # 3. ABSOLUTE SPACE CONVERSION
        full_residual_bg_linear = self.transform.network_to_bg(full_residual_bg_stretched)
        full_reconstruction_linear_abs = reconstruction_stars_linear + full_residual_bg_linear + chunk_median
        img_linear_abs = self.transform.network_to_image(img_stretched, chunk_median)
        residual_linear = img_linear_abs - full_reconstruction_linear_abs
        full_bg_abs = full_residual_bg_linear + chunk_median
        full_gt_bg_abs = self.transform.network_to_bg(full_gt_residual_bg_stretched) + chunk_median

        # FITS Output
        hdul = fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(img_linear_abs, name="INPUT_LINEAR"),
            fits.ImageHDU(full_reconstruction_linear_abs, name="MODEL_LINEAR"),
            fits.ImageHDU(residual_linear, name="RESIDUAL_LINEAR"),
            fits.ImageHDU(reconstruction_missed_linear, name="MISSED_LINEAR"),
            fits.ImageHDU(full_bg_abs, name="BG_PRED_LINEAR"),
            fits.ImageHDU(full_gt_bg_abs, name="BG_TRUE_LINEAR")
        ])
        fits_path = output_path.replace(".png", ".fits")
        hdul.writeto(fits_path, overwrite=True)

        # Statistics
        matched_true_mags = [np.log10(true_catalogue[m[0]][3] + 1e-9) for m in matches]
        matched_pred_mags = [np.log10(detected_stars[m[1]][2] + 1e-9) for m in matches]
        all_true_mags = [np.log10(s[3] + 1e-9) for s in true_catalogue]
        missed_true_mags = [np.log10(true_catalogue[i][3] + 1e-9) for i in range(len(true_catalogue)) if i not in matched_true_indices]

        # Figure Layout
        fig = plt.figure(figsize=(30, 24))
        gs = fig.add_gridspec(5, 4, hspace=0.3, wspace=0.3)
        
        def add_colorbar(im, ax):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)

        def sanitize_for_plot(data, fill_val=0.0):
            d = data.copy(); d[~np.isfinite(d)] = fill_val
            return np.clip(d, -1e15, 1e15)

        img_linear_abs = sanitize_for_plot(img_linear_abs, fill_val=chunk_median)
        full_reconstruction_linear_abs = sanitize_for_plot(full_reconstruction_linear_abs, fill_val=chunk_median)
        residual_linear = sanitize_for_plot(residual_linear, fill_val=0.0)

        l_vmin, l_vmax = np.percentile(img_linear_abs, [10, 99.9])
        norm = LogNorm(vmin=max(1.0, l_vmin), vmax=max(l_vmin+1.0, l_vmax), clip=True)
        
        ax1 = fig.add_subplot(gs[0:2, 0])
        ax1.imshow(img_linear_abs, cmap='inferno', origin='lower', norm=norm, aspect='equal')
        ax1.set_title("Input (Missed Sources = Cyan)")
        
        ax2 = fig.add_subplot(gs[0:2, 1], sharex=ax1, sharey=ax1)
        im2 = ax2.imshow(full_reconstruction_linear_abs, cmap='inferno', origin='lower', norm=norm, aspect='equal')
        ax2.set_title("Model (Matched Sources = Lime)")
        
        visible_true_indices = [i for i, s in enumerate(true_catalogue) if s[0] >= 0.1]
        for i in visible_true_indices:
            s = true_catalogue[i]
            if i in matched_true_indices: ax2.plot(s[1], s[2], color='lime', marker='+', markersize=10, alpha=0.8)
            else: ax1.plot(s[1], s[2], color='cyan', marker='+', markersize=10, alpha=0.8)
        
        pred_x, pred_y = [s[0] for s in detected_stars], [s[1] for s in detected_stars]
        ax2.scatter(pred_x, pred_y, color='red', s=1, alpha=0.5)
        add_colorbar(im2, ax2)
        
        ax3 = fig.add_subplot(gs[0:2, 2], sharex=ax1, sharey=ax1)
        r_lim = np.percentile(np.abs(residual_linear), 99)
        im3 = ax3.imshow(residual_linear, cmap='bwr', origin='lower', vmin=-r_lim, vmax=r_lim, aspect='equal')
        ax3.set_title("Linear Residual (Missed = Black)")
        for i in visible_true_indices:
            if i not in matched_true_indices:
                s = true_catalogue[i]; ax3.plot(s[1], s[2], 'k+', markersize=10, alpha=0.8)
        add_colorbar(im3, ax3)

        # Background Row
        full_bg_abs, full_gt_bg_abs = sanitize_for_plot(full_bg_abs, fill_val=chunk_median), sanitize_for_plot(full_gt_bg_abs, fill_val=chunk_median)
        bg_vmin, bg_vmax = min(full_bg_abs.min(), full_gt_bg_abs.min()), max(full_bg_abs.max(), full_gt_bg_abs.max())
        ax4 = fig.add_subplot(gs[2, 0], sharex=ax1, sharey=ax1)
        ax4.imshow(full_bg_abs, cmap='viridis', origin='lower', vmin=bg_vmin, vmax=bg_vmax, aspect='equal')
        ax4.set_title("Predicted Background")
        ax5 = fig.add_subplot(gs[2, 1], sharex=ax1, sharey=ax1)
        im5 = ax5.imshow(full_gt_bg_abs, cmap='viridis', origin='lower', vmin=bg_vmin, vmax=bg_vmax, aspect='equal')
        ax5.set_title("Truth Background")
        add_colorbar(im5, ax5)

        # Statistics Row
        if matched_true_mags:
            ax8 = fig.add_subplot(gs[3, 0])
            ax8.scatter(matched_true_mags, matched_pred_mags, alpha=0.5, s=10)
            m_all = matched_true_mags + matched_pred_mags
            ax8.plot([min(m_all), max(m_all)], [min(m_all), max(m_all)], 'r--', alpha=0.8)
            ax8.set_title("Photometry Accuracy"); ax8.set_xlabel("True log10(Flux)"); ax8.set_ylabel("Pred log10(Flux)"); ax8.grid(True, alpha=0.3)

        if all_true_mags:
            ax_hist = fig.add_subplot(gs[3, 1])
            m_p10 = [np.log10(s[2] + 1e-9) for s in predicted_stars if s[3] >= 0.1]
            ax_hist.hist(all_true_mags, bins=30, alpha=0.2, label='Truth', color='black')
            ax_hist.hist(m_p10, bins=30, alpha=0.4, label='p >= 0.1', histtype='step'); ax_hist.set_title("LF Recovery"); ax_hist.legend(); ax_hist.grid(True, alpha=0.2)

        # Threshold Trade-off
        if true_catalogue and predicted_stars:
            ax_err = fig.add_subplot(gs[4, 0])
            thresholds = np.linspace(0.01, 0.99, 50)
            fpr_list, fnr_rates = [], []
            t_list = [(s[1], s[2], s[3]) for s in true_catalogue]
            for thr in thresholds:
                p_list = [(s[0], s[1], s[2]) for s in predicted_stars if s[3] >= thr]
                if not p_list: fpr_list.append(0.0); fnr_rates.append(100.0); continue
                _, ut, up = match_stars(t_list, p_list, distance_threshold=2.0)
                fpr_list.append(100.0 * len(up) / len(p_list)); fnr_rates.append(100.0 * len(ut) / len(t_list))
            ax_err.plot(thresholds, fnr_rates, 'r-', label='False Neg %', linewidth=3)
            ax_err.plot(thresholds, fpr_list, '-', color='orange', label='False Pos %', linewidth=3)
            ax_err.set_title("Detection Trade-off"); ax_err.set_ylim(-5, 105); ax_err.grid(True, alpha=0.3); ax_err.legend()

        if all_true_mags:
            ax_comp = fig.add_subplot(gs[4, 1])
            ax_comp.hist([matched_true_mags, missed_true_mags], bins=30, stacked=True, label=['Detected', 'Missed'], color=['green', 'red'], alpha=0.7)
            ax_comp.set_title("Completeness by Magnitude"); ax_comp.legend(); ax_comp.grid(True, alpha=0.2)

        # --- ALEATORIC UNCERTAINTY VISUALIZATION ---
        if matches:
            # detected_stars[m[1]] is (x, y, flux, p, sigmas)
            # sigmas: (sigma_x, sigma_y, sigma_flux)
            matched_sigmas = np.array([detected_stars[m[1]][4] for m in matches])
            matched_fluxes = np.array([detected_stars[m[1]][2] for m in matches])
            
            ax_sig_x = fig.add_subplot(gs[3, 2])
            ax_sig_y = fig.add_subplot(gs[3, 3])
            ax_sig_f = fig.add_subplot(gs[4, 2])
            
            # Plot Sigma vs Flux (Higher flux should generally have lower relative uncertainty)
            ax_sig_x.scatter(matched_fluxes, matched_sigmas[:, 0], alpha=0.5, color='C0')
            ax_sig_x.set_xscale('log'); ax_sig_x.set_yscale('log')
            ax_sig_x.set_title("Astrometric Uncertainty (X)"); ax_sig_x.set_xlabel("Flux"); ax_sig_x.set_ylabel("sigma_x (pixels)")
            
            ax_sig_y.scatter(matched_fluxes, matched_sigmas[:, 1], alpha=0.5, color='C1')
            ax_sig_y.set_xscale('log'); ax_sig_y.set_yscale('log')
            ax_sig_y.set_title("Astrometric Uncertainty (Y)"); ax_sig_y.set_xlabel("Flux"); ax_sig_y.set_ylabel("sigma_y (pixels)")
            
            # Relative flux uncertainty (sigma_m is in log-space, so it roughly corresponds to fractional flux error)
            ax_sig_f.scatter(matched_fluxes, matched_sigmas[:, 2], alpha=0.5, color='C2')
            ax_sig_f.set_xscale('log'); ax_sig_f.set_yscale('log')
            ax_sig_f.set_title("Photometric Uncertainty"); ax_sig_f.set_xlabel("Flux"); ax_sig_f.set_ylabel("sigma_log_flux")
            
            for ax in [ax_sig_x, ax_sig_y, ax_sig_f]: ax.grid(True, alpha=0.3, which="both")

        plt.suptitle(f"Aleatoric Uncertainty Diagnostic | Predicted Stars (p>=0.5): {len(detected_stars)}", fontsize=24)
        plt.savefig(output_path); print(f"Comparison saved to {output_path}")

        plt.suptitle(f"Generative Diagnostic | Predicted Stars (p>=0.5): {len(detected_stars)}", fontsize=24)
        plt.savefig(output_path); print(f"Comparison saved to {output_path}")
