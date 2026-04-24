import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
from scipy.ndimage import zoom, shift, center_of_mass
from astropy.io import fits
from castor.data.transforms import AstroSpaceTransform
from castor.constants import GLOBAL_STRETCH_SCALE, SHAPE_SIZE, DEFAULT_CELL_SIZE
from scipy.signal import fftconvolve

def upsample_background(bg_map, target_size):
    """
    Upsamples a grid-based background map to full image resolution.

    Uses bilinear interpolation with correct physical centering (cell centers).

    Parameters
    ----------
    bg_map : numpy.ndarray
        The low-resolution background map.
    target_size : tuple of int
        The (height, width) of the target high-resolution image.

    Returns
    -------
    numpy.ndarray
        The upsampled background map.
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

def generate_custom_inference_prior(star_catalog, img_size, sigma=1.0, device='cpu'):
    """
    Generates a 1-channel bilinear splat confidence prior map for inference testing.

    Parameters
    ----------
    star_catalog : list of tuple
        A list of stars, where each star is (cx, cy, p).
    img_size : int
        The size of the output square image.
    sigma : float, optional
        Gaussian sigma (currently unused in this bilinear implementation), by default 1.0.
    device : str, optional
        The device to perform calculations on ('cpu' or 'cuda'), by default 'cpu'.

    Returns
    -------
    numpy.ndarray
        The generated prior map of shape [img_size, img_size].
    """
    # 1 channel: P
    prior_map = torch.zeros((img_size, img_size), device=device)
    if not star_catalog:
        return prior_map.cpu().numpy()
        
    # Convert to tensors
    catalog = torch.tensor(star_catalog, device=device)
    # catalog expected to be list of (cx, cy, p)
    cx, cy, p = catalog[:, 0], catalog[:, 1], catalog[:, 2]
    
    # Map to pixels
    # Ensure anchors are in [0, img_size-2] for 2x2 splat
    cx = torch.clamp(cx, 0, img_size - 1.001)
    cy = torch.clamp(cy, 0, img_size - 1.001)
    
    x0 = torch.floor(cx).long()
    y0 = torch.floor(cy).long()
    dx = cx - x0.float()
    dy = cy - y0.float()
    
    # Bilinear weights scaled by P
    w00 = (1 - dx) * (1 - dy) * p
    w10 = dx * (1 - dy) * p
    w01 = (1 - dx) * dy * p
    w11 = dx * dy * p
    
    # Scatter add into the flat array
    prior_flat = prior_map.view(-1)
    
    idx00 = y0 * img_size + x0
    idx10 = idx00 + 1
    idx01 = idx00 + img_size
    idx11 = idx01 + 1
    
    prior_flat.scatter_add_(0, idx00, w00)
    prior_flat.scatter_add_(0, idx10, w10)
    prior_flat.scatter_add_(0, idx01, w01)
    prior_flat.scatter_add_(0, idx11, w11)
    
    return prior_map.cpu().numpy()

from castor.data.stage0_gaussian import render_gaussian_stars, get_gaussian_psf, get_oversampled_gaussian_psf

class InferenceEngine:
    """
    Handles model inference and visualization of results.

    Attributes
    ----------
    model : torch.nn.Module
        The trained neural network model.
    device : torch.device
        The device to run inference on.
    config : dict
        Configuration dictionary.
    img_size : int
        Size of the input images.
    stretch_scale : float
        Scale for arcsinh stretching.
    transform : AstroSpaceTransform
        Transform for image processing.
    cell_size : int
        Size of the grid cells.
    """

    def __init__(self, model, device, config):
        """
        Initialize the InferenceEngine.

        Parameters
        ----------
        model : torch.nn.Module
            The trained model.
        device : torch.device
            The device to use.
        config : dict
            The configuration dictionary.
        """
        self.model = model
        self.device = device
        self.config = config
        self.img_size = config["data_params"]["image_size"]
        self.stretch_scale = config["data_params"].get("GLOBAL_STRETCH_SCALE", GLOBAL_STRETCH_SCALE)
        self.transform = AstroSpaceTransform(stretch_scale=self.stretch_scale)
        # Get cell_size from config, fallback to default
        self.cell_size = config.get("curriculum", {}).get("stage0", {}).get("cell_size", DEFAULT_CELL_SIZE)
        # Get sigma from config, fallback to realistic default
        self.sigma = config.get("data_params", {}).get("physics_params", {}).get("sigma_fixed", 0.405)

    def _get_centered_psf(self):
        """
        Generates a diagnostic centered Gaussian PSF for reconstruction.

        Returns
        -------
        numpy.ndarray
            A centered Gaussian PSF of size SHAPE_SIZE.
        """
        # Centralized Source of Truth (Uses value from config)
        return get_oversampled_gaussian_psf(sigma_detector=self.sigma, grid_size=SHAPE_SIZE, oversample=4)

    def predict(self, image_tensor, threshold=0.1, prior_map=None, chunk_median=None):
        """
        Runs inference on a single 2D image tensor.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Input image tensor of shape [H, W].
        threshold : float, optional
            Probability threshold for detecting stars, by default 0.1.
        prior_map : numpy.ndarray or torch.Tensor, optional
            Optional prior map to guide inference, by default None.
        chunk_median : float, optional
            Optional pre-calculated median for subtraction, by default None.

        Returns
        -------
        tuple
            A tuple (predicted_stars, bg_map). `predicted_stars` is a list 
            of ((x, y), flux, p, sigmas).
        """
        self.model.eval()
        
        with torch.no_grad():
            # 1. Pre-processing: Robust Median Subtraction and Arcsinh Stretch
            if chunk_median is None:
                # Use a lower percentile for robust background estimation in dense fields
                # This prevents bright stars from skewing the median and destroying faint sources.
                chunk_median = float(torch.quantile(image_tensor.view(-1), 0.10))
            
            stretched_tensor = torch.arcsinh((image_tensor - chunk_median) / self.stretch_scale)
            
            if stretched_tensor.dim() == 2:
                input_tensor = stretched_tensor.unsqueeze(0).unsqueeze(0)
            elif stretched_tensor.dim() == 3:
                input_tensor = stretched_tensor.unsqueeze(0)
            else:
                input_tensor = stretched_tensor
                
            input_tensor = input_tensor.to(self.device)
            
            # Handle Prior Map
            if prior_map is not None:
                if isinstance(prior_map, np.ndarray):
                    prior_tensor = torch.from_numpy(prior_map).float()
                else:
                    prior_tensor = prior_map.float()
                
                if prior_tensor.dim() == 2:
                    prior_tensor = prior_tensor.unsqueeze(0).unsqueeze(0)
                elif prior_tensor.dim() == 3:
                    prior_tensor = prior_tensor.unsqueeze(0)
                
                prior_tensor = prior_tensor.to(self.device)
            else:
                prior_tensor = None

            # Match training mixed precision context
            device_type = self.device.type if self.device.type != 'cpu' else 'cpu'
            dtype = torch.float16 if self.device.type != 'cpu' else torch.bfloat16
            with torch.autocast(device_type=device_type, dtype=dtype):
                prediction_dict = self.model(input_tensor, prior=prior_tensor)
            
            prediction = prediction_dict["stars"].squeeze(0).float().cpu().numpy()
            bg_map = prediction_dict["background"].squeeze(0).float().cpu().numpy()
            
        predicted_stars = []
        grid_h, grid_w, K, _ = prediction.shape
        cell_size = self.img_size // grid_h
        
        for y in range(grid_h):
            for x in range(grid_w):
                for k in range(K):
                    p, dx, dy, physical_flux = prediction[y, x, k, :4]
                    if p > threshold:
                        # physical_flux is already linear!
                        
                        # Uncertainty Estimates (Log-variance)
                        log_vars = prediction[y, x, k, 4:7]
                        sigmas = np.exp(0.5 * log_vars)
                        
                        # Fix the Photometric Sigma propagation for your plots
                        # cosh(arcsinh(F/S)) = sqrt(1 + (F/S)^2)
                        scaled_f = physical_flux / self.stretch_scale
                        sigma_f_linear = self.stretch_scale * np.sqrt(1 + scaled_f**2) * sigmas[2]
                        linear_sigmas = np.array([sigmas[0], sigmas[1], sigma_f_linear])
                        
                        predicted_stars.append(((x * cell_size) + dx, (y * cell_size) + dy, float(physical_flux), float(p), linear_sigmas))
                            
        return predicted_stars, bg_map

    def visualize(self, hero_data, global_true, global_pred, threshold=0.43, output_path="inference_comparison.png", num_chunks=1):
        """
        Visualizes inference results with the full 12-axis diagnostic suite.

        Parameters
        ----------
        hero_data : dict
            Data for the "hero" sample used in visual comparisons.
        global_true : list
            List of all true stars across all evaluated chunks.
        global_pred : list
            List of all predicted stars across all evaluated chunks.
        threshold : float, optional
            Detection threshold, by default 0.43.
        output_path : str, optional
            Path to save the diagnostic plot, by default "inference_comparison.png".
        num_chunks : int, optional
            Number of chunks evaluated, by default 1.
        """
        from castor.engine.evaluator import match_stars
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        # 1. COMPONENT PREPARATION (HERO SAMPLE)
        img_stretched = hero_data["image_stretched"].squeeze().numpy()
        H, W = img_stretched.shape
        hero_true = hero_data["true_stars"]
        hero_pred = hero_data["pred_stars"]
        bg_map = hero_data["bg_map"]
        gt_bg_map = hero_data["gt_bg_map"]
        chunk_median = hero_data.get("chunk_median", 0.0)
        prior_map = hero_data.get("prior_map")

        # Match for Hero Plot
        h_match_pred_filtered = [s for s in hero_pred if s[3] >= threshold]
        h_matches, _, _ = match_stars([(s[1], s[2], s[3]) for s in hero_true], [(s[0], s[1], s[2]) for s in h_match_pred_filtered], distance_threshold=1.0)
        h_matched_true_indices = [m[0] for m in h_matches]

        # Match for Global Stats
        g_match_true = [(s[1], s[2], s[3]) for s in global_true]
        g_pred_filtered = [s for s in global_pred if s[3] >= threshold]
        g_matches, g_unmatched_true, g_unmatched_pred = match_stars(g_match_true, [(s[0], s[1], s[2]) for s in g_pred_filtered], distance_threshold=1.0)
        g_matched_true_indices = [m[0] for m in g_matches]
        
        base_psf = self._get_centered_psf()
        flipped_psf = base_psf[::-1, ::-1]

        # Draw Hero Reconstruction
        def draw_stars_on_grid(stars, height, width, is_predicted=True):
            grid = np.zeros((height, width), dtype=np.float32)
            if not stars: return grid
            if is_predicted:
                px, py, fluxes = np.array([s[0] for s in stars]), np.array([s[1] for s in stars]), np.array([s[2] for s in stars])
            else:
                px, py, fluxes = np.array([s[1] for s in stars]), np.array([s[2] for s in stars]), np.array([s[3] for s in stars])
            
            x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
            dx, dy = px - x0, py - y0
            w00, w10, w01, w11 = (1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy
            def paint(x, y, w):
                mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)
                flat_idx = y[mask] * width + x[mask]
                grid.flat += np.bincount(flat_idx, weights=fluxes[mask] * w[mask], minlength=grid.size)
            paint(x0, y0, w00); paint(x0+1, y0, w10); paint(x0, y0+1, w01); paint(x0+1, y0+1, w11)
            return grid

        h_rec_stars_linear = fftconvolve(draw_stars_on_grid(h_match_pred_filtered, H, W, True), flipped_psf, mode='same')
        h_missed_stars = [hero_true[i] for i in range(len(hero_true)) if i not in h_matched_true_indices and hero_true[i][0] >= 0.1]
        h_rec_missed_linear = fftconvolve(draw_stars_on_grid(h_missed_stars, H, W, False), flipped_psf, mode='same')

        # Convert low-res grid to absolute linear photons FIRST
        bg_linear_lowres = self.transform.network_to_image(bg_map.squeeze(), chunk_median)
        gt_bg_linear_lowres = self.transform.network_to_image(gt_bg_map.squeeze(), chunk_median)

        # THEN upsample in linear physical space
        full_bg_abs = upsample_background(bg_linear_lowres, (H, W))
        full_gt_bg_abs = upsample_background(gt_bg_linear_lowres, (H, W))

        h_rec_abs = h_rec_stars_linear + full_bg_abs
        img_linear_abs = self.transform.network_to_image(img_stretched, chunk_median)
        h_residual_linear = img_linear_abs - h_rec_abs

        # FITS Output
        fits_path = output_path.replace(".png", ".fits")
        h_rec_missed_abs = h_rec_missed_linear + full_gt_bg_abs
        
        hdul = fits.HDUList([
            fits.PrimaryHDU(), 
            fits.ImageHDU(img_linear_abs, name="INPUT_LINEAR"), 
            fits.ImageHDU(h_rec_abs, name="MODEL_LINEAR"), 
            fits.ImageHDU(h_residual_linear, name="RESIDUAL_LINEAR"),
            fits.ImageHDU(h_rec_missed_linear, name="MISSED_LINEAR"),
            fits.ImageHDU(h_rec_missed_abs, name="MISSED_ABS"),
            fits.ImageHDU(full_bg_abs, name="BG_PRED_LINEAR"), 
            fits.ImageHDU(full_gt_bg_abs, name="BG_TRUE_LINEAR")
        ])
        if prior_map is not None:
            hdul.append(fits.ImageHDU(prior_map, name="PRIOR_MAP"))
        hdul.writeto(fits_path, overwrite=True)

        # 3. STATISTICS (GLOBAL)
        physics_cfg = self.config.get("data_params", {}).get("physics_params", {})
        zp, exp_time = physics_cfg.get("zp", 26.5), physics_cfg.get("exp_time_min", 30.0)
        def flux_to_mag(f): return zp - 2.5 * np.log10(np.maximum(f, 1e-9) / exp_time)

        g_true_mags = np.array([flux_to_mag(s[3]) for s in global_true])
        g_pred_mags = np.array([flux_to_mag(s[2]) for s in g_pred_filtered])
        g_matched_true_mags = np.array([flux_to_mag(global_true[m[0]][3]) for m in g_matches])
        g_matched_pred_mags = np.array([flux_to_mag(g_pred_filtered[m[1]][2]) for m in g_matches])
        g_missed_true_mags = np.array([flux_to_mag(global_true[i][3]) for i in range(len(global_true)) if i not in g_matched_true_indices])

        # Figure Layout
        fig = plt.figure(figsize=(30, 24))
        gs = fig.add_gridspec(6, 4, hspace=0.3, wspace=0.3)
        
        def add_colorbar(im, ax):
            divider = make_axes_locatable(ax); cax = divider.append_axes("right", size="5%", pad=0.05); fig.colorbar(im, cax=cax)

        # Plot Hero Images
        l_vmin, l_vmax = np.percentile(img_linear_abs, [10, 99.9])
        norm = LogNorm(vmin=max(1.0, l_vmin), vmax=max(l_vmin+1.0, l_vmax), clip=True)
        ax1 = fig.add_subplot(gs[0:2, 0]); ax1.imshow(img_linear_abs, cmap='inferno', origin='lower', norm=norm)
        ax1.set_title("Input (Hero Sample)")
        ax2 = fig.add_subplot(gs[0:2, 1], sharex=ax1, sharey=ax1); im2 = ax2.imshow(h_rec_abs, cmap='inferno', origin='lower', norm=norm)
        ax2.set_title("Model Reconstruction"); add_colorbar(im2, ax2)
        ax3 = fig.add_subplot(gs[0:2, 2], sharex=ax1, sharey=ax1); r_lim = np.percentile(np.abs(h_residual_linear), 99)
        im3 = ax3.imshow(h_residual_linear, cmap='bwr', origin='lower', vmin=-r_lim, vmax=r_lim); ax3.set_title("Linear Residual"); add_colorbar(im3, ax3)

        # Prior Map
        if prior_map is not None:
            ax_prior = fig.add_subplot(gs[2, 2], sharex=ax1, sharey=ax1)
            # Display P channel (channel 0) if 3D, else just show it
            if prior_map.ndim == 3:
                display_prior = prior_map[0]
            else:
                display_prior = prior_map
            im_prior = ax_prior.imshow(display_prior, cmap='magma', origin='lower', vmin=0, vmax=1)
            ax_prior.set_title("Inference Prior Map"); add_colorbar(im_prior, ax_prior)

        # Backgrounds
        ax4 = fig.add_subplot(gs[2, 0], sharex=ax1, sharey=ax1); ax4.imshow(full_bg_abs, cmap='viridis', origin='lower'); ax4.set_title("Predicted Background")
        ax5 = fig.add_subplot(gs[2, 1], sharex=ax1, sharey=ax1); im5 = ax5.imshow(full_gt_bg_abs, cmap='viridis', origin='lower'); ax5.set_title("Truth Background"); add_colorbar(im5, ax5)

        # Astrometric Residual Histograms
        if g_matches:
            pos_res = np.array([(g_pred_filtered[m[1]][0]-global_true[m[0]][1], g_pred_filtered[m[1]][1]-global_true[m[0]][2]) for m in g_matches])
            ax_res_hist = fig.add_subplot(gs[0, 3])
            ax_res_hist.hist(pos_res[:, 0], bins=50, alpha=0.5, label='dx', color='C0')
            ax_res_hist.hist(pos_res[:, 1], bins=50, alpha=0.5, label='dy', color='C1')
            ax_res_hist.set_title("Astrometric Residuals (px)"); ax_res_hist.legend()
            
            # Photometric Residual Histogram
            ax_mag_hist = fig.add_subplot(gs[1, 3])
            mag_res = g_matched_pred_mags - g_matched_true_mags
            ax_mag_hist.hist(mag_res, bins=50, color='C2', alpha=0.7)
            ax_mag_hist.set_title("Photometric Residuals (mag)"); ax_mag_hist.set_xlabel("Pred - True")

        # Detection Tradeoff (FPR/FNR)
        ax_err = fig.add_subplot(gs[4, 0])
        thresholds = np.linspace(0.01, 0.99, 50)
        t_list_vis = [(s[1], s[2], s[3]) for s in global_true if s[0] > 0.5]
        t_list_all = [(s[1], s[2], s[3]) for s in global_true if s[0] > 0.1]
        fpr_list, fnr_rates = [], []
        for thr in thresholds:
            p_cand = [s for s in global_pred if s[3] >= thr]
            if not p_cand: fpr_list.append(0.0); fnr_rates.append(100.0 if t_list_vis else 0.0); continue
            _, _, up = match_stars(t_list_all, [(s[0], s[1], s[2]) for s in p_cand], distance_threshold=1.0); fpr_list.append(100.0 * len(up) / len(p_cand))
            if t_list_vis: _, ut_v, _ = match_stars(t_list_vis, [(s[0], s[1], s[2]) for s in p_cand], distance_threshold=1.0); fnr_rates.append(100.0 * len(ut_v) / len(t_list_vis))
            else: fnr_rates.append(0.0)
        ax_err.plot(thresholds, fnr_rates, 'r-', label='Missed %', linewidth=3)
        ax_err.plot(thresholds, fpr_list, '-', color='orange', label='False Pos %', linewidth=3)
        ax_err.set_title("Detection Tradeoff"); ax_err.set_ylim(-5, 105); ax_err.legend()

        # Precision vs Threshold
        ax_prec_p = fig.add_subplot(gs[2, 3])
        ax_prec_p.plot(thresholds, 100.0 - np.array(fpr_list), 'b-', linewidth=3)
        ax_prec_p.set_title("Precision vs Threshold"); ax_prec_p.set_ylim(-5, 105)

        # 2D Astrometric Residuals
        if g_matches:
            ax_ast_2d = fig.add_subplot(gs[5, 3])
            matched_flux = np.array([g_pred_filtered[m[1]][2] for m in g_matches])
            sc = ax_ast_2d.scatter(pos_res[:, 0], pos_res[:, 1], c=np.log10(np.maximum(matched_flux, 1e-9)), cmap='viridis', alpha=0.4, s=1)
            ax_ast_2d.set_aspect('equal'); ax_ast_2d.set_title("Astrometric Residuals"); add_colorbar(sc, ax_ast_2d)
            r_lim = np.percentile(np.sqrt(np.sum(pos_res**2, axis=1)), 95) * 1.5; ax_ast_2d.set_xlim(-r_lim, r_lim); ax_ast_2d.set_ylim(-r_lim, r_lim)

            # Uncertainty Sigmas
            matched_sigmas = np.array([g_pred_filtered[m[1]][4] for m in g_matches])
            ax_sig_x = fig.add_subplot(gs[4, 2]); ax_sig_x.scatter(matched_flux, matched_sigmas[:, 0], alpha=0.2, s=1, color='C0'); ax_sig_x.set_xscale('log'); ax_sig_x.set_yscale('log'); ax_sig_x.set_title("Astrometric Sigma (X)")
            ax_sig_y = fig.add_subplot(gs[4, 3]); ax_sig_y.scatter(matched_flux, matched_sigmas[:, 1], alpha=0.2, s=1, color='C1'); ax_sig_y.set_xscale('log'); ax_sig_y.set_yscale('log'); ax_sig_y.set_title("Astrometric Sigma (Y)")
            ax_sig_f = fig.add_subplot(gs[5, 2]); ax_sig_f.scatter(matched_flux, matched_sigmas[:, 2], alpha=0.2, s=1, color='C2'); ax_sig_f.set_xscale('log'); ax_sig_f.set_yscale('log'); ax_sig_f.set_title("Photometric Sigma")

        # Recovery vs True-p
        ax_rec_p = fig.add_subplot(gs[3, 2]); true_labels = np.array([s[0] for s in global_true]); valid_mask = true_labels > 0.05
        if np.any(valid_mask):
            v_labels, is_recovered = true_labels[valid_mask], np.array([1 if i in g_matched_true_indices else 0 for i in range(len(global_true)) if valid_mask[i]])
            p_bins = np.linspace(0, 1.0, 11); rec_stats = []
            for i in range(len(p_bins)-1):
                m = (v_labels >= p_bins[i]) & (v_labels <= p_bins[i+1]) if i == len(p_bins)-2 else (v_labels >= p_bins[i]) & (v_labels < p_bins[i+1])
                rec_stats.append(100.0 * np.mean(is_recovered[m]) if np.any(m) else np.nan)
            ax_rec_p.step(p_bins, np.append(rec_stats, rec_stats[-1]), where='post', color='green', linewidth=3)
            ax_rec_p.set_title("Recovery vs True-p"); ax_rec_p.set_ylim(-5, 105); ax_rec_p.set_xlim(0, 1.0)

        # LF Recovery
        ax_hist = fig.add_subplot(gs[3, 1])
        ax_hist.hist(g_true_mags, bins=50, alpha=0.2, color='black', label='Truth')
        ax_hist.hist(g_pred_mags, bins=50, alpha=0.4, color='blue', label=f'Pred (p>={threshold})')
        ax_hist.set_title("LF Recovery"); ax_hist.invert_xaxis(); ax_hist.legend()
        
        # Completeness vs Magnitude
        ax_comp = fig.add_subplot(gs[4, 1])
        mag_bins = np.linspace(18, 28, 21)
        hist_true, _ = np.histogram(g_true_mags, bins=mag_bins)
        hist_matched, _ = np.histogram(g_matched_true_mags, bins=mag_bins)
        ax_comp.bar(mag_bins[:-1], hist_true, width=0.5, alpha=0.3, color='gray', label='Truth', align='edge')
        ax_comp.bar(mag_bins[:-1], hist_matched, width=0.5, alpha=0.7, color='green', label='Found', align='edge')
        ax_comp.set_title("Completeness vs Mag"); ax_comp.invert_xaxis(); ax_comp.legend(loc='upper left')
        
        # Photometry scatter
        if g_matches:
            ax8 = fig.add_subplot(gs[3, 0])
            # Use p-values for color mapping
            matched_p_values = np.array([g_pred_filtered[m[1]][3] for m in g_matches])
            sc_photo = ax8.scatter(g_matched_true_mags, g_matched_pred_mags, 
                                  c=matched_p_values, cmap='viridis', 
                                  alpha=0.6, s=10, vmin=0, vmax=1)
            
            all_mags = np.concatenate([g_matched_true_mags, g_matched_pred_mags])
            if len(all_mags) > 0:
                m_min, m_max = np.min(all_mags), np.max(all_mags)
                ax8.plot([m_min, m_max], [m_min, m_max], 'k--', alpha=0.5, zorder=0)
            
            ax8.set_title("Global Photometry (Color: p-value)")
            ax8.invert_xaxis(); ax8.invert_yaxis()
            ax8.set_xlabel("True Mag"); ax8.set_ylabel("Pred Mag")
            add_colorbar(sc_photo, ax8)

        plt.suptitle(f"Global Validation Summary ({num_chunks:,} Chunks) | Predicted: {len(g_pred_filtered):,}", fontsize=24)
        plt.savefig(output_path); plt.close()
        print(f"Global diagnostic saved to {output_path}")
