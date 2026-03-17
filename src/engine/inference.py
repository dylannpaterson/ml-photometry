import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

class InferenceEngine:
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config

    def predict(self, image_tensor, threshold=0.5):
        """Runs inference on a single 2D image tensor [1, H, W]."""
        self.model.eval()
        
        with torch.no_grad():
            input_tensor = image_tensor.unsqueeze(0).to(self.device)
            prediction = self.model(input_tensor).squeeze(0).cpu().numpy()
            
        predicted_stars = []
        predicted_shapes = []
        cell_size = 2
        grid_h, grid_w, K, _ = prediction.shape
        
        for y in range(grid_h):
            for x in range(grid_w):
                for k in range(K):
                    # log_m is log10(flux)
                    p, dx, dy, log_m, c = prediction[y, x, k, :5]
                    if p > threshold:
                        global_x = (x * cell_size) + dx
                        global_y = (y * cell_size) + dy
                        # Convert log10(flux) back to linear flux
                        m = 10**log_m
                        predicted_stars.append((global_x, global_y, m, c, p))
                        
                        # Extract PSF shape (Dynamic size)
                        shape_vector = prediction[y, x, k, 5:]
                        S = int(np.sqrt(len(shape_vector)))
                        shape_psf = shape_vector.reshape(S, S)
                        predicted_shapes.append(shape_psf)
                        
        return predicted_stars, predicted_shapes

    def visualize(self, image_tensor, true_catalogue, predicted_stars, predicted_shapes, threshold, output_path="inference_comparison.png"):
        from src.engine.evaluator import match_stars
        img = image_tensor.squeeze().numpy()
        
        # 1. Match stars for the Magnitude Scatter Plot
        matches, _, _ = match_stars(true_catalogue, predicted_stars)
        
        true_mags = []
        pred_mags = []
        for t_idx, p_idx, _ in matches:
            true_mags.append(np.log10(true_catalogue[t_idx][2] + 1e-9))
            pred_mags.append(np.log10(predicted_stars[p_idx][2] + 1e-9))

        # 2. Create Model Reconstruction
        reconstruction = np.zeros_like(img)
        H, W = img.shape
        
        for (x, y, flux, c, p), shape in zip(predicted_stars, predicted_shapes):
            ix, iy = int(round(x)), int(round(y))
            # Dynamic PSF size handling
            S = shape.shape[0]
            half = S // 2
            
            y0, y1 = max(0, iy - half), min(H, iy + half + 1)
            x0, x1 = max(0, ix - half), min(W, ix + half + 1)
            sy0, sy1 = half - (iy - y0), half + (y1 - iy)
            sx0, sx1 = half - (ix - x0), half + (x1 - ix)
            
            patch = flux * shape[sy0:sy1, sx0:sx1]
            reconstruction[y0:y1, x0:x1] += patch

        # 3. Calculate Residual
        bg_level = np.median(img)
        residual = (img - bg_level) - reconstruction

        # 4. Create Complex Layout
        fig = plt.figure(figsize=(24, 24))
        gs = fig.add_gridspec(5, 3)
        
        # Row 1-2: Image Comparison
        # Original
        ax1 = fig.add_subplot(gs[0:2, 0])
        ax1.imshow(img, cmap='inferno', origin='lower', norm=LogNorm(vmin=10, vmax=img.max()))
        ax1.set_title(f"Original Input ({len(true_catalogue)} stars)")
        for star in true_catalogue:
            ax1.plot(star[0], star[1], 'g+', markersize=8, alpha=0.4)
        
        # Reconstruction
        ax2 = fig.add_subplot(gs[0:2, 1])
        viz_reconstruction = reconstruction + bg_level
        ax2.imshow(viz_reconstruction, cmap='inferno', origin='lower', norm=LogNorm(vmin=10, vmax=img.max()))
        ax2.set_title(f"Model Reconstruction ({len(predicted_stars)} stars)")
        
        # Residual
        ax3 = fig.add_subplot(gs[0:2, 2])
        rmax = np.percentile(np.abs(residual), 99)
        im3 = ax3.imshow(residual, cmap='bwr', origin='lower', vmin=-rmax, vmax=rmax)
        ax3.set_title(f"Residual (bg={bg_level:.1f})")
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # Mag Scatter Plot (Row 3)
        ax4 = fig.add_subplot(gs[2, 0:2])
        if true_mags:
            ax4.scatter(true_mags, pred_mags, alpha=0.5, color='blue', label='Detections')
            all_vals = true_mags + pred_mags
            lims = [min(all_vals), max(all_vals)]
            ax4.plot(lims, lims, 'r--', alpha=0.7, label='1:1 Line')
            ax4.set_xlabel("True log10(Flux)")
            ax4.set_ylabel("Predicted log10(Flux)")
            ax4.set_title(f"Magnitude Recovery (Matched Stars: {len(matches)})")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "No matches found", ha='center')

        # Bottom Rows: PSF Comparisons
        if predicted_stars:
            indices = np.argsort([s[4] for s in predicted_stars])[::-1]
            num_to_show = min(8, len(indices))
            psf_gs = gs[3:5, :].subgridspec(2, 8)
            
            def get_ideal_psf(size=9):
                half = size // 2
                x = np.arange(-half, half + 1)
                y = np.arange(-half, half + 1)
                xx, yy = np.meshgrid(x, y)
                psf = np.exp(-(xx**2 + yy**2) / (2 * 1.5**2))
                psf /= (psf.sum() + 1e-9)
                return psf
            
            S = predicted_shapes[0].shape[0]
            ideal_psf = get_ideal_psf(size=S)

            for i in range(num_to_show):
                idx = indices[i]
                ax_pred = fig.add_subplot(psf_gs[0, i])
                ax_pred.imshow(predicted_shapes[idx], cmap='viridis', origin='lower')
                ax_pred.set_title(f"Pred p={predicted_stars[idx][4]:.2f}")
                ax_pred.axis('off')
                
                ax_true = fig.add_subplot(psf_gs[1, i])
                ax_true.imshow(ideal_psf, cmap='viridis', origin='lower')
                if i == 0: ax_true.set_ylabel("Truth", rotation=0, labelpad=20, size=12)
                ax_true.axis('off')

        plt.suptitle("Inference Diagnostic: Reconstruction & Magnitude Accuracy", fontsize=24)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_path)
        print(f"Comparison saved to {output_path}")
