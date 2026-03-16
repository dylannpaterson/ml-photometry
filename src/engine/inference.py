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
            # Output: [1, H/2, W/2, K, 5]
            prediction = self.model(input_tensor).squeeze(0).cpu().numpy()
            
        predicted_stars = []
        cell_size = 2 # Fixed by current architecture
        grid_h, grid_w, K, _ = prediction.shape
        
        for y in range(grid_h):
            for x in range(grid_w):
                for k in range(K):
                    p, dx, dy, m, c = prediction[y, x, k]
                    if p > threshold:
                        global_x = (x * cell_size) + dx
                        global_y = (y * cell_size) + dy
                        predicted_stars.append((global_x, global_y, m, c, p))
                        
        return predicted_stars

    def visualize(self, image_tensor, true_catalogue, predicted_stars, threshold, output_path="inference_comparison.png"):
        img = image_tensor.squeeze().numpy()
        img_min = img.min()
        if img_min <= 0: img = img - img_min + 1e-3

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        ax1.imshow(img, cmap='inferno', origin='lower', norm=LogNorm())
        ax1.set_title(f"Ground Truth ({len(true_catalogue)} stars)")
        for star in true_catalogue:
            x, y = star[0], star[1]
            ax1.plot(x, y, 'g+', markersize=10, alpha=0.6)
        
        ax2.imshow(img, cmap='inferno', origin='lower', norm=LogNorm())
        ax2.set_title(f"Model Predictions ({len(predicted_stars)} stars, p > {threshold})")
        for x, y, m, c, p in predicted_stars:
            ax2.plot(x, y, 'r+', markersize=10, alpha=min(1.0, p))

        plt.suptitle("Dense Grid Model Inference: Roman Point Source Pipeline", fontsize=16)
        plt.savefig(output_path)
        print(f"Comparison saved to {output_path}")
