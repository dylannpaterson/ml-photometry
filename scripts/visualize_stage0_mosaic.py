import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
from scipy.ndimage import map_coordinates
from castor.data.transforms import AstroSpaceTransform
from castor.constants import GLOBAL_STRETCH_SCALE

def visualize_mosaic_optimized():
    data_dir = "data/bulge_stage0_full/mosaics"
    img_path = os.path.join(data_dir, "mosaic_000_img.npy")
    cat_path = os.path.join(data_dir, "mosaic_000_cat.npy")
    meta_path = os.path.join(data_dir, "mosaic_000_meta.npy")
    
    if not os.path.exists(img_path):
        print(f"Error: Mosaic not found at {img_path}")
        return

    # 1. Load Data (Optimized NumPy path)
    star_signal = np.load(img_path)
    structured_cat = np.load(cat_path)
    meta = np.load(meta_path)
    
    exp_time, zp, sky_mag = meta[0], meta[1], meta[2]
    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (0.11**2) * exp_time
    
    # Simulate a noisy observation (Match dataset live noise)
    img_noisy = np.random.poisson(np.maximum(star_signal + sky_level, 0)).astype(np.float32)
    img_noisy += np.random.normal(0, 5.0, size=img_noisy.shape)
    
    chunk_median = np.median(img_noisy)
    transform = AstroSpaceTransform(stretch_scale=GLOBAL_STRETCH_SCALE)
    network_input = transform.image_to_network(img_noisy, chunk_median)

    # --- 2. Correct SNR Filtering (Matching Dataset) ---
    n_pix = 4 * np.pi * (1.5 ** 2)
    psf_peak = 1.0 / (2 * np.pi * 1.5**2)
    
    fluxes = structured_cat['flux']
    x, y = structured_cat['x'], structured_cat['y']
    
    # Sub-pixel sample background from baked physics
    total_local_light = map_coordinates(star_signal, [y, x], order=1, mode='nearest')
    local_background = np.maximum(0, total_local_light - (fluxes * psf_peak))
    
    noise_var = fluxes + n_pix * (sky_level + local_background + 25.0)
    snrs = fluxes / np.sqrt(noise_var)
    
    # Identify model targets
    target_mask = snrs >= 5.0
    targets_x = x[target_mask]
    targets_y = y[target_mask]
    targets_mag = structured_cat['mag'][target_mask]
    
    print(f"Total stars in culled catalog: {len(structured_cat):,}")
    print(f"Detectable targets (SNR >= 5):  {len(targets_x):,}")

    # 3. Create Plot
    fig = plt.figure(figsize=(24, 12))
    
    # Left: Full Mosaic View (Network Input)
    ax0 = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2)
    im0 = ax0.imshow(network_input, cmap='magma', vmin=np.percentile(network_input, 1), vmax=np.percentile(network_input, 99.9))
    ax0.set_title(f"Seamless Mosaic (1024x1024)\n{len(targets_x):,} Targets (SNR >= 5) | Median: {chunk_median:.0f}")
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    
    # Right-Top: High-Density Zoom (256x256)
    zoom_size = 256
    zx, zy = 400, 400
    ax1 = plt.subplot2grid((2, 3), (0, 2))
    crop = network_input[zy:zy+zoom_size, zx:zx+zoom_size]
    im1 = ax1.imshow(crop, cmap='magma')
    ax1.set_title("Target Selection (SNR >= 5)\nZoomed 256x256")
    
    # Overlay ONLY correctly identified targets
    zoom_mask = (targets_x >= zx) & (targets_x < zx + zoom_size) & \
                (targets_y >= zy) & (targets_y < zy + zoom_size)
    ax1.scatter(targets_x[zoom_mask] - zx, targets_y[zoom_mask] - zy, s=30, edgecolors='cyan', facecolors='none', alpha=0.6)
    
    # Right-Bottom: LF Check
    ax2 = plt.subplot2grid((2, 3), (1, 2))
    hist_range = (12, 30)
    ax2.hist(structured_cat['mag'], bins=50, range=hist_range, color='gray', alpha=0.3, label='Culled Pop')
    ax2.hist(targets_mag, bins=50, range=hist_range, color='cyan', alpha=0.7, edgecolor='black', label='Targets')
    ax2.set_title("Target Selection LF")
    ax2.set_xlabel("Magnitude")
    ax2.invert_xaxis()
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("optimized_mosaic_validation.png", dpi=150)
    print(f"✅ Optimized mosaic validation saved to optimized_mosaic_validation.png")

if __name__ == "__main__":
    visualize_mosaic_optimized()
