import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
from scipy.ndimage import map_coordinates
from castor.data.transforms import AstroSpaceTransform
from castor.constants import GLOBAL_STRETCH_SCALE

def visualize_mosaic_with_correct_snr():
    data_dir = "data/bulge_stage0_full/mosaics"
    img_path = os.path.join(data_dir, "mosaic_000_img.npy")
    cat_path = os.path.join(data_dir, "mosaic_000_catalog.parquet")
    
    if not os.path.exists(img_path):
        print(f"Error: Mosaic not found at {img_path}")
        return

    # 1. Load Data
    star_signal = np.load(img_path)
    catalog = pd.read_parquet(cat_path)
    
    exp_time, zp, sky_mag = catalog['exp_time'].iloc[0], catalog['zp'].iloc[0], catalog['sky_mag'].iloc[0]
    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (0.11**2) * exp_time
    
    # Simulate a noisy observation
    img_noisy = np.random.poisson(np.maximum(star_signal + sky_level, 0)).astype(np.float32)
    img_noisy += np.random.normal(0, 5.0, size=img_noisy.shape)
    
    chunk_median = np.median(img_noisy)
    transform = AstroSpaceTransform(stretch_scale=GLOBAL_STRETCH_SCALE)
    network_input = transform.image_to_network(img_noisy, chunk_median)

    # --- 2. Correct SNR Filtering (Matching Dataset) ---
    n_pix = 4 * np.pi * (1.5 ** 2)
    psf_peak = 1.0 / (2 * np.pi * 1.5**2)
    
    fluxes = catalog['flux'].values
    x, y = catalog['x'].values, catalog['y'].values
    
    # Sub-pixel sample background
    total_local_light = map_coordinates(star_signal, [y, x], order=1, mode='nearest')
    local_background = np.maximum(0, total_local_light - (fluxes * psf_peak))
    
    noise_var = fluxes + n_pix * (sky_level + local_background + 25.0)
    snrs = fluxes / np.sqrt(noise_var)
    
    # Identify model targets
    target_mask = snrs >= 5.0
    targets = catalog[target_mask]
    print(f"Total stars in catalog: {len(catalog):,}")
    print(f"Detectable targets:     {len(targets):,}")

    # 3. Create Plot
    fig = plt.figure(figsize=(24, 12))
    
    # Left: Full Mosaic View (Network Input)
    ax0 = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2)
    im0 = ax0.imshow(network_input, cmap='magma', vmin=np.percentile(network_input, 1), vmax=np.percentile(network_input, 99.9))
    ax0.set_title(f"Seamless Mosaic (1024x1024)\n{len(targets):,} Targets (SNR >= 5) | Median: {chunk_median:.0f}")
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    
    # Right-Top: High-Density Zoom (256x256)
    zoom_size = 256
    zx, zy = 400, 400
    ax1 = plt.subplot2grid((2, 3), (0, 2))
    crop = network_input[zy:zy+zoom_size, zx:zx+zoom_size]
    im1 = ax1.imshow(crop, cmap='magma')
    ax1.set_title("Target Selection (SNR >= 5)\nZoomed 256x256")
    
    # Overlay ONLY correctly identified targets
    zoom_targets = targets[(targets['x'] >= zx) & (targets['x'] < zx + zoom_size) & 
                           (targets['y'] >= zy) & (targets['y'] < zy + zoom_size)]
    ax1.scatter(zoom_targets['x'] - zx, zoom_targets['y'] - zy, s=30, edgecolors='cyan', facecolors='none', alpha=0.6)
    
    # Right-Bottom: LF Check (Focus on 12-24)
    ax2 = plt.subplot2grid((2, 3), (1, 2))
    hist_range = (12, 24)
    ax2.hist(catalog['mag'], bins=50, range=hist_range, color='gray', alpha=0.3, label='Full Pop')
    ax2.hist(targets['mag'], bins=50, range=hist_range, color='cyan', alpha=0.7, edgecolor='black', label='Targets')
    ax2.set_title("Target Selection LF (Mag 12-24)")
    ax2.set_xlabel("Magnitude")
    ax2.invert_xaxis()
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("full_mosaic_validation.png", dpi=150)
    print(f"✅ Full mosaic validation saved to full_mosaic_validation.png")

if __name__ == "__main__":
    visualize_mosaic_with_correct_snr()
