import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg') # Headless Backend
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import argparse
from castor.data.transforms import AstroSpaceTransform
from castor.constants import GLOBAL_STRETCH_SCALE

def visualize_stage0_chunk(chunk_idx=0, h5_path="data/stage0_test_batch/stage0_train.h5"):
    if not os.path.exists(h5_path):
        print(f"Error: HDF5 file not found at {h5_path}")
        return

    # 1. Load Data from HDF5
    with h5py.File(h5_path, 'r') as f:
        if chunk_idx >= len(f['images']):
            print(f"Error: chunk_idx {chunk_idx} out of range (total {len(f['images'])})")
            return
            
        img = f['images'][chunk_idx] # (256, 256)
        target = f['targets'][chunk_idx] # (64, 64, K*5+1)
        chunk_median = f['chunk_medians'][chunk_idx]
        meta = f['metas'][chunk_idx]

    exp_time, zp, sky_mag = meta[0], meta[1], meta[2]
    fwhm = meta[3]
    # In new format, read_noise is meta[12]
    read_noise = meta[12] if len(meta) > 12 else 5.0
    
    # 2. Arcsinh Stretch (Network Input Style)
    # Reconstruct original image (signal + background)
    linear_img = img + chunk_median
    img_noisy = np.random.poisson(np.maximum(linear_img, 0)).astype(np.float32)
    img_noisy += np.random.normal(0, read_noise, size=img_noisy.shape)
    
    transform = AstroSpaceTransform(stretch_scale=GLOBAL_STRETCH_SCALE)
    network_input = transform.image_to_network(img_noisy, chunk_median)
    
    # Background truth is already in network space in the HDF5 target
    bg_stretched = target[:, :, -1]
    
    # 3. Extract targets from grid
    K = (target.shape[-1] - 1) // 5
    grid_size = target.shape[0]
    cell_size = img.shape[0] // grid_size
    max_cap = (grid_size ** 2) * K
    
    star_targets = target[:, :, :-1].reshape(grid_size, grid_size, K, 5)
    p_map = star_targets[:, :, :, 0]
    dx_map = star_targets[:, :, :, 1]
    dy_map = star_targets[:, :, :, 2]
    flux_map = star_targets[:, :, :, 3]
    snr_map = star_targets[:, :, :, 4]
    
    active_mask = p_map > 0.05
    actual_count = np.sum(active_mask)
    actual_occ = actual_count / max_cap
    cy, cx, ck = np.where(active_mask)
    
    targets_x = cx * cell_size + dx_map[active_mask]
    targets_y = cy * cell_size + dy_map[active_mask]
    targets_flux = flux_map[active_mask]
    targets_snr = snr_map[active_mask]
    
    targets_mag = zp - 2.5 * np.log10(np.maximum(targets_flux / exp_time, 1e-9))
    detectable_mask = targets_snr >= 5.0
    visible_mask = targets_snr > 2.0

    print(f"📊 Metadata for Stage 0 Chunk {chunk_idx}:")
    print(f"   Actual Occupancy: {actual_occ*100:.1f}% | Active Stars: {actual_count}")
    print(f"   Exp Time: {exp_time:.1f}s | ZP: {zp:.1f}")

    # 4. Create Plot (Matching Mosaic Plot Exactly)
    fig = plt.figure(figsize=(24, 12))
    
    # ax0: Full Image (Arcsinh)
    ax0 = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)
    im0 = ax0.imshow(network_input, cmap='magma', vmin=np.percentile(network_input, 1), vmax=np.percentile(network_input, 99.9), origin='lower')
    ax0.set_title(f"Stage 0 Chunk {chunk_idx} ({img.shape[1]}x{img.shape[0]})\n{np.sum(detectable_mask):,} Targets (SNR >= 5)")
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    # ax_bg: Background Map
    ax_bg = plt.subplot2grid((2, 4), (0, 2), rowspan=1, colspan=1)
    im_bg = ax_bg.imshow(bg_stretched, cmap='magma', vmin=np.percentile(bg_stretched, 1), vmax=np.percentile(bg_stretched, 99.9), origin='lower')
    ax_bg.set_title("Truth Background Map\n(Unresolved Stars)")
    plt.colorbar(im_bg, ax=ax_bg, fraction=0.046, pad=0.04)
    
    # ax1: Zoomed Target Selection
    zoom_size = 128
    zx, zy = img.shape[1]//2 - zoom_size//2, img.shape[0]//2 - zoom_size//2
    ax1 = plt.subplot2grid((2, 4), (1, 2))
    crop = network_input[zy:zy+zoom_size, zx:zx+zoom_size]
    ax1.imshow(crop, cmap='magma', origin='lower')
    ax1.set_title(f"Target Selection (SNR >= 5)\nZoomed {zoom_size}x{zoom_size}")
    
    zoom_mask = (targets_x >= zx) & (targets_x < zx + zoom_size) & \
                (targets_y >= zy) & (targets_y < zy + zoom_size) & detectable_mask
    ax1.scatter(targets_x[zoom_mask] - zx, targets_y[zoom_mask] - zy, s=30, edgecolors='cyan', facecolors='none', alpha=0.6)
    
    # ax2: Histogram
    ax2 = plt.subplot2grid((2, 4), (0, 3), rowspan=2)
    hist_range = (12, 30)
    ax2.hist(targets_mag, bins=50, range=hist_range, color='gray', alpha=0.3, label='Culled Pop')
    ax2.hist(targets_mag[visible_mask], bins=50, range=hist_range, color='orange', alpha=0.5, label='SNR > 2')
    ax2.hist(targets_mag[detectable_mask], bins=50, range=hist_range, color='cyan', alpha=0.7, edgecolor='black', label='Targets (SNR >= 5)')
    
    ax2.set_title("Target Selection LF")
    ax2.set_xlabel("Magnitude")
    ax2.invert_xaxis()
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_png = f"stage0_chunk_{chunk_idx}_visualisation.png"
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"✅ Chunk visualisation saved to {output_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Stage 0 Chunk")
    parser.add_argument("--chunk_idx", type=int, default=0, help="Index of the chunk to visualize")
    parser.add_argument("--h5_path", type=str, default="data/stage0_test_batch/stage0_train.h5", help="Path to HDF5 data")
    args = parser.parse_args()
    
    visualize_stage0_chunk(chunk_idx=args.chunk_idx, h5_path=args.h5_path)
