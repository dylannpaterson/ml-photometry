import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg') # Headless Backend
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import argparse
from castor.constants import GLOBAL_STRETCH_SCALE

def visualize_stage1_chunk(chunk_idx=0, h5_path="data/stage1_multitel/stage1_data.h5"):
    if not os.path.exists(h5_path):
        print(f"Error: HDF5 file not found at {h5_path}")
        return

    # 1. Load Data from HDF5
    with h5py.File(h5_path, 'r') as f:
        if chunk_idx >= len(f['images']):
            print(f"Error: chunk_idx {chunk_idx} out of range (total {len(f['images'])})")
            return
            
        img = f['images'][chunk_idx][:]
        target = f['targets'][chunk_idx][:]
        chunk_median = f['chunk_medians'][chunk_idx]
        meta = f['metas'][chunk_idx][:]

    exp_time = meta[0]
    zp = meta[1]
    read_noise = meta[12]
    pixel_scale = meta[7]

    # 2. Reconstruct Linear image as seen by trainer (adding back median)
    img_positive = np.maximum(img, 0)
    img_noisy = np.random.poisson(img_positive).astype(np.float32)
    img_noisy += np.random.normal(0, read_noise, size=img_noisy.shape)
    
    # We want LINEAR units (photons) but LOG scale for visualization
    linear_img = np.maximum(img_noisy, 1e-1) # Ensure positive for LogNorm

    # Background truth is stored in the last channel of targets (downsampled 64x64)
    bg_truth_linear = np.maximum(target[:, :, -1], 1e-1)
    
    # 3. Extract targets from grid for scatter overlay
    K = (target.shape[-1] - 1) // 5
    grid_size = target.shape[0]
    cell_size = img.shape[0] // grid_size
    
    star_targets = target[:, :, :-1].reshape(grid_size, grid_size, K, 5)
    p_map = star_targets[:, :, :, 0]
    dx_map = star_targets[:, :, :, 1]
    dy_map = star_targets[:, :, :, 2]
    flux_map = star_targets[:, :, :, 3]
    snr_map = star_targets[:, :, :, 4]
    
    active_mask = p_map > 0.05
    cy, cx, ck = np.where(active_mask)
    
    target_px = cx * cell_size + dx_map[active_mask]
    target_py = cy * cell_size + dy_map[active_mask]
    target_flux = flux_map[active_mask]
    target_snr = snr_map[active_mask]
    
    target_mag = zp - 2.5 * np.log10(np.maximum(target_flux / exp_time, 1e-9))
    detectable_mask = target_snr >= 5.0
    visible_mask = target_snr > 2.0

    print(f"📊 Metadata for Chunk {chunk_idx}:")
    print(f"   Pixel Scale: {pixel_scale:.3f}\" | FWHM: {meta[3]:.2f} | Read Noise: {read_noise:.2f}")
    print(f"   Exp Time: {exp_time:.1f}s | Max Extinction: {meta[13]:.2f}")
    print(f"   Active Stars in Targets: {len(target_px)}")

    # 4. Create Plot
    fig = plt.figure(figsize=(24, 12))
    
    # Main Image (Log scale)
    ax0 = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)
    vmin0, vmax0 = np.percentile(linear_img, 1), np.percentile(linear_img, 99.9)
    im0 = ax0.imshow(linear_img, cmap='magma', norm=LogNorm(vmin=max(0.1, vmin0), vmax=vmax0), origin='lower')
    ax0.set_title(f"Stage 1 Chunk {chunk_idx} (Linear photons, Log Scale)\nScale: {pixel_scale:.3f}\"/px, RN: {read_noise:.1f}")
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    # Background Truth (Log scale, independent scaling)
    ax_bg = plt.subplot2grid((2, 4), (0, 2), rowspan=1, colspan=1)
    vmin_bg, vmax_bg = np.percentile(bg_truth_linear, 1), np.percentile(bg_truth_linear, 99.9)
    im_bg = ax_bg.imshow(bg_truth_linear, cmap='magma', norm=LogNorm(vmin=max(0.1, vmin_bg), vmax=vmax_bg), origin='lower')
    ax_bg.set_title("Background Truth (Linear, Log Scale)\n(Downsampled 64x64)")
    plt.colorbar(im_bg, ax=ax_bg, fraction=0.046, pad=0.04)
    
    # Target overlay
    ax1 = plt.subplot2grid((2, 4), (1, 2))
    ax1.imshow(linear_img, cmap='magma', norm=LogNorm(vmin=max(0.1, vmin0), vmax=vmax0), origin='lower')
    ax1.set_title("Targets (SNR >= 5)\nLinear Log View")
    ax1.scatter(target_px[detectable_mask], target_py[detectable_mask], s=30, edgecolors='cyan', facecolors='none', alpha=0.6, label='SNR >= 5')
    ax1.legend(loc='upper right', fontsize='small')
    
    # Magnitude Histogram
    ax2 = plt.subplot2grid((2, 4), (0, 3), rowspan=2)
    hist_range = (12, 30)
    ax2.hist(target_mag, bins=40, range=hist_range, color='gray', alpha=0.3, label='All Active')
    ax2.hist(target_mag[visible_mask], bins=40, range=hist_range, color='orange', alpha=0.5, label='SNR > 2')
    ax2.hist(target_mag[detectable_mask], bins=40, range=hist_range, color='cyan', alpha=0.7, edgecolor='black', label='Targets (SNR >= 5)')
    
    ax2.set_title("Target Magnitude Distribution")
    ax2.set_xlabel("Magnitude")
    ax2.invert_xaxis()
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_png = f"stage1_chunk_{chunk_idx}_visualisation.png"
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"✅ Chunk visualisation saved to {output_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Stage 1 GalSim Chunk")
    parser.add_argument("--chunk_idx", type=int, default=0, help="Index of the chunk to visualize")
    parser.add_argument("--h5_path", type=str, default="data/stage1_multitel/stage1_data.h5", help="Path to HDF5 data")
    args = parser.parse_args()
    
    visualize_stage1_chunk(chunk_idx=args.chunk_idx, h5_path=args.h5_path)
