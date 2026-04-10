#!/usr/bin/env python3
import os
import glob
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from roman_datamodels import datamodels
import argparse
from tqdm import tqdm

def paczynski_magnification(t, t0, tE, u0):
    """ Theoretical magnification formula. """
    u = np.sqrt(u0**2 + ((t - t0) / tE)**2)
    return (u**2 + 2) / (u * np.sqrt(u**2 + 4) + 1e-9)

def main():
    parser = argparse.ArgumentParser(description="Animate microlensing events from ASDF stack.")
    parser.add_argument("--indir", default="data/microlensing_mag23_strict", help="Directory containing ASDF and JSON files")
    parser.add_argument("--out", default="microlensing_animation.mp4", help="Output animation file")
    parser.add_argument("--event_idx", type=int, default=0, help="Index of the microlensing event to focus on")
    parser.add_argument("--cutout_size", type=int, default=50, help="Size of the cutout around the event")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second")
    parser.add_argument("--num_frames", type=int, default=None, help="Limit number of frames for testing")
    args = parser.parse_args()

    # 1. Load file list
    asdf_files = sorted(glob.glob(os.path.join(args.indir, "epoch_*.asdf")))
    if not asdf_files:
        print(f"❌ No ASDF files found in {args.indir}")
        return
    
    if args.num_frames is not None:
        asdf_files = asdf_files[:args.num_frames]
    
    print(f"🎬 Found {len(asdf_files)} epochs. Preparing animation...")

    # 2. Extract positions and pixel values for the selected event
    positions = []
    pixel_values = []
    times = []
    ml_params = {}
    
    for i, f in enumerate(tqdm(asdf_files, desc="Pre-processing light curve")):
        gt_path = f.replace(".asdf", "_gt.json")
        with open(gt_path, 'r') as f_gt:
            gt = json.load(f_gt)
        
        event = gt['events'][args.event_idx]
        px, py = event['target_x_sca'], event['target_y_sca']
        positions.append((px, py))
        
        if i == 0:
            ml_params['t0'] = event.get('t0', 'N/A')
            ml_params['tE'] = event.get('tE', 'N/A')
            ml_params['u0'] = event.get('u0', 'N/A')
            ml_params['mag_base'] = event.get('true_mag', 'N/A')
            if ml_params['u0'] != 'N/A' and ml_params['u0'] != 0:
                u0 = ml_params['u0']
                ml_params['peak_A'] = (u0**2 + 2) / (u0 * np.sqrt(u0**2 + 4))
            else:
                ml_params['peak_A'] = 'N/A'

        with datamodels.open(f) as model:
            img = model.data.copy()
            ix, iy = int(round(px)), int(round(py))
            w = 2
            if w <= ix < img.shape[1]-w and w <= iy < img.shape[0]-w:
                patch = img[iy-w:iy+w+1, ix-w:ix+w+1]
                
                # Optional but recommended: subtract a local background median 
                # so the baseline doesn't artificially inflate
                local_bg = np.median(img[iy-10:iy+10, ix-10:ix+10])
                total_flux = np.sum(patch) - (patch.size * local_bg)
                
                pixel_values.append(total_flux)
            elif 0 <= ix < img.shape[1] and 0 <= iy < img.shape[0]:
                pixel_values.append(img[iy, ix])
            else:
                pixel_values.append(0)
            
            try:
                times.append(model.meta.exposure.start_time.mjd)
            except:
                times.append(len(times))

    times = np.array(times)
    pixel_values = np.array(pixel_values)
    times -= times[0]

    # 3. Setup Animation Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(bottom=0.2)
    
    with datamodels.open(asdf_files[0]) as model:
        img0 = model.data.copy()
    
    px0, py0 = positions[0]
    half = args.cutout_size // 2
    ix0, iy0 = int(np.floor(px0)), int(np.floor(py0))
    
    x_min, x_max = ix0 - half, ix0 + half
    y_min, y_max = iy0 - half, iy0 + half
    
    def get_cutout(img, x1, x2, y1, y2):
        h, w = img.shape
        iy1, iy2 = max(0, y1), min(h, y2)
        ix1, ix2 = max(0, x1), min(w, x2)
        patch = img[iy1:iy2, ix1:ix2]
        
        if patch.shape[0] != (y2-y1) or patch.shape[1] != (x2-x1):
            full_patch = np.zeros((y2-y1, x2-x1), dtype=patch.dtype)
            py1, py2 = iy1 - y1, iy2 - y1
            px1, px2 = ix1 - x1, ix2 - x1
            full_patch[py1:py2, px1:px2] = patch
            return full_patch
        return patch

    cutout0 = get_cutout(img0, x_min, x_max, y_min, y_max)
    
    vmin = np.percentile(cutout0[cutout0 > 0], 5) if np.any(cutout0 > 0) else 1e-3
    vmax = np.percentile(cutout0, 99.9) if np.any(cutout0 > 0) else 1
    if vmin <= 0: vmin = 1e-3
    if vmax <= vmin: vmax = vmin * 10

    im_plot = ax1.imshow(cutout0, origin='lower', cmap='inferno', interpolation='nearest', 
                         norm=LogNorm(vmin=vmin, vmax=vmax))
    ax1.margins(0)
    ax1.set_title(f"Event {args.event_idx} Cutout (Log Scale)")
    
    rel_x0, rel_y0 = px0 - x_min, py0 - y_min
    
    ch_h = ax1.axhline(rel_y0, color='white', alpha=0.5, linewidth=0.5)
    ch_v = ax1.axvline(rel_x0, color='white', alpha=0.5, linewidth=0.5)
    
    circ = patches.Circle((rel_x0, rel_y0), radius=3, edgecolor='cyan', facecolor='none', linewidth=1)
    ax1.add_patch(circ)
    
    ax1.set_xlim(-0.5, args.cutout_size - 0.5)
    ax1.set_ylim(-0.5, args.cutout_size - 0.5)

    param_text = ""
    if ml_params.get('mag_base', 'N/A') != 'N/A':
        param_text += f"Baseline Mag: {ml_params['mag_base']:.2f} | "
    if ml_params.get('t0', 'N/A') != 'N/A':
        param_text += f"t0: {ml_params['t0']:.2f} | tE: {ml_params['tE']:.2f} | u0: {ml_params['u0']:.3f} | peak A: {ml_params['peak_A']:.2f}"
    else:
        param_text += "ML Parameters not in JSON (re-generate data to see them)"
    
    fig.text(0.5, 0.05, param_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    lc_plot, = ax2.plot([], [], 'b.', markersize=2, alpha=0.6, label='Peak Pixel Value')
    lc_point, = ax2.plot([], [], 'ro', markersize=5)
    ax2.set_xlim(times.min(), times.max())
    ax2.set_ylim(pixel_values.min() * 0.9, pixel_values.max() * 1.1)
    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel("Peak Pixel Value")
    ax2.set_title("Light Curve vs Theoretical A(t)")
    ax2.grid(True, alpha=0.3)

    # Secondary axis for Magnification
    ax2_mag = ax2.twinx()
    ax2_mag.set_ylabel("Magnification (A)", color='tab:orange')
    ax2_mag.tick_params(axis='y', labelcolor='tab:orange')
    
    if ml_params.get('t0', 'N/A') != 'N/A':
        t_smooth = np.linspace(times.min(), times.max(), 1000)
        a_smooth = paczynski_magnification(t_smooth, ml_params['t0'], ml_params['tE'], ml_params['u0'])
        ax2_mag.plot(t_smooth, a_smooth, 'tab:orange', alpha=0.3, linewidth=2, label='Theoretical A(t)')
        ax2_mag.set_ylim(1.0, max(a_smooth) * 1.2)
    
    mag_points, = ax2_mag.plot([], [], 'x', color='tab:orange', markersize=4, alpha=0.5, label='Actual Magnification')

    def update(frame):
        f = asdf_files[frame]
        with datamodels.open(f) as model:
            cutout = get_cutout(model.data, x_min, x_max, y_min, y_max).copy()
        
        px, py = positions[frame]
        im_plot.set_data(cutout)
        
        dx, dy = px - x_min , py - y_min 
        circ.set_center((dx, dy))
        ch_h.set_ydata([dy, dy])
        ch_v.set_xdata([dx, dx])
        
        lc_plot.set_data(times[:frame+1], pixel_values[:frame+1])
        lc_point.set_data([times[frame]], [pixel_values[frame]])
        
        # Update Actual Magnification Points
        if ml_params.get('t0', 'N/A') != 'N/A':
            a_actual = paczynski_magnification(times[:frame+1], ml_params['t0'], ml_params['tE'], ml_params['u0'])
            mag_points.set_data(times[:frame+1], a_actual)
        
        ax1.set_xlabel(f"Frame {frame} | Time {times[frame]:.2f}d\nPos: ({px:.1f}, {py:.1f})")
        return im_plot, circ, ch_h, ch_v, lc_plot, lc_point, mag_points

    print(f"📽️ Encoding animation to {args.out}...")
    ani = FuncAnimation(fig, update, frames=len(asdf_files), blit=True)
    
    from matplotlib.animation import writers
    has_ffmpeg = 'ffmpeg' in writers.list()

    try:
        if args.out.endswith(".mp4") and has_ffmpeg:
            ani.save(args.out, fps=args.fps, extra_args=['-vcodec', 'libx264'])
        elif args.out.endswith(".mp4") and not has_ffmpeg:
            print("⚠️ ffmpeg not found. Falling back to GIF.")
            args.out = args.out.replace(".mp4", ".gif")
            ani.save(args.out, writer='pillow', fps=args.fps)
        else:
            writer = 'pillow' if args.out.endswith(".gif") else None
            ani.save(args.out, writer=writer, fps=args.fps)
        print(f"✅ Animation saved to {args.out}")
    except Exception as e:
        print(f"❌ Failed to save animation: {e}")

if __name__ == "__main__":
    main()
