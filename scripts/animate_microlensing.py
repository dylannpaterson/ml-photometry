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
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

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

    # Photometry Parameters (Using smaller r=2.0 for crowded fields)
    r_ap = 2.0
    r_in = 10.0
    r_out = 15.0
    ap_correction = 0.7100 # Enclosed energy fraction at r=2.0

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
    net_fluxes = []
    raw_net_fluxes = []
    total_ap_fluxes = []
    bkg_ap_fluxes = []
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
            # RA and DEC are at the root of the JSON
            ml_params['ra'] = gt.get('target_ra', 'N/A')
            ml_params['dec'] = gt.get('target_dec', 'N/A')
            if ml_params['u0'] != 'N/A' and ml_params['u0'] != 0:
                u0 = ml_params['u0']
                ml_params['peak_A'] = (u0**2 + 2) / (u0 * np.sqrt(u0**2 + 4))
            else:
                ml_params['peak_A'] = 'N/A'

        with datamodels.open(f) as model:
            img = model.data.copy()
            pos = [(px, py)]
            aperture = CircularAperture(pos, r=r_ap)
            annulus = CircularAnnulus(pos, r_in=r_in, r_out=r_out)
            
            # Aperture photometry
            phot_table = aperture_photometry(img, aperture)
            
            # Robust background from annulus
            annulus_masks = annulus.to_mask(method='center')
            annulus_data = annulus_masks[0].get_values(img)
            
            if annulus_data is not None and annulus_data.size > 0:
                _, bkg_median, _ = sigma_clipped_stats(annulus_data, sigma=3.0)
                bkg_val = bkg_median
            else:
                bkg_val = 0
                
            actual_aperture_area = aperture.area_overlap(img)
            ap_total = phot_table['aperture_sum'][0]
            ap_bkg = bkg_val * actual_aperture_area
            
            total_ap_fluxes.append(ap_total)
            bkg_ap_fluxes.append(ap_bkg)
            raw_net = ap_total - ap_bkg
            raw_net_fluxes.append(raw_net)
            net_fluxes.append(raw_net / ap_correction)
            
            try:
                times.append(model.meta.exposure.start_time.mjd)
            except:
                times.append(len(times))

    times = np.array(times)
    times -= times[0]
    net_fluxes = np.array(net_fluxes)
    raw_net_fluxes = np.array(raw_net_fluxes)
    total_ap_fluxes = np.array(total_ap_fluxes)
    bkg_ap_fluxes = np.array(bkg_ap_fluxes)

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
        iy1, iy2 = int(max(0, min(h, y1))), int(max(0, min(h, y2)))
        ix1, ix2 = int(max(0, min(w, x1))), int(max(0, min(w, x2)))
        patch = img[iy1:iy2, ix1:ix2]
        tw, th = int(x2 - x1), int(y2 - y1)
        if patch.shape[0] != th or patch.shape[1] != tw:
            full_patch = np.zeros((th, tw), dtype=patch.dtype)
            py1, py2 = int(iy1 - y1), int(iy2 - y1)
            px1, px2 = int(ix1 - x1), int(ix2 - x1)
            if py2 > py1 and px2 > px1 and patch.size > 0:
                full_patch[py1:py2, px1:px2] = patch
            return full_patch
        return patch

    cutout0 = get_cutout(img0, x_min, x_max, y_min, y_max)
    vmin = np.percentile(cutout0[cutout0 > 0], 5) if np.any(cutout0 > 0) else 1e-3
    vmax = np.percentile(cutout0, 99.9) if np.any(cutout0 > 0) else 1
    if vmin <= 0: vmin = 1e-3
    if vmax <= vmin: vmax = vmin * 10

    im_plot = ax1.imshow(cutout0, origin='lower', cmap='inferno', interpolation='nearest', 
                         norm=LogNorm(vmin=vmin, vmax=vmax),
                         extent=[-0.5, args.cutout_size - 0.5, -0.5, args.cutout_size - 0.5])
    ax1.set_title(f"Event {args.event_idx} Cutout")
    
    rel_x0, rel_y0 = px0 - x_min, py0 - y_min
    ap_patch = patches.Circle((rel_x0, rel_y0), r_ap, edgecolor='cyan', facecolor='none', linewidth=1.5, label='Aperture')
    an_in_patch = patches.Circle((rel_x0, rel_y0), r_in, edgecolor='white', facecolor='none', linewidth=1, linestyle='--', alpha=0.7)
    an_out_patch = patches.Circle((rel_x0, rel_y0), r_out, edgecolor='white', facecolor='none', linewidth=1, linestyle='--', alpha=0.7)
    ax1.add_patch(ap_patch); ax1.add_patch(an_in_patch); ax1.add_patch(an_out_patch)
    ax1.set_xlim(-0.5, args.cutout_size - 0.5); ax1.set_ylim(-0.5, args.cutout_size - 0.5)

    param_text = ""
    if ml_params.get('ra', 'N/A') != 'N/A':
        param_text += f"RA: {ml_params['ra']:.5f} | DEC: {ml_params['dec']:.5f} | "
    if ml_params.get('mag_base', 'N/A') != 'N/A':
        param_text += f"Baseline Mag: {ml_params['mag_base']:.2f} | "
    if ml_params.get('t0', 'N/A') != 'N/A':
        param_text += f"t0: {ml_params['t0']:.2f} | tE: {ml_params['tE']:.2f} | u0: {ml_params['u0']:.3f} | peak A: {ml_params['peak_A']:.2f}"
    else:
        param_text += "ML Parameters not in JSON (re-generate data to see them)"
    
    fig.text(0.5, 0.05, param_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    # Plot lightcurve components (initialize as empty for animation)
    lc_total, = ax2.plot([], [], 'k.', markersize=3, alpha=0.3, label='Total in Circle')
    lc_bkg, = ax2.plot([], [], 'r.', markersize=3, alpha=0.3, label='Est. Background')
    lc_raw_net, = ax2.plot([], [], 'g.', markersize=3, alpha=0.3, label='Raw Net (Inside Circle)')
    lc_net, = ax2.plot([], [], 'b.', markersize=4, label='Corrected Net Flux')
    lc_point, = ax2.plot([], [], 'ro', markersize=6)
    
    ax2.set_xlim(times.min(), times.max())
    # Adjust ylim to see both background and net flux clearly
    ax2.set_ylim(min(net_fluxes.min(), bkg_ap_fluxes.min()) * 0.8, max(total_ap_fluxes.max(), net_fluxes.max()) * 1.1)
    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel("Flux (e-/s)")
    ax2.legend(loc='upper left', fontsize='x-small')
    ax2.grid(True, alpha=0.2)

    ax2_mag = ax2.twinx()
    ax2_mag.set_ylabel("Magnification (A)", color='tab:orange')
    if ml_params.get('t0', 'N/A') != 'N/A':
        t_smooth = np.linspace(times.min(), times.max(), 1000)
        a_smooth = paczynski_magnification(t_smooth, ml_params['t0'], ml_params['tE'], ml_params['u0'])
        ax2_mag.plot(t_smooth, a_smooth, 'tab:orange', alpha=0.2, linewidth=3)
        ax2_mag.set_ylim(1.0, max(a_smooth) * 1.2)
    mag_points, = ax2_mag.plot([], [], 'x', color='tab:orange', markersize=4, alpha=0.6)

    def update(frame):
        f = asdf_files[frame]
        with datamodels.open(f) as model:
            cutout = get_cutout(model.data, x_min, x_max, y_min, y_max).copy()
        px, py = positions[frame]
        im_plot.set_data(cutout)
        dx, dy = px - x_min , py - y_min 
        ap_patch.set_center((dx, dy)); an_in_patch.set_center((dx, dy)); an_out_patch.set_center((dx, dy))
        
        # Update all lightcurve components
        lc_total.set_data(times[:frame+1], total_ap_fluxes[:frame+1])
        lc_bkg.set_data(times[:frame+1], bkg_ap_fluxes[:frame+1])
        lc_raw_net.set_data(times[:frame+1], raw_net_fluxes[:frame+1])
        lc_net.set_data(times[:frame+1], net_fluxes[:frame+1])
        
        lc_point.set_data([times[frame]], [net_fluxes[frame]])
        if ml_params.get('t0', 'N/A') != 'N/A':
            a_actual = paczynski_magnification(times[:frame+1], ml_params['t0'], ml_params['tE'], ml_params['u0'])
            mag_points.set_data(times[:frame+1], a_actual)
        
        ax1.set_xlabel(f"Frame {frame} | Time {times[frame]:.2f}d")
        return im_plot, ap_patch, an_in_patch, an_out_patch, lc_total, lc_bkg, lc_raw_net, lc_net, lc_point, mag_points

    print(f"📽️ Encoding animation to {args.out}...")
    ani = FuncAnimation(fig, update, frames=len(asdf_files), blit=True)
    
    from matplotlib.animation import writers
    has_ffmpeg = 'ffmpeg' in writers.list()

    try:
        if args.out.endswith(".mp4") and not has_ffmpeg:
            print("⚠️ ffmpeg not found. Falling back to .gif extension and Pillow writer.")
            args.out = args.out.replace(".mp4", ".gif")
            ani.save(args.out, fps=args.fps, writer='pillow')
        elif args.out.endswith(".gif"):
            ani.save(args.out, fps=args.fps, writer='pillow')
        else:
            # Default behavior (will use ffmpeg for .mp4 if available)
            ani.save(args.out, fps=args.fps)
        print(f"✅ Animation saved to {args.out}")
    except Exception as e:
        print(f"❌ Failed to save animation: {e}")

if __name__ == "__main__":
    main()
