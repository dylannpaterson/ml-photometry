#!/usr/bin/env python3
import os
import torch
import numpy as np
import argparse
import time
import json
from tqdm import tqdm
from castor.cloud.config_utils import load_config
from castor.data.stage0_gaussian import (
    sample_bulge_magnitudes, 
    generate_field_realistic_psf_library, 
    _compute_eigen_psfs,
    calculate_safe_magnitude_cutoff
)
from castor.constants import GLOBAL_STRETCH_SCALE, SHAPE_SIZE, N_PCA_COMPONENTS
from scipy.signal import fftconvolve
from scipy.ndimage import shift
from roman_datamodels import datamodels
from romanisim import ris_make_utils as ris
from romanisim import wcs as ris_wcs
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
import asdf
import copy

def paczynski_magnification(t, t0, tE, u0):
    """ Standard Paczynski microlensing magnification formula. """
    u = np.sqrt(u0**2 + ((t - t0) / tE)**2)
    A = (u**2 + 2) / (u * np.sqrt(u**2 + 4) + 1e-9)
    return A

def render_16phase_mosaic(px, py, fluxes, mosaic_size, psf_library, O=4):
    """ High-fidelity 16-phase rendering logic. """
    full_image = np.zeros((mosaic_size, mosaic_size), dtype=np.float32)
    x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
    dx_idx = np.clip(np.floor((px - x0) * O).astype(int), 0, O-1)
    dy_idx = np.clip(np.floor((py - y0) * O).astype(int), 0, O-1)
    valid = (x0 >= 0) & (x0 < mosaic_size) & (y0 >= 0) & (y0 < mosaic_size)
    
    for dyi in range(O):
        for dxi in range(O):
            mask = valid & (dx_idx == dxi) & (dy_idx == dyi)
            if not mask.any(): continue
            flat_indices = y0[mask] * mosaic_size + x0[mask]
            grid = np.bincount(flat_indices, weights=fluxes[mask], minlength=mosaic_size*mosaic_size).reshape(mosaic_size, mosaic_size)
            full_image += fftconvolve(grid, psf_library[dyi, dxi], mode='same')
    return full_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--outdir", default="data/microlensing_stack_strict")
    parser.add_argument("--mosaic_size", type=int, default=4088)
    parser.add_argument("--target_mag", type=float, default=21.0)
    parser.add_argument("--num_events", type=int, default=5, help="Number of simultaneous microlensing events")
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--cadence", type=float, default=1.0)
    parser.add_argument("--t0", type=float, default=36.0)
    parser.add_argument("--tE_min", type=float, default=5.0)
    parser.add_argument("--tE_max", type=float, default=50.0)
    parser.add_argument("--drift_rate", type=float, default=0.5)
    parser.add_argument("--drift_angle", type=float, default=45.0)
    parser.add_argument("--exp_time", type=float, default=66.0)
    parser.add_argument("--template", default="bulge_empty.asdf")
    parser.add_argument("--psf_library", default="master_psf_library.pt")
    
    args = parser.parse_args()
    config = load_config(args.config); data_cfg = config["data_params"]; os.makedirs(args.outdir, exist_ok=True)
    mosaic_size = args.mosaic_size; O = 4; zp, sky_mag = 26.5, 22.0
    read_noise = data_cfg["physics_params"].get("read_noise", 5.0)
    exp_time = args.exp_time
    mag_limit = calculate_safe_magnitude_cutoff(exp_time, zp, sky_mag, snr_cutoff=1.0)

    # 1. Load Header Template
    if not os.path.exists(args.template):
        print(f"⚠️ Template {args.template} not found. Regenerating...")
        os.system("PYTHONPATH=. python3 scripts/generate_bulge_exposure.py")
    
    template = datamodels.open(args.template)
    ra_center, dec_center = template.meta.pointing.ra_v1, template.meta.pointing.dec_v1
    pa_aper, obs_time_base = template.meta.wcsinfo.roll_ref, Time(template.meta.exposure.start_time)
    
    # Extract the pristine gWCS object from the template ONCE before the loop
    original_wcs = ris_wcs.get_wcs(template.meta)
    center_x, center_y = mosaic_size / 2.0, mosaic_size / 2.0
    
    from romanisim import parameters
    valid_tables = list(parameters.read_pattern.keys())
    ma_table = 10 if 10 in valid_tables else valid_tables[0]

    # 2. PSF Library Setup
    if os.path.exists(args.psf_library):
        master_data = torch.load(args.psf_library, map_location='cpu', weights_only=False)
        if isinstance(master_data, dict): master_psf_data = (master_data['eigen_psfs'], master_data['weights_lib'], master_data['mean_psf'])
        else: master_psf_data = master_data
    else:
        kb_array = generate_field_realistic_psf_library(num_psfs=100, grid_size=SHAPE_SIZE, oversample=O)
        master_psf_data = _compute_eigen_psfs(kb_array, n_components=N_PCA_COMPONENTS)
    
    eigen_psfs, psf_weights_lib, mean_psf = master_psf_data
    repr_weights = psf_weights_lib[np.random.randint(0, len(psf_weights_lib))]
    
    # Reconstruct 4x representative PSF
    if mean_psf.shape[0] == SHAPE_SIZE:
        from scipy.ndimage import zoom
        mean_psf_4x = zoom(mean_psf, O, order=3)
        eigen_psfs_4x = np.array([zoom(e, O, order=3) for e in eigen_psfs])
    else: mean_psf_4x, eigen_psfs_4x = mean_psf, eigen_psfs

    repr_psf_4x = np.maximum(0, mean_psf_4x + np.tensordot(repr_weights, eigen_psfs_4x, axes=1))
    repr_psf_4x /= (repr_psf_4x.sum() + 1e-9)

    # Apply Jitter
    s_jit, q_jit, t_jit = np.random.normal(0.127, 0.01), np.random.uniform(0.8, 1.0), np.random.uniform(0, np.pi)
    kh = (SHAPE_SIZE * O) // 2; gy, gx = np.meshgrid(np.arange(SHAPE_SIZE*O) - kh, np.arange(SHAPE_SIZE*O) - kh, indexing='ij')
    cos, sin = np.cos(t_jit), np.sin(t_jit); gxp, gyp = gx * cos + gy * sin, -gx * sin + gy * cos
    jitter_k = np.exp(-(gxp**2 / (2 * (s_jit*O)**2) + gyp**2 / (2 * (s_jit*O * q_jit)**2)))
    repr_psf_jit_4x = fftconvolve(repr_psf_4x, jitter_k / (jitter_k.sum() + 1e-9), mode='same')

    # Bin phases
    psf_library = np.zeros((O, O, SHAPE_SIZE, SHAPE_SIZE), dtype=np.float32)
    padded_psf = np.pad(repr_psf_jit_4x, ((0, O), (0, O)))
    for dy_idx in range(O):
        for dx_idx in range(O):
            window = padded_psf[dy_idx : dy_idx + SHAPE_SIZE*O, dx_idx : dx_idx + SHAPE_SIZE*O]
            psf_library[dy_idx, dx_idx] = window.reshape(SHAPE_SIZE, O, SHAPE_SIZE, O).mean(axis=(1, 3))

    # 3. Super-Mosaic Setup
    cadence_days = args.cadence / 24.0
    times = (args.t0 - (args.num_epochs // 2) * cadence_days) + np.arange(args.num_epochs) * cadence_days
    drift_vec = np.array([np.cos(np.radians(args.drift_angle)), np.sin(np.radians(args.drift_angle))]) * args.drift_rate
    padding = int(args.drift_rate * (times.max() - times.min())) + 20
    super_size = mosaic_size + padding * 2

    print(f"🌌 Rendering Super-SCA ({super_size}x{super_size})...")
    n_stars = int(np.random.uniform(data_cfg['min_stars'], data_cfg['max_stars']) * (super_size / 256)**2)
    mags = sample_bulge_magnitudes(n_stars, 15.5, 0.35, 10.0, m_min=12.0, m_max=32.0, gamma=0.3)
    fluxes = exp_time * (10 ** (-0.4 * (mags - zp)))
    x_c, y_c = np.random.uniform(0, super_size, len(mags)), np.random.uniform(0, super_size, len(mags))
    
    # Select Multiple Microlensing Events across a range of magnitudes
    target_mag_min, target_mag_max = args.target_mag - 3.0, args.target_mag + 3.0
    candidates = np.where((mags >= target_mag_min) & (mags <= target_mag_max))[0]
    
    if len(candidates) < args.num_events:
        event_indices = np.random.choice(len(mags), args.num_events, replace=False)
    else:
        # Try to pick events across the magnitude range
        candidate_mags = mags[candidates]
        bins = np.linspace(target_mag_min, target_mag_max, args.num_events + 1)
        event_indices = []
        for j in range(args.num_events):
            in_bin = candidates[(candidate_mags >= bins[j]) & (candidate_mags < bins[j+1])]
            if len(in_bin) > 0:
                event_indices.append(np.random.choice(in_bin))
            else:
                event_indices.append(np.random.choice(candidates))
    
    total_duration = times.max() - times.min()
    events = []
    for idx in event_indices:
        # 1. Significant baseline constraint: 2 * tE < 0.5 * total_duration => tE < 0.25 * total_duration
        max_tE = min(args.tE_max, 0.25 * total_duration)
        min_tE = min(args.tE_min, max_tE * 0.5) # Ensure min is sane
        tE = float(np.random.uniform(min_tE, max_tE))
        
        # 2. Well-covered constraint: t0 - tE > times.min() and t0 + tE < times.max()
        # => times.min() + tE < t0 < times.max() - tE
        t0 = float(np.random.uniform(times.min() + tE, times.max() - tE))
        
        events.append({
            'idx': int(idx),
            't0': t0,
            'tE': tE,
            'u0': float(np.random.uniform(0.01, 0.5)),
            'mag_base': float(mags[idx]),
            'flux_base': float(fluxes[idx])
        })
    
    # Render static background (excluding event stars)
    s_mask = np.ones(len(mags), dtype=bool); s_mask[event_indices] = False
    static_super = render_16phase_mosaic(x_c[s_mask], y_c[s_mask], fluxes[s_mask], super_size, psf_library, O=O)
    
    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (0.11**2) * exp_time
    pixel_scale_deg = 0.11 / 3600.0
    cos_pa, sin_pa = np.cos(np.radians(pa_aper)), np.sin(np.radians(pa_aper))

    print(f"🎬 Generating {len(times)} full-SCA epochs with {args.num_events} events...")
    for i, t in enumerate(tqdm(times)):
        curr_drift = drift_vec * (t - times[0])
        offset_y, offset_x = padding + curr_drift[1], padding + curr_drift[0]
        y0_int, x0_int = int(np.round(offset_y)), int(np.round(offset_x))
        dy_sub, dx_sub = offset_y - y0_int, offset_x - x0_int
        
        # 1. Clean background
        epoch_img = shift(static_super[y0_int:y0_int+mosaic_size, x0_int:x0_int+mosaic_size], [-dy_sub, -dx_sub], order=3)
        
        # 2. Inject Events
        current_event_data = []
        for ev in events:
            A = paczynski_magnification(t, ev['t0'], ev['tE'], ev['u0'])
            
            # Use same spline-shift logic as background for the event star
            # 1. Calculate precise position in SCA frame
            frame_x, frame_y = x_c[ev['idx']] - offset_x, y_c[ev['idx']] - offset_y
            
            # 2. Render a high-res patch for the star once (at magnification 1.0)
            # Or just render it in the frame every time but we want to avoid aliasing differences
            # Option 3 says: Render a "static" baseline version and shift it.
            # We'll render a 64x64 patch centered on the star's super-mosaic position.
            # But since magnification changes, we only render the PSF shape.
            
            patch_size = 64
            # We render it at the nearest integer pixel in the super-mosaic to keep PSF centered
            sx_int, sy_int = int(round(x_c[ev['idx']])), int(round(y_c[ev['idx']]))
            # Position relative to patch center
            dx_patch, dy_patch = x_c[ev['idx']] - sx_int, y_c[ev['idx']] - sy_int
            
            # Render the PSF into a small patch
            star_patch = render_16phase_mosaic(
                np.array([patch_size//2 + dx_patch]), 
                np.array([patch_size//2 + dy_patch]), 
                np.array([ev['flux_base']]), 
                patch_size, psf_library, O=O
            )
            
            # Shift the patch by the exact same sub-pixel amount as the background
            shifted_star = shift(star_patch, [-dy_sub, -dx_sub], order=3) * A
            
            # Add to epoch image at the correct integer location
            # Note: y0_int, x0_int are the top-left of the mosaic in super-mosaic
            # Star is at sx_int, sy_int. Relative to mosaic top-left:
            ry, rx = sy_int - y0_int, sx_int - x0_int
            
            y1, y2 = ry - patch_size//2, ry + patch_size//2
            x1, x2 = rx - patch_size//2, rx + patch_size//2
            
            # Handle bounds
            iy1, iy2 = max(0, y1), min(mosaic_size, y2)
            ix1, ix2 = max(0, x1), min(mosaic_size, x2)
            py1, py2 = iy1 - y1, iy2 - y1
            px1, px2 = ix1 - x1, ix2 - x1
            
            if iy2 > iy1 and ix2 > ix1:
                epoch_img[iy1:iy2, ix1:ix2] += shifted_star[py1:py2, px1:px2]

            current_event_data.append({
                'target_x_sca': float(frame_x), 'target_y_sca': float(frame_y),
                'magnification': float(A), 'true_mag': float(ev['mag_base']),
                't0': float(ev['t0'] - times[0]), 'tE': float(ev['tE']), 'u0': float(ev['u0'])
            })
        
        noisy = np.random.poisson(np.maximum(0, epoch_img + sky_level)).astype(np.float32)
        noisy += np.random.normal(0, read_noise, noisy.shape)
        
        # 3. Precise WCS
        dra = -(curr_drift[0] * pixel_scale_deg) / np.cos(np.radians(dec_center))
        ddec = (curr_drift[1] * pixel_scale_deg)
        dra_rot, ddec_rot = dra * cos_pa - ddec * sin_pa, dra * sin_pa + ddec * cos_pa
        epoch_ra, epoch_dec = ra_center + dra_rot, dec_center + ddec_rot
        
        # 4. Strict Model Creation
        model = template.copy()
        model.data = noisy.astype(np.float32)
        model.dq = np.zeros_like(noisy, dtype=np.uint32)
        model.err = np.full_like(noisy, read_noise, dtype=np.float16)
        
        model.meta.exposure.start_time = obs_time_base + (t * u.day)
        model.meta.exposure.exposure_time = exp_time
        ris_wcs.fill_in_parameters(model.meta, SkyCoord(epoch_ra, epoch_dec, unit='deg', frame='icrs'), boresight=False, pa_aper=pa_aper)
        
        # FIX: Extract the actual gWCS object from the romanisim wrapper
        model.meta.wcs = ris_wcs.get_wcs(model.meta).wcs

        output_path = os.path.join(args.outdir, f"epoch_{i:04d}.asdf")
        model.save(output_path)
        
        # 5. Save Multi-Event Ground Truth Sidecar
        gt = {'events': current_event_data, 'target_ra': float(epoch_ra), 'target_dec': float(epoch_dec)}
        with open(output_path.replace(".asdf", "_gt.json"), 'w') as f: json.dump(gt, f)
    
    print(f"✨ Success! Multi-event SCA series ready in: {args.outdir}")

if __name__ == "__main__": main()
