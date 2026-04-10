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

def render_bilinear_mosaic(px, py, fluxes, mosaic_size, psf_1x):
    """ High-fidelity bi-linear rendering logic. """
    full_image_grid = np.zeros((mosaic_size, mosaic_size), dtype=np.float32)
    x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
    dx, dy = px - x0, py - y0
    
    valid = (x0 >= 0) & (x0 < mosaic_size-1) & (y0 >= 0) & (y0 < mosaic_size-1)
    
    w00 = (1 - dx) * (1 - dy)
    w10 = dx * (1 - dy)
    w01 = (1 - dx) * dy
    w11 = dx * dy
    
    def paint_flux(grid, x, y, w, f, mask):
        flat_indices = y[mask] * mosaic_size + x[mask]
        grid.flat += np.bincount(flat_indices, weights=f[mask] * w[mask], minlength=grid.size)

    paint_flux(full_image_grid, x0, y0, w00, fluxes, valid)
    paint_flux(full_image_grid, x0 + 1, y0, w10, fluxes, valid)
    paint_flux(full_image_grid, x0, y0 + 1, w01, fluxes, valid)
    paint_flux(full_image_grid, x0 + 1, y0 + 1, w11, fluxes, valid)

    return fftconvolve(full_image_grid, psf_1x, mode='same')

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
    
    original_wcs = ris_wcs.get_wcs(template.meta)
    
    from romanisim import parameters
    valid_tables = list(parameters.read_pattern.keys())
    ma_table = 10 if 10 in valid_tables else valid_tables[0]

    # 2. PSF Library Setup
    if os.path.exists(args.psf_library):
        master_data = torch.load(args.psf_library, map_location='cpu', weights_only=False)
        if 'kb_array' in master_data:
            master_psf_library = master_data['kb_array']
        else:
            master_psf_library = master_data['mean_psf'][np.newaxis, ...]
    else:
        master_psf_library = generate_field_realistic_psf_library(num_psfs=100, grid_size=SHAPE_SIZE, oversample=O)
    
    # Select a single Physical PSF
    repr_idx = np.random.randint(0, len(master_psf_library))
    repr_psf_4x = master_psf_library[repr_idx] 

    # Correct Center (S-1)/2.0 and binning
    psf_1x = repr_psf_4x.reshape(SHAPE_SIZE, O, SHAPE_SIZE, O).mean(axis=(1, 3))
    psf_1x /= (np.sum(psf_1x) + 1e-9)

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
    
    # Select Multiple Microlensing Events
    # Filter for stars brighter than 25.0 (numerically <= 25.0
    # Sampling randomly from this set naturally follows the Bulge Luminosity Function distribution
    candidates = np.where(mags <= 25.0)[0]
    if len(candidates) < args.num_events:
        event_indices = np.random.choice(len(mags), args.num_events, replace=False)
    else:
        event_indices = np.random.choice(candidates, args.num_events, replace=False)
    
    total_duration = times.max() - times.min()
    events = []
    for idx in event_indices:
        # Tight constraint: tE < 0.125 * total_duration (8*tE fits in the series)
        # This provides roughly 3*tE of baseline on BOTH sides if centered.
        upper_limit = min(args.tE_max, 0.125 * total_duration)
        
        # Ensure lower limit is strictly less than upper limit
        lower_limit = min(args.tE_min, upper_limit * 0.5)
        
        tE = float(np.random.uniform(lower_limit, upper_limit))
        
        # Center the event with at least 2*tE margin from either edge
        t0 = float(np.random.uniform(times.min() + 2*tE, times.max() - 2*tE))
        
        events.append({
            'idx': int(idx), 't0': t0, 'tE': tE, 'u0': float(np.random.uniform(0.01, 0.5)),
            'mag_base': float(mags[idx]), 'flux_base': float(fluxes[idx])
        })
    
    # Render static background
    s_mask = np.ones(len(mags), dtype=bool); s_mask[event_indices] = False
    static_super = render_bilinear_mosaic(x_c[s_mask], y_c[s_mask], fluxes[s_mask], super_size, psf_1x)
    
    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (0.11**2) * exp_time
    pixel_scale_deg = 0.11 / 3600.0
    cos_pa, sin_pa = np.cos(np.radians(pa_aper)), np.sin(np.radians(pa_aper))

    print(f"🎬 Generating {len(times)} full-SCA epochs with {args.num_events} events...")
    for i, t in enumerate(tqdm(times)):
        curr_drift = drift_vec * (t - times[0])
        offset_y, offset_x = padding + curr_drift[1], padding + curr_drift[0]
        y0_int, x0_int = int(np.round(offset_y)), int(np.round(offset_x))
        dy_sub, dx_sub = offset_y - y0_int, offset_x - x0_int
        
        # 1. Background Shift (using order=1 for speed/consistency with bi-linear)
        epoch_img = shift(static_super[y0_int:y0_int+mosaic_size, x0_int:x0_int+mosaic_size], [-dy_sub, -dx_sub], order=1)
        
        # 2. Inject Events
        current_event_data = []
        for ev in events:
            A = paczynski_magnification(t, ev['t0'], ev['tE'], ev['u0'])
            
            # Position in SCA frame
            frame_x, frame_y = x_c[ev['idx']] - offset_x, y_c[ev['idx']] - offset_y
            
            # Render a high-res patch for the star using bi-linear logic
            patch_size = 64
            sx_int, sy_int = int(round(x_c[ev['idx']])), int(round(y_c[ev['idx']]))
            dx_patch, dy_patch = x_c[ev['idx']] - sx_int, y_c[ev['idx']] - sy_int
            
            star_patch = render_bilinear_mosaic(
                np.array([patch_size//2 + dx_patch]), 
                np.array([patch_size//2 + dy_patch]), 
                np.array([ev['flux_base']]), 
                patch_size, psf_1x
            )
            
            # Background-consistent shift
            shifted_star = shift(star_patch, [-dy_sub, -dx_sub], order=1) * A
            
            ry, rx = sy_int - y0_int, sx_int - x0_int
            y1, y2 = ry - patch_size//2, ry + patch_size//2
            x1, x2 = rx - patch_size//2, rx + patch_size//2
            
            iy1, iy2 = max(0, y1), min(mosaic_size, y2)
            ix1, ix2 = max(0, x1), min(mosaic_size, x2)
            if iy2 > iy1 and ix2 > ix1:
                epoch_img[iy1:iy2, ix1:ix2] += shifted_star[iy1-y1:iy2-y1, ix1-x1:ix2-x1]

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
