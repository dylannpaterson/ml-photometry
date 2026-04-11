#!/usr/bin/env python3
import os
import torch
import numpy as np
import argparse
import time
import json
import galsim
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
from roman_datamodels import datamodels
from romanisim import ris_make_utils as ris
from romanisim import wcs as ris_wcs
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
import asdf
import copy
import gc

def paczynski_magnification(t, t0, tE, u0):
    """ Standard Paczynski microlensing magnification formula. """
    u = np.sqrt(u0**2 + ((t - t0) / tE)**2)
    A = (u**2 + 2) / (u * np.sqrt(u**2 + 4) + 1e-9)
    return A

def render_tiled(curr_wcs, star_ra, star_dec, fluxes, mosaic_size, psf_1x, tile_size=1024):
    """ Memory-efficient rendering using tiling. """
    full_image = np.zeros((mosaic_size, mosaic_size), dtype=np.float32)
    psf_half = psf_1x.shape[0] // 2
    
    # 1. Project all stars in chunks to stay within float32 and save RAM
    n_stars = len(star_ra)
    px = np.zeros(n_stars, dtype=np.float32)
    py = np.zeros(n_stars, dtype=np.float32)
    projection_chunk = 1000000
    for i in range(0, n_stars, projection_chunk):
        end = min(i + projection_chunk, n_stars)
        px[i:end], py[i:end] = curr_wcs.radecToxy(star_ra[i:end], star_dec[i:end], units=galsim.degrees)
    
    # 2. Process in tiles
    for y0 in range(0, mosaic_size, tile_size):
        for x0 in range(0, mosaic_size, tile_size):
            y1 = min(y0 + tile_size, mosaic_size)
            x1 = min(x0 + tile_size, mosaic_size)
            
            # Tile bounds with PSF buffer
            tx0, tx1 = x0 - psf_half - 2, x1 + psf_half + 2
            ty0, ty1 = y0 - psf_half - 2, y1 + psf_half + 2
            
            # Filter stars for this tile with safety buffer for bi-linear interpolation (1px)
            mask = (px >= tx0) & (px < tx1 - 1) & (py >= ty0) & (py < ty1 - 1)
            if not np.any(mask):
                continue
                
            t_px, t_py, t_fluxes = px[mask], py[mask], fluxes[mask].astype(np.float32)
            
            # Local grid for tile (including buffer)
            tw, th = int(tx1 - tx0), int(ty1 - ty0)
            local_grid = np.zeros((th, tw), dtype=np.float32)
            
            # Relative coordinates in local grid
            lx, ly = t_px - tx0, t_py - ty0
            ix, iy = np.floor(lx).astype(np.int32), np.floor(ly).astype(np.int32)
            dx, dy = (lx - ix).astype(np.float32), (ly - iy).astype(np.float32)
            
            # Bi-linear weights
            w00, w10, w01, w11 = (1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy
            
            # Paint to local grid
            def paint(x, y, w):
                flat_idx = y * tw + x
                # Ensure we only use valid weights for valid indices
                local_grid.flat += np.bincount(flat_idx, weights=t_fluxes * w, minlength=local_grid.size)
            
            paint(ix, iy, w00)
            paint(ix+1, iy, w10)
            paint(ix, iy+1, w01)
            paint(ix+1, iy+1, w11)
            
            # Convolve local tile
            convolved = fftconvolve(local_grid, psf_1x, mode='same')
            
            # Extract central part
            gx0, gy0 = x0 - tx0, y0 - ty0
            gx1, gy1 = gx0 + (x1 - x0), gy0 + (y1 - y0)
            full_image[y0:y1, x0:x1] = convolved[gy0:gy1, gx0:gx1]
            
            del t_px, t_py, t_fluxes, mask, local_grid, lx, ly, ix, iy, dx, dy, w00, w10, w01, w11, convolved
            gc.collect()
            
    del px, py
    gc.collect()
    return full_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--outdir", default="data/microlensing_stack_strict")
    parser.add_argument("--mosaic_size", type=int, default=4088)
    parser.add_argument("--num_events", type=int, default=5, help="Number of simultaneous microlensing events")
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--cadence", type=float, default=1.0)
    parser.add_argument("--t0", type=float, default=36.0)
    parser.add_argument("--tE_min", type=float, default=5.0)
    parser.add_argument("--tE_max", type=float, default=50.0)
    parser.add_argument("--drift_rate", type=float, default=0.5, help="Drift rate in pixels per day")
    parser.add_argument("--drift_angle", type=float, default=45.0)
    parser.add_argument("--exp_time", type=float, default=66.0)
    parser.add_argument("--template", default="bulge_empty.asdf")
    parser.add_argument("--psf_library", default="master_psf_library.pt")
    
    args = parser.parse_args()
    config = load_config(args.config); data_cfg = config["data_params"]; os.makedirs(args.outdir, exist_ok=True)
    mosaic_size = args.mosaic_size; O = 4; zp, sky_mag = 26.5, 22.0
    read_noise = data_cfg["physics_params"].get("read_noise", 5.0)
    exp_time = args.exp_time
    
    if not os.path.exists(args.template):
        print(f"⚠️ Template {args.template} not found. Regenerating...")
        os.system("PYTHONPATH=. python3 scripts/generate_bulge_exposure.py")
    
    template = datamodels.open(args.template)
    pa_aper, obs_time_base = template.meta.wcsinfo.roll_ref, Time(template.meta.exposure.start_time)
    
    original_wcs = ris_wcs.get_wcs(template.meta)
    sca_center_x, sca_center_y = (mosaic_size - 1) / 2.0, (mosaic_size - 1) / 2.0
    ref_ra, ref_dec = original_wcs.xyToradec(sca_center_x, sca_center_y, units=galsim.degrees)
    print(f"📍 Sky Reference Point (SCA 1 Center): RA {ref_ra:.5f}, DEC {ref_dec:.5f}")

    if os.path.exists(args.psf_library):
        master_data = torch.load(args.psf_library, map_location='cpu', weights_only=False)
        if 'kb_array' in master_data:
            master_psf_library = master_data['kb_array']
        else:
            master_psf_library = master_data['mean_psf'][np.newaxis, ...]
    else:
        master_psf_library = generate_field_realistic_psf_library(num_psfs=100, grid_size=SHAPE_SIZE, oversample=O)
    
    repr_idx = np.random.randint(0, len(master_psf_library))
    repr_psf_4x = master_psf_library[repr_idx] 
    psf_1x = repr_psf_4x.reshape(SHAPE_SIZE, O, SHAPE_SIZE, O).mean(axis=(1, 3))
    psf_1x /= (np.sum(psf_1x) + 1e-9)

    cadence_days = args.cadence / 24.0
    times = (args.t0 - (args.num_epochs // 2) * cadence_days) + np.arange(args.num_epochs) * cadence_days
    drift_vec_px = np.array([np.cos(np.radians(args.drift_angle)), np.sin(np.radians(args.drift_angle))]) * args.drift_rate
    
    max_drift_px = args.drift_rate * (times.max() - times.min())
    padding_px = int(max_drift_px) + 100
    
    super_size = mosaic_size + padding_px * 2
    n_stars_approx = int(np.random.uniform(data_cfg['min_stars'], data_cfg['max_stars']) * (super_size / 256)**2)
    mags = sample_bulge_magnitudes(n_stars_approx, 15.5, 0.35, 10.0, m_min=12.0, m_max=32.0, gamma=0.3)
    fluxes_base = exp_time * (10 ** (-0.4 * (mags - zp)))
    
    n_total_stars = len(mags)
    x_sca_ref = np.random.uniform(-padding_px, mosaic_size + padding_px, n_total_stars)
    y_sca_ref = np.random.uniform(-padding_px, mosaic_size + padding_px, n_total_stars)
    
    star_ra, star_dec = original_wcs.xyToradec(x_sca_ref, y_sca_ref, units=galsim.degrees)
    
    dt0 = times[0] - times[0]
    curr_drift0 = drift_vec_px * dt0
    epoch_ra0, epoch_dec0 = original_wcs.xyToradec(
        sca_center_x + curr_drift0[0], 
        sca_center_y + curr_drift0[1], 
        units=galsim.degrees
    )
    
    temp_meta = template.copy().meta
    ris_wcs.fill_in_parameters(temp_meta, SkyCoord(epoch_ra0, epoch_dec0, unit='deg', frame='icrs'), boresight=False, pa_aper=pa_aper)
    wcs0 = ris_wcs.get_wcs(temp_meta)
    
    # Project in chunks to stay safe
    px0 = np.zeros(n_total_stars, dtype=np.float32)
    py0 = np.zeros(n_total_stars, dtype=np.float32)
    for i in range(0, n_total_stars, 1000000):
        end = min(i + 1000000, n_total_stars)
        px0[i:end], py0[i:end] = wcs0.radecToxy(star_ra[i:end], star_dec[i:end], units=galsim.degrees)
    
    margin = 200
    in_sca0 = (px0 >= margin) & (px0 < mosaic_size - margin) & (py0 >= margin) & (py0 < mosaic_size - margin)
    candidates = np.where((mags <= 25.0) & in_sca0)[0]
    
    if len(candidates) < args.num_events:
        print(f"⚠️ Warning: Only found {len(candidates)} suitable event candidates. Adjusting num_events.")
        event_indices = candidates
    else:
        event_indices = np.random.choice(candidates, args.num_events, replace=False)
    
    total_duration = times.max() - times.min()
    events = []
    for idx in event_indices:
        upper_limit = min(args.tE_max, total_duration * 0.2)
        lower_limit = max(args.tE_min, upper_limit * 0.1)
        tE = float(np.random.uniform(lower_limit, upper_limit))
        t0 = float(np.random.uniform(times.min() + 2*tE, times.max() - 2*tE))
        
        events.append({
            'idx': int(idx), 't0': t0, 'tE': tE, 'u0': float(np.random.uniform(0.01, 0.5)),
            'mag_base': float(mags[idx]), 'flux_base': float(fluxes_base[idx]),
            'ra': float(star_ra[idx]), 'dec': float(star_dec[idx])
        })
    
    del px0, py0, in_sca0, candidates, x_sca_ref, y_sca_ref
    gc.collect()
    
    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (0.11**2) * exp_time

    print(f"🎬 Generating {len(times)} full-SCA epochs with {len(events)} events...")
    for i, t in enumerate(tqdm(times)):
        dt = t - times[0]
        curr_drift = drift_vec_px * dt
        
        epoch_ra, epoch_dec = original_wcs.xyToradec(
            sca_center_x + curr_drift[0], 
            sca_center_y + curr_drift[1], 
            units=galsim.degrees
        )
        
        model = template.copy()
        ris_wcs.fill_in_parameters(
            model.meta, 
            SkyCoord(epoch_ra, epoch_dec, unit='deg', frame='icrs'), 
            boresight=False, 
            pa_aper=pa_aper
        )
        curr_wcs = ris_wcs.get_wcs(model.meta)
        
        current_fluxes = fluxes_base.copy()
        current_event_data = []
        for ev in events:
            A = paczynski_magnification(t, ev['t0'], ev['tE'], ev['u0'])
            current_fluxes[ev['idx']] *= A
            
            ex, ey = curr_wcs.radecToxy(ev['ra'], ev['dec'], units=galsim.degrees)
            current_event_data.append({
                'target_x_sca': float(ex), 'target_y_sca': float(ey),
                'magnification': float(A), 'true_mag': float(ev['mag_base']),
                't0': float(ev['t0'] - times[0]), 'tE': float(ev['tE']), 'u0': float(ev['u0']),
                'ra': float(ev['ra']), 'dec': float(ev['dec'])
            })
            
        epoch_img = render_tiled(curr_wcs, star_ra, star_dec, current_fluxes, mosaic_size, psf_1x, tile_size=1024)
        noisy = np.random.poisson(np.maximum(0, epoch_img + sky_level)).astype(np.float32)
        noisy += np.random.normal(0, read_noise, noisy.shape)
        
        model.data = noisy.astype(np.float32)
        model.dq = np.zeros_like(noisy, dtype=np.uint32)
        model.err = np.full_like(noisy, read_noise, dtype=np.float16)
        model.meta.exposure.start_time = obs_time_base + (t * u.day)
        model.meta.exposure.exposure_time = exp_time

        # Add mission-fidelity photometry metadata
        # pixel_area in steradians
        pixel_area_arcsec2 = 0.11**2
        pixel_area_sr = pixel_area_arcsec2 / 4.254517e10
        model.meta.photometry.pixel_area = float(pixel_area_sr)
        
        # conversion_microjanskys: uJy per (e-/s)
        # mag = -2.5*log10(flux_uJy) + 23.9
        # mag = -2.5*log10(counts * conv_uJy) + 23.9
        # mag = -2.5*log10(counts) - 2.5*log10(conv_uJy) + 23.9
        # ZP = 23.9 - 2.5*log10(conv_uJy)
        # conv_uJy = 10**((23.9 - ZP) / 2.5)
        conv_uJy = float(10**((23.9 - zp) / 2.5))
        model.meta.photometry.conversion_microjanskys = conv_uJy
        model.meta.photometry.conversion_microjanskys_uncertainty = conv_uJy * 0.01
        
        # conversion_megajanskys: MJy/sr per (e-/s)
        # 1 uJy = 1e-12 MJy
        conv_MJy_sr = (conv_uJy * 1e-12) / pixel_area_sr
        model.meta.photometry.conversion_megajanskys = float(conv_MJy_sr)
        model.meta.photometry.conversion_megajanskys_uncertainty = float(conv_MJy_sr * 0.01)

        # Legacy/General factor
        model.meta.photometry.conversion_factor = float(conv_MJy_sr)
        model.meta.photometry.conversion_factor_err = float(conv_MJy_sr * 0.01)

        # Physics params
        try:
            model.meta.exposure.gain = 1.0 # counts per electron
        except:
            pass

        output_path = os.path.join(args.outdir, f"epoch_{i:04d}.asdf")
        model.save(output_path)

        # Save Multi-Event Ground Truth Sidecar
        gt = {
            'events': current_event_data, 
            'target_ra': float(epoch_ra), 
            'target_dec': float(epoch_dec),
            'sky_level': float(sky_level),
            'read_noise': float(read_noise),
            'mag_zp': float(zp)
        }
        with open(output_path.replace(".asdf", "_gt.json"), 'w') as f: json.dump(gt, f)

        del model, noisy, epoch_img, current_fluxes, current_event_data, gt, curr_wcs
        gc.collect()
    
    print(f"✨ Success! Rigorous forward-modeled SCA series ready in: {args.outdir}")

if __name__ == "__main__": main()
