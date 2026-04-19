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
    generate_stpsf_roman_psf,
    calculate_safe_magnitude_cutoff
)
from castor.constants import GLOBAL_STRETCH_SCALE, SHAPE_SIZE
from scipy.signal import fftconvolve
from scipy.ndimage import map_coordinates
from roman_datamodels import datamodels
from romanisim import ris_make_utils as ris
from romanisim import wcs as ris_wcs
from romanisim import image as sim_image
from romanisim import persistence as sim_persistence
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
import asdf
import copy
import gc
from romancal.assign_wcs import AssignWcsStep

def paczynski_magnification(t, t0, tE, u0):
    """ Standard Paczynski microlensing magnification formula. """
    u = np.sqrt(u0**2 + ((t - t0) / tE)**2)
    A = (u**2 + 2) / (u * np.sqrt(u**2 + 4) + 1e-9)
    return A

def create_mission_template(ra=266.4167, dec=-29.0078, band='F146', sca=1):
    """ Generates a mission-fidelity empty ImageModel for metadata bootstrapping. """
    print(f"🛰️  Creating mission-fidelity metadata template for SCA {sca} ({band})...")
    obs_time = '2026-10-31T00:00:00'
    
    # 1. Setup metadata and WCS
    metadata = ris.set_metadata(date=obs_time, bandpass=band, sca=sca, ma_table_number=1002, usecrds=True)
    ris_wcs.fill_in_parameters(metadata, SkyCoord(ra, dec, unit='deg', frame='icrs'), boresight=False, pa_aper=0.0)

    # 2. Minimal empty catalog
    minimal_catalog = Table()
    minimal_catalog['ra'], minimal_catalog['dec'], minimal_catalog[band] = [ra], [dec], [1e-9]

    # 3. Simulate minimal frame to populate DataModel tree correctly
    res, _ = sim_image.simulate(
        metadata, minimal_catalog, 
        rng=galsim.UniformDeviate(42), 
        persistence=sim_persistence.Persistence(), 
        usecrds=True, psftype='webbpsf', level=2
    )
    return datamodels.ImageModel(res)

def render_tiled(px, py, fluxes, mosaic_size, psf_1x, tile_size=1024):
    """ Memory-efficient rendering using tiling. Coordinates must be 0-indexed (NumPy). """
    full_image = np.zeros((mosaic_size, mosaic_size), dtype=np.float32)
    psf_half = psf_1x.shape[0] // 2
    
    # Process in tiles
    for y0 in range(0, mosaic_size, tile_size):
        for x0 in range(0, mosaic_size, tile_size):
            y1 = min(y0 + tile_size, mosaic_size)
            x1 = min(x0 + tile_size, mosaic_size)
            
            # Tile bounds with PSF buffer
            tx0, tx1 = x0 - psf_half - 2, x1 + psf_half + 2
            ty0, ty1 = y0 - psf_half - 2, y1 + psf_half + 2
            
            # Filter stars for this tile (px/py are 0-indexed)
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
    parser.add_argument("--psf_library", default="master_psf_library.pt")
    parser.add_argument("--dust", action="store_true", help="Enable consistent interstellar cirrus (dust) extinction")
    
    args = parser.parse_args()
    config = load_config(args.config); data_cfg = config["data_params"]; os.makedirs(args.outdir, exist_ok=True)
    mosaic_size = args.mosaic_size; O = 4; zp, sky_mag = 26.5, 22.0
    read_noise = data_cfg["physics_params"].get("read_noise", 5.0)
    exp_time = args.exp_time
    
    # --- 🏗️ TEMPLATE BOOTSTRAPPING ---
    template = create_mission_template()
    pa_aper, obs_time_base = template.meta.wcsinfo.roll_ref, Time(template.meta.exposure.start_time)
    
    # Internal GalSim WCS for boresight calculation
    original_wcs_gs = ris_wcs.get_wcs(template.meta)
    sca_center_x, sca_center_y = (mosaic_size - 1) / 2.0, (mosaic_size - 1) / 2.0

    print(f"🛰️  Generating realistic Roman PSF on-the-fly...")
    repr_psf_4x = generate_stpsf_roman_psf(grid_size=SHAPE_SIZE, oversample=O)
    psf_1x = repr_psf_4x.reshape(SHAPE_SIZE, O, SHAPE_SIZE, O).mean(axis=(1, 3))
    psf_1x /= (np.sum(psf_1x) + 1e-9)

    cadence_days = args.cadence / 24.0
    times = (args.t0 - (args.num_epochs // 2) * cadence_days) + np.arange(args.num_epochs) * cadence_days
    drift_vec_px = np.array([np.cos(np.radians(args.drift_angle)), np.sin(np.radians(args.drift_angle))]) * args.drift_rate
    
    max_drift_px = args.drift_rate * (times.max() - times.min())
    padding_px = int(max_drift_px) + 100
    
    # --- 🌫️ MASTER DUST MAP GENERATION ---
    if args.dust:
        from castor.data.stage0_gaussian import generate_dust_cirrus
        print("🌫️ Generating Master Interstellar Cirrus (Dust) Map...")
        dust_map_size = mosaic_size + padding_px * 2
        raw_dust_map = generate_dust_cirrus(dust_map_size, 1.0)
        max_extinction = np.random.uniform(1.0, 4.0)
        master_transmission = 10 ** (-0.4 * raw_dust_map * max_extinction)
        master_scattering = raw_dust_map * np.random.uniform(10, 50)
    else:
        master_transmission = None
        master_scattering = None

    super_size = mosaic_size + padding_px * 2
    n_stars_approx = int(np.random.uniform(data_cfg['min_stars'], data_cfg['max_stars']) * (super_size / 256)**2)
    mags = sample_bulge_magnitudes(n_stars_approx, 15.5, 0.35, 10.0, m_min=12.0, m_max=32.0, gamma=0.3)
    fluxes_base = exp_time * (10 ** (-0.4 * (mags - zp)))
    
    n_total_stars = len(mags)
    x_sca_ref = np.random.uniform(-padding_px, mosaic_size + padding_px, n_total_stars)
    y_sca_ref = np.random.uniform(-padding_px, mosaic_size + padding_px, n_total_stars)
    
    # NOTE: fluxes_base here is the ABSOLUTE base flux (no dust yet)
    
    # Convert initial reference pixels to RA/Dec using GalSim (1-indexed for simulation)
    star_ra, star_dec = original_wcs_gs.xyToradec(x_sca_ref + 1.0, y_sca_ref + 1.0, units=galsim.degrees)
    
    # --- 🛡️ ROBUST CANDIDATE SELECTION ---
    # We must ensure the selected stars are on the detector according to the TEMPLATE WCS
    # This prevents NaN issues where AssignWcsStep might return null for stars slightly out of bounds.
    from romancal.assign_wcs import AssignWcsStep
    template_model = template.copy()
    template_model = AssignWcsStep.call(template_model)
    template_gwcs = template_model.meta.wcs
    
    px_init, py_init = template_gwcs.world_to_pixel_values(star_ra, star_dec)
    
    # Select events with sufficient baseline
    # We select stars well within the detector (500px buffer) to survive drift
    candidates = np.where(
        (mags <= 25.0) & 
        (~np.isnan(px_init)) & (~np.isnan(py_init)) &
        (px_init > 500) & (px_init < mosaic_size - 500) &
        (py_init > 500) & (py_init < mosaic_size - 500)
    )[0]
    event_indices = np.random.choice(candidates, min(len(candidates), args.num_events), replace=False)
    
    events = []
    times_range = times.max() - times.min()
    for idx in event_indices:
        tE = float(np.random.uniform(args.tE_min, min(args.tE_max, times_range / 4.0)))
        # Ensure t0 is far enough from edges for baseline
        t0_min = times.min() + 1.5 * tE
        t0_max = times.max() - 1.5 * tE
        if t0_max <= t0_min:
            t0 = float(np.random.uniform(times.min(), times.max()))
        else:
            t0 = float(np.random.uniform(t0_min, t0_max))
            
        events.append({
            'idx': int(idx), 't0': t0, 'tE': tE, 'u0': float(np.random.uniform(0.01, 0.5)),
            'mag_base': float(mags[idx]), 'flux_base': float(fluxes_base[idx]),
            'ra': float(star_ra[idx]), 'dec': float(star_dec[idx])
        })
    
    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (0.11**2) * exp_time

    print(f"🎬 Generating {len(times)} full-SCA epochs with {len(events)} events...")
    for i, t in enumerate(tqdm(times)):
        dt = t - times[0]
        curr_drift = drift_vec_px * dt
        
        # Calculate epoch boresight using GalSim (1-indexed)
        epoch_ra, epoch_dec = original_wcs_gs.xyToradec(
            sca_center_x + curr_drift[0] + 1.0, 
            sca_center_y + curr_drift[1] + 1.0, 
            units=galsim.degrees
        )
        
        model = template.copy()
        ris_wcs.fill_in_parameters(
            model.meta, 
            SkyCoord(epoch_ra, epoch_dec, unit='deg', frame='icrs'), 
            boresight=False, 
            pa_aper=pa_aper
        )
        
        # 2. Rebuild the GWCS object tree to reflect the new wcsinfo
        # This guarantees 100% Roman Data Model compliance for the output .asdf
        model = AssignWcsStep.call(model)
        
        # 3. Extract the fresh WCS object and calculate new pixel positions
        gwcs_obj = model.meta.wcs 
        px_epoch, py_epoch = gwcs_obj.world_to_pixel_values(star_ra, star_dec)

        # Start from base fluxes every epoch
        current_fluxes = fluxes_base.copy()
        
        # --- 🌫️ EPOCH-SPECIFIC DUST SAMPLING ---
        if master_transmission is not None:
            # 1. Generate full-resolution transmission map for the image
            y_indices, x_indices = np.meshgrid(np.arange(mosaic_size), np.arange(mosaic_size), indexing='ij')
            ref_x = x_indices + (padding_px + curr_drift[0])
            ref_y = y_indices + (padding_px + curr_drift[1])
            
            epoch_transmission = map_coordinates(master_transmission, [ref_y, ref_x], order=1, mode='nearest')
            epoch_scattering = map_coordinates(master_scattering, [ref_y, ref_x], order=1, mode='nearest')
            
            # 2. Sample transmission at ALL star positions for this epoch
            # Stars drift across the dust map
            star_transmissions = map_coordinates(master_transmission, [y_sca_ref + padding_px + curr_drift[1], x_sca_ref + padding_px + curr_drift[0]], order=1, mode='nearest')
            current_fluxes *= star_transmissions
            
            frac_bg = 0.60
            sky_foreground = sky_level * (1.0 - frac_bg)
            sky_background_attenuated = (sky_level * frac_bg) * epoch_transmission
            epoch_sky_map = sky_foreground + sky_background_attenuated + epoch_scattering
        else:
            epoch_sky_map = sky_level
            star_transmissions = np.ones_like(px_epoch)

        current_event_data = []
        for ev in events:
            A = paczynski_magnification(t, ev['t0'], ev['tE'], ev['u0'])
            
            # Attenuated flux BEFORE microlensing
            ev_attenuation = star_transmissions[ev['idx']]
            attenuated_base_flux = ev['flux_base'] * ev_attenuation
            
            # Apply microlensing to the already attenuated star
            current_fluxes[ev['idx']] *= A 
            
            current_event_data.append({
                'target_x_sca': float(px_epoch[ev['idx']]), 
                'target_y_sca': float(py_epoch[ev['idx']]),
                'magnification': float(A), 
                'dust_attenuation': float(ev_attenuation),
                'apparent_flux': float(attenuated_base_flux * A),
                'true_mag': float(ev['mag_base']),
                't0': float(ev['t0'] - times[0]), 'tE': float(ev['tE']), 'u0': float(ev['u0']),
                'ra': float(ev['ra']), 'dec': float(ev['dec'])
            })
            
        epoch_img = render_tiled(px_epoch, py_epoch, current_fluxes, mosaic_size, psf_1x, tile_size=1024)
        
        noisy = np.random.poisson(np.maximum(0, epoch_img + epoch_sky_map)).astype(np.float32)
        noisy += np.random.normal(0, read_noise, noisy.shape)
        
        model.data = noisy.astype(np.float32)
        model.dq = np.zeros_like(noisy, dtype=np.uint32)
        model.err = np.full_like(noisy, read_noise, dtype=np.float16)
        model.meta.exposure.start_time = obs_time_base + (t * u.day)
        model.meta.exposure.exposure_time = exp_time

        # Mission-fidelity photometry metadata
        pixel_area_arcsec2 = 0.11**2
        pixel_area_sr = pixel_area_arcsec2 / 4.254517e10
        model.meta.photometry.pixel_area = float(pixel_area_sr)
        conv_uJy = float(10**((23.9 - zp) / 2.5))
        model.meta.photometry.conversion_microjanskys = conv_uJy
        conv_MJy_sr = (conv_uJy * 1e-12) / pixel_area_sr
        model.meta.photometry.conversion_megajanskys = float(conv_MJy_sr)

        output_path = os.path.join(args.outdir, f"epoch_{i:04d}.asdf")
        model.save(output_path)

        # Identify top 1000 brightest stars currently on the detector for verification
        bright_stars = []
        b_idx = np.argsort(current_fluxes)[-1000:][::-1]
        for idx in b_idx:
            x, y = px_epoch[idx], py_epoch[idx]
            # Only record if it's on the detector
            if 0 <= x < mosaic_size and 0 <= y < mosaic_size:
                bright_stars.append({
                    'ra': float(star_ra[idx]),
                    'dec': float(star_dec[idx]),
                    'x': float(x),
                    'y': float(y),
                    'flux': float(current_fluxes[idx])
                })
            if len(bright_stars) >= 100: # 100 is plenty for verification
                break


        gt = {
            'events': current_event_data, 
            'bright_stars': bright_stars,
            'target_ra': float(epoch_ra), 
            'target_dec': float(epoch_dec),
            'sky_level': float(sky_level),
            'read_noise': float(read_noise),
            'mag_zp': float(zp)
        }
        with open(output_path.replace(".asdf", "_gt.json"), 'w') as f: json.dump(gt, f)

        del model, noisy, epoch_img, current_fluxes, current_event_data, gt, gwcs_obj
        gc.collect()
    
    print(f"✨ Success! Timeseries generated in: {args.outdir}")

if __name__ == "__main__": main()
