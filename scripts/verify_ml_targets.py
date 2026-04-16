import os
import sqlite3
import json
import numpy as np
from roman_datamodels import datamodels
from astropy.coordinates import SkyCoord
import astropy.units as u

def verify_ml_targets(series_dir="data/microlensing_full_series", db_path="../pollux/master_reference.db", stack_path="data/master_stack_full.asdf"):
    # 1. Load ML targets from first sidecar
    gt_path = os.path.join(series_dir, "epoch_0000_gt.json")
    with open(gt_path, 'r') as f:
        gt = json.load(f)
    
    events = gt['events']
    n_events = len(events)
    target_ra = np.array([e['ra'] for e in events])
    target_dec = np.array([e['dec'] for e in events])
    target_mag = np.array([e['true_mag'] for e in events])
    
    print(f"🎯 Tracking {n_events} microlensing targets...")

    # 2. Connect to the Pollux Database
    if not os.path.exists(db_path):
        print(f"❌ Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute('''
        SELECT t.ra_weighted, t.dec_weighted, p.flux_weighted
        FROM targets t
        LEFT JOIN target_photometry p ON t.uuid = p.target_uuid
    ''')
    rows = c.fetchall()
    conn.close()
    
    found_ra = np.array([r[0] for r in rows])
    found_dec = np.array([r[1] for r in rows])
    found_flux = np.array([r[2] for r in rows])
    
    print(f"📸 Pollux Database contains {len(found_ra)} sources.")

    # 3. Load Stack WCS
    with datamodels.open(stack_path) as model:
        wcs = model.meta.wcs

    # 4. Cross-match targets using RA/Dec
    match_radius = 0.5 / 3600.0 # 0.5 arcsec in degrees
    
    matches = []
    offsets_px = []
    
    for i in range(n_events):
        dist = np.sqrt((found_ra - target_ra[i])**2 + (found_dec - target_dec[i])**2)
        idx = np.argmin(dist)
        
        if dist[idx] < match_radius:
            # Projected target pixels in master stack
            target_coord = SkyCoord(ra=target_ra[i]*u.deg, dec=target_dec[i]*u.deg, frame='icrs')
            found_coord = SkyCoord(ra=found_ra[idx]*u.deg, dec=found_dec[idx]*u.deg, frame='icrs')
            
            tx, ty = wcs.world_to_pixel(target_coord)
            fx, fy = wcs.world_to_pixel(found_coord)
            
            dx = fx - tx
            dy = fy - ty
            
            matches.append({
                'id': i,
                'dist_asec': dist[idx] * 3600,
                'dx': dx,
                'dy': dy,
                'found_flux': found_flux[idx],
                'target_mag': target_mag[i]
            })
            offsets_px.append([dx, dy])

    offsets_px = np.array(offsets_px)
    
    print(f"\n📊 MATCHING RESULTS:")
    print(f"-------------------")
    print(f"Targets Found: {len(matches)} / {n_events}")
    
    if matches:
        mean_dx = np.mean(offsets_px[:, 0])
        mean_dy = np.mean(offsets_px[:, 1])
        std_dx = np.std(offsets_px[:, 0])
        std_dy = np.std(offsets_px[:, 1])
        
        print(f"Mean Pixel Offset (Pollux vs Truth): DX={mean_dx:.4f}, DY={mean_dy:.4f}")
        print(f"Offset Jitter (σ):                   DX={std_dx:.4f}, DY={std_dy:.4f}")
        
        if np.abs(mean_dx) < 0.1 and np.abs(mean_dy) < 0.1:
            print("\n✅ VERDICT: ML target stars are EXACTLY where they should be in the Pollux database.")
        else:
            print("\n❌ VERDICT: Systematic offset detected between simulation and recovery.")
    else:
        print("\n❌ VERDICT: No targets matched. Check WCS/Coordinates.")

if __name__ == "__main__":
    verify_ml_targets()
