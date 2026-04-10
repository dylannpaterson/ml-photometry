#!/usr/bin/env python3
import numpy as np
import torch
import os
from scipy.ndimage import center_of_mass
import castor.data.stage0_gaussian as s0
from castor.data.stage0_gaussian import generate_mosaic_data, generate_field_realistic_psf_library
from castor.constants import SHAPE_SIZE

# --- Force Stars ---
def fake_sample_magnitudes(n_total, *args, **kwargs):
    """ Returns mag 20.0 for everything. """
    return np.array([20.0] * n_total)

# --- Force Symmetric Gaussians ---
def fake_psf_library(num_psfs=100, grid_size=127, oversample=4):
    """ Generates perfectly symmetric Gaussians centered at (S-1)/2.0. """
    S = SHAPE_SIZE * oversample
    library = np.zeros((num_psfs, S, S), dtype=np.float32)
    # Center at (S-1)/2
    center = (S - 1) / 2.0
    y, x = np.meshgrid(np.arange(S) - center, np.arange(S) - center, indexing='ij')
    psf = np.exp(-(x**2 + y**2) / (2 * (1.5 * oversample)**2))
    psf /= psf.sum()
    for i in range(num_psfs):
        library[i] = psf
    return library

# Monkeypatch the module
s0.sample_bulge_magnitudes = fake_sample_magnitudes
s0.generate_field_realistic_psf_library = fake_psf_library

def verify_cpu_absolute_official():
    print("🧪 Starting Local CPU Renderer Absolute Accuracy Test (OFFICIAL Engine)...")
    print(f"💡 FORCING: Stars @ Mag 20.0 | Size ({SHAPE_SIZE}) | Symmetric Gaussian PSF")
    
    mosaic_size = 512 # Larger mosaic to avoid edge effects
    O = 4
    num_psfs = 10
    
    # 1. Setup params
    params = {
        'image_size': 512,
        'min_stars': 1,
        'max_stars': 1
    }
    
    # 2. Generate master PSF library
    kb_array = fake_psf_library(num_psfs=num_psfs, grid_size=SHAPE_SIZE, oversample=O)
    
    # 3. Use the OFFICIAL engine
    print("🎨 Rendering mosaic via OFFICIAL engine...")
    img, cat, meta, _ = generate_mosaic_data(mosaic_size, params, kb_array)
    
    if len(cat) == 0:
        print("❌ Error: No stars rendered.")
        return

    # 4. Measure Centroid of the brightest star
    cat = np.sort(cat, order='flux')[::-1]
    star = cat[0]
    target_x, target_y = star['x'], star['y']
    
    # Calculate crop around target - use larger patch since PSF is 129x129
    patch = 64 
    ix, iy = int(target_x), int(target_y)
    
    # Ensure patch is within bounds
    y1, y2 = max(0, iy-patch), min(mosaic_size, iy+patch)
    x1, x2 = max(0, ix-patch), min(mosaic_size, ix+patch)
    
    crop = img[y1:y2, x1:x2]
    local_centroid = center_of_mass(crop)
    
    # Translate to global
    actual_x = x1 + local_centroid[1]
    actual_y = y1 + local_centroid[0]
    
    err_x = actual_x - target_x
    err_y = actual_y - target_y
    
    print("\n" + "="*40)
    print(f"🎯 ABSOLUTE ACCURACY RESULTS (OFFICIAL Engine)")
    print(f"  Target X:  {target_x:.3f}")
    print(f"  Actual X:  {actual_x:.3f} -> Error: {err_x:+.6f}")
    print(f"\n  Target Y:  {target_y:.3f}")
    print(f"  Actual Y:  {actual_y:.3f} -> Error: {err_y:+.6f}")
    
    # Error should be very small (< 0.01)
    if abs(err_x) < 0.01 and abs(err_y) < 0.01:
        print("\n✅ SUCCESS: Official CPU Renderer is tracking absolute coordinates accurately.")
    else:
        print("\n❌ FAILURE: Official CPU Renderer still has absolute drift.")
            
    print("="*40)

if __name__ == "__main__":
    verify_cpu_absolute_official()
