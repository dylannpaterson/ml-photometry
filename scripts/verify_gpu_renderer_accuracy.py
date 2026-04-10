#!/usr/bin/env python3
import numpy as np
import torch
import os
from scipy.ndimage import center_of_mass
import castor.data.stage0_gaussian as s0
from castor.data.stage0_gaussian import generate_mosaic_data, generate_field_realistic_psf_library
from castor.data.gpu_renderer import render_generate_and_filter_gpu, is_gpu_available
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

def verify_gpu_absolute_official():
    print(f"🧪 Starting GPU Renderer Absolute Accuracy Test (GPU Available: {is_gpu_available()})...")
    print(f"💡 FORCING: Stars @ Mag 20.0 | Size ({SHAPE_SIZE}) | Symmetric Gaussian PSF")
    
    mosaic_size = 512 
    O = 4
    num_psfs = 1
    
    # 1. Setup params
    # We need to generate star params first to pass to GPU renderer
    exp_time, zp = 50.0, 26.5
    flux_20 = exp_time * (10 ** (-0.4 * (20.0 - zp)))
    
    # Forced position
    target_x, target_y = mosaic_size / 2.0 + 0.65, mosaic_size / 2.0 + 0.85
    fluxes = np.array([flux_20], dtype=np.float32)
    mags = np.array([20.0], dtype=np.float32)
    
    # 2. Generate master PSF library (4x)
    kb_array = fake_psf_library(num_psfs=num_psfs, grid_size=SHAPE_SIZE, oversample=O)
    single_psf_4x = kb_array[0]
    
    # 3. Use the GPU engine
    print("🎨 Rendering mosaic via GPU engine...")
    # render_generate_and_filter_gpu randomizes positions, but we want to test a SPECIFIC position.
    # So we'll call the core renderer if we can, or just accept the random one and check its error.
    # Actually, the GPU renderer is designed to be a generator too. 
    # Let's use the random one but check the error of WHATEVER it produces.
    
    # We'll generate 1 star
    img, x_gpu, y_gpu, f_gpu, m_gpu = render_generate_and_filter_gpu(fluxes, mags, single_psf_4x, mosaic_size)
    
    if len(x_gpu) == 0:
        print("❌ Error: No stars rendered.")
        return

    # 4. Measure Centroid
    # We test the first star produced
    tx, ty = x_gpu[0], y_gpu[0]
    
    # Calculate crop around target
    patch = 64 
    ix, iy = int(tx), int(ty)
    
    # Ensure patch is within bounds
    y1, y2 = max(0, iy-patch), min(mosaic_size, iy+patch)
    x1, x2 = max(0, ix-patch), min(mosaic_size, ix+patch)
    
    crop = img[y1:y2, x1:x2]
    local_centroid = center_of_mass(crop)
    
    # Translate to global
    actual_x = x1 + local_centroid[1]
    actual_y = y1 + local_centroid[0]
    
    err_x = actual_x - tx
    err_y = actual_y - ty
    
    print("\n" + "="*40)
    print(f"🎯 ABSOLUTE ACCURACY RESULTS (GPU Engine)")
    print(f"  Target X:  {tx:.3f}")
    print(f"  Actual X:  {actual_x:.3f} -> Error: {err_x:+.6f}")
    print(f"\n  Target Y:  {ty:.3f}")
    print(f"  Actual Y:  {actual_y:.3f} -> Error: {err_y:+.6f}")
    
    # Error should be very small (< 0.01)
    if abs(err_x) < 0.01 and abs(err_y) < 0.01:
        print("\n✅ SUCCESS: GPU Renderer is tracking absolute coordinates accurately.")
    else:
        print("\n❌ FAILURE: GPU Renderer still has absolute drift.")
            
    print("="*40)

if __name__ == "__main__":
    verify_gpu_absolute_official()
