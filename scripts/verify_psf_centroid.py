import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import center_of_mass
import torch
import os
import sys

# Ensure project root is in path
sys.path.append(os.getcwd())

from castor.data.stage0_gaussian import get_gaussian_psf, get_oversampled_gaussian_psf
from castor.engine.inference import InferenceEngine
from castor.constants import SHAPE_SIZE

def test_psf_alignment(psf, name, target_x, target_y):
    img_size = 256
    flux = 1000.0
    
    # 1. Bilinear splat logic (pixel-center accurate)
    grid = np.zeros((img_size, img_size), dtype=np.float32)
    x0, y0 = int(np.floor(target_x)), int(np.floor(target_y))
    dx, dy = target_x - x0, target_y - y0
    w00, w10, w01, w11 = (1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy
    
    def paint(gr, x, y, w, f):
        if 0 <= x < img_size and 0 <= y < img_size:
            gr[y, x] += f * w

    paint(grid, x0, y0, w00, flux)
    paint(grid, x0+1, y0, w10, flux)
    paint(grid, x0, y0+1, w01, flux)
    paint(grid, x0+1, y0+1, w11, flux)
    
    # 2. Convolve
    # Note: center_of_mass returns (y, x). 
    # For a centered PSF, center_of_mass(psf) should be (S-1)/2, (S-1)/2
    img = fftconvolve(grid, psf, mode='same')
    
    # 3. Measure Centroid
    meas_y, meas_x = center_of_mass(img)
    
    offset_x = meas_x - target_x
    offset_y = meas_y - target_y
    
    print(f"\n--- Testing {name} at ({target_x}, {target_y}) ---")
    print(f"🎯 Target:   ({target_x:.6f}, {target_y:.6f})")
    print(f"🔭 Measured: ({meas_x:.6f}, {meas_y:.6f})")
    print(f"⚠️  Offset:   ({offset_x:.6e}, {offset_y:.6e})")
    
    return abs(offset_x) < 1e-4 and abs(offset_y) < 1e-4

if __name__ == "__main__":
    success = True
    
    # 1. Test Generation PSF
    gen_psf = get_gaussian_psf(kernel_size=25, sigma=0.405)
    success &= test_psf_alignment(gen_psf, "Generation PSF (1x)", 128.0, 128.0)
    success &= test_psf_alignment(gen_psf, "Generation PSF (1x) subpixel", 128.3, 128.7)
    
    # 2. Test Inference (Diagnostic) PSF
    # Create a dummy engine
    dummy_config = {"data_params": {"image_size": 256}}
    engine = InferenceEngine(None, torch.device('cpu'), dummy_config)
    inf_psf = engine._get_centered_psf()
    
    success &= test_psf_alignment(inf_psf, "Inference Diagnostic PSF", 128.0, 128.0)
    
    # 3. Test 4x Oversampled Block-Integrated Gaussian
    # This matches the new Stage 0 training logic
    try:
        psf_integrated = get_oversampled_gaussian_psf(sigma_detector=0.405, grid_size=SHAPE_SIZE, oversample=4)
        success &= test_psf_alignment(psf_integrated, "4x Oversampled Integrated PSF", 128.0, 128.0)
    except Exception as e:
        print(f"\n❌ Error during oversampled verification: {e}")
        success = False
    
    if success:
        print("\n✅ ALL TESTS PASSED: PSF Centering is correct across Generation and Verification.")
    else:
        print("\n❌ TESTS FAILED: Centroid shift detected.")
        sys.exit(1)
