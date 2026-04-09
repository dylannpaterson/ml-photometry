#!/usr/bin/env python3
import os
import numpy as np
import time
import argparse
from scipy.ndimage import center_of_mass
from scipy.signal import fftconvolve
from castor.data.gpu_renderer import _get_jax_renderer_core, is_gpu_available

# Hardcoded constants to avoid extra imports
SHAPE_SIZE = 32
N_PCA_COMPONENTS = 10

def render_cpu_reference(px, py, fluxes, mosaic_size, psf_4x, O=4):
    """Reference implementation using NumPy area-integrated binning."""
    full_image = np.zeros((mosaic_size, mosaic_size), dtype=np.float32)
    x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
    dx_idx = np.clip(np.floor((px - x0) * O).astype(int), 0, O-1)
    dy_idx = np.clip(np.floor((py - y0) * O).astype(int), 0, O-1)
    valid = (x0 >= 0) & (x0 < mosaic_size) & (y0 >= 0) & (y0 < mosaic_size)
    
    # Pre-bin the 16 phases correctly
    psf_library = np.zeros((O, O, SHAPE_SIZE, SHAPE_SIZE), dtype=np.float32)
    padded_psf = np.pad(psf_4x, ((0, O), (0, O)))
    for dyi in range(O):
        for dxi in range(O):
            window = padded_psf[dyi : dyi + SHAPE_SIZE*O, dxi : dxi + SHAPE_SIZE*O]
            psf_library[dyi, dxi] = window.reshape(SHAPE_SIZE, O, SHAPE_SIZE, O).mean(axis=(1, 3))

    for dyi in range(O):
        for dxi in range(O):
            mask = valid & (dx_idx == dxi) & (dy_idx == dyi)
            if not mask.any(): continue
            flat_indices = y0[mask] * mosaic_size + x0[mask]
            grid = np.bincount(flat_indices, weights=fluxes[mask], minlength=mosaic_size*mosaic_size).reshape(mosaic_size, mosaic_size)
            full_image += fftconvolve(grid, psf_library[dyi, dxi], mode='same')
    return full_image

def main():
    parser = argparse.ArgumentParser(description="Verify GPU Renderer Mosaic Alignment")
    parser.add_argument("--mosaic_size", type=int, default=256)
    args = parser.parse_args()

    print(f"🚀 Initializing Verification (GPU Available: {is_gpu_available()})")
    
    O = 4
    S = SHAPE_SIZE
    mosaic_size = args.mosaic_size
    
    # 1. Single Star at sub-pixel position
    # We pick a position that previously triggered the 1-pixel shift (e.g., fractional part > 0.5)
    target_x, target_y = mosaic_size / 2.0 + 0.65, mosaic_size / 2.0 + 0.85
    px, py = np.array([target_x]), np.array([target_y])
    fluxes = np.array([1000.0])
    mags = np.array([20.0])
    
    # 2. Mock 4x PSF and Jitter
    # A simple Gaussian 4x PSF
    kh = (S * O) // 2
    gy, gx = np.meshgrid(np.arange(S*O) - kh, np.arange(S*O) - kh, indexing='ij')
    psf_4x = np.exp(-(gx**2 + gy**2) / (2 * 1.5**2))
    psf_4x /= psf_4x.sum()
    
    # Delta-function jitter (no-op) for simplicity
    jitter_kernel_4x = np.zeros((S*O, S*O))
    jitter_kernel_4x[kh, kh] = 1.0

    # 3. Render CPU Reference
    print("💻 Rendering CPU Reference...")
    img_cpu = render_cpu_reference(px, py, fluxes, mosaic_size, psf_4x, O=O)

    # 4. Render GPU Engine (JAX)
    print("🔥 Rendering GPU Engine (JAX)...")
    import jax
    import jax.numpy as jnp
    
    render_jax = _get_jax_renderer_core()
    
    # Mock PCA inputs (unused for this base test)
    eigen_psfs_4x = jnp.zeros((N_PCA_COMPONENTS, S*O, S*O))
    weights_anchors = jnp.zeros((5, N_PCA_COMPONENTS))
    
    img_gpu_jax, _ = render_jax(
        jnp.array(px), jnp.array(py), jnp.array(fluxes), jnp.array(mags),
        weights_anchors, jnp.array(psf_4x), eigen_psfs_4x, 
        jnp.array(jitter_kernel_4x), mosaic_size, 27.0
    )
    img_gpu = np.array(img_gpu_jax)

    # 5. Analysis
    residual = img_cpu - img_gpu
    max_res = np.abs(residual).max()
    
    print("\n" + "="*40)
    print(f"📊 ACCURACY CHECK")
    print(f"  Max Residual (CPU vs GPU): {max_res:.2e}")
    
    # Absolute Centroid Check
    patch = 20
    ix, iy = int(target_x), int(target_y)
    
    # Center of mass relative to the patch
    c_cpu_patch = center_of_mass(img_cpu[iy-patch:iy+patch, ix-patch:ix+patch])
    c_gpu_patch = center_of_mass(img_gpu[iy-patch:iy+patch, ix-patch:ix+patch])
    
    # Translate patch coordinates back to global image coordinates
    actual_x_cpu = (ix - patch) + c_cpu_patch[1]
    actual_y_cpu = (iy - patch) + c_cpu_patch[0]
    
    actual_x_gpu = (ix - patch) + c_gpu_patch[1]
    actual_y_gpu = (iy - patch) + c_gpu_patch[0]
    
    # Calculate absolute error against ground truth
    err_x_cpu = actual_x_cpu - target_x
    err_y_cpu = actual_y_cpu - target_y
    
    err_x_gpu = actual_x_gpu - target_x
    err_y_gpu = actual_y_gpu - target_y
    
    print(f"\n🎯 ABSOLUTE CENTROID ALIGNMENT")
    print(f"  Target Sub-pixel: ({target_x:.3f}, {target_y:.3f})")
    print(f"  CPU Rendered at:  ({actual_x_cpu:.3f}, {actual_y_cpu:.3f}) -> Error: {err_x_cpu:+.3f}, {err_y_cpu:+.3f}")
    print(f"  GPU Rendered at:  ({actual_x_gpu:.3f}, {actual_y_gpu:.3f}) -> Error: {err_x_gpu:+.3f}, {err_y_gpu:+.3f}")
    
    if abs(err_x_gpu) < 0.05 and abs(err_y_gpu) < 0.05:
        print("\n✅ SUCCESS: GPU Mosaic is accurately tracking absolute coordinates.")
    else:
        print("\n❌ FAILURE: GPU Renderer has absolute sub-pixel drift!")
    print("="*40)

if __name__ == "__main__":
    main()
