import numpy as np
from castor.data.stage0_gaussian import render_gaussian_stars, get_oversampled_gaussian_psf
from castor.constants import SHAPE_SIZE

def verify_flux():
    img_size = 256
    # Single star with flux 1000.0 at a subpixel position
    px = np.array([128.3], dtype=np.float32)
    py = np.array([128.7], dtype=np.float32)
    fluxes = np.array([1000.0], dtype=np.float32)
    
    sigma = 0.405
    psf_kernel = get_oversampled_gaussian_psf(sigma_detector=sigma, grid_size=SHAPE_SIZE, oversample=4)
    
    # Render
    img = render_gaussian_stars(img_size, img_size, px, py, fluxes, psf_kernel=psf_kernel)
    
    total_rendered_flux = np.sum(img)
    input_flux = np.sum(fluxes)
    
    diff = total_rendered_flux - input_flux
    rel_diff = abs(diff) / input_flux
    
    print(f"--- Flux Preservation Test ---")
    print(f"Input Flux:          {input_flux:.6f}")
    print(f"Rendered Flux (Sum): {total_rendered_flux:.6f}")
    print(f"Absolute Difference: {diff:.6e}")
    print(f"Relative Difference: {rel_diff:.6e}")
    
    # Test multiple stars
    n_stars = 100
    px_m = np.random.uniform(50, 200, n_stars).astype(np.float32)
    py_m = np.random.uniform(50, 200, n_stars).astype(np.float32)
    fluxes_m = np.random.uniform(10, 1000, n_stars).astype(np.float32)
    
    img_m = render_gaussian_stars(img_size, img_size, px_m, py_m, fluxes_m, psf_kernel=psf_kernel)
    total_rendered_flux_m = np.sum(img_m)
    input_flux_m = np.sum(fluxes_m)
    
    diff_m = total_rendered_flux_m - input_flux_m
    rel_diff_m = abs(diff_m) / input_flux_m
    
    print(f"\n--- Multi-star Flux Preservation Test ({n_stars} stars) ---")
    print(f"Input Flux:          {input_flux_m:.6f}")
    print(f"Rendered Flux (Sum): {total_rendered_flux_m:.6f}")
    print(f"Absolute Difference: {diff_m:.6e}")
    print(f"Relative Difference: {rel_diff_m:.6e}")

    if rel_diff < 1e-5 and rel_diff_m < 1e-5:
        print("\n✅ Flux is preserved to high precision.")
    else:
        print("\n❌ Flux preservation failed.")

if __name__ == "__main__":
    verify_flux()
