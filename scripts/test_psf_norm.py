
import numpy as np
from castor.data.stage0_gaussian import get_gaussian_psf, get_oversampled_gaussian_psf

def test_psf():
    sigma = 0.405
    kernel_size = 25
    
    print("--- Testing get_gaussian_psf (Deprecated) ---")
    psf = get_gaussian_psf(sigma=sigma, kernel_size=kernel_size)
    print(f"PSF Sum: {psf.sum()}")
    print(f"PSF Max: {psf.max()}")
    print(f"PSF Shape: {psf.shape}")
    assert np.allclose(psf.sum(), 1.0), f"PSF sum is {psf.sum()}, expected 1.0"

    print("\n--- Testing get_oversampled_gaussian_psf ---")
    psf_os = get_oversampled_gaussian_psf(sigma_detector=sigma, grid_size=kernel_size, oversample=4)
    print(f"PSF OS Sum: {psf_os.sum()}")
    print(f"PSF OS Max: {psf_os.max()}")
    print(f"PSF OS Shape: {psf_os.shape}")
    assert np.allclose(psf_os.sum(), 1.0), f"PSF OS sum is {psf_os.sum()}, expected 1.0"
    
    # Comparison
    print(f"\nMax Value Comparison:")
    print(f"1x Central Sampling Max: {psf.max():.6f}")
    print(f"4x Oversampled Max:     {psf_os.max():.6f}")
    print(f"Difference:              {psf_os.max() - psf.max():.6f}")

if __name__ == "__main__":
    test_psf()
