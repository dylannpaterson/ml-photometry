import numpy as np
from castor.data.stage0_gaussian import sample_bulge_magnitudes

def calculate_detectable_stars(n_chunk_total, exp_time=45.0, zp=26.5, sky_mag=22.0):
    """
    Simulates a 256x256 chunk and counts stars with SNR >= 5.
    """
    # 1. Physics Constants
    pixel_scale = 0.11
    sigma = 1.5
    read_noise = 5.0
    n_pix = 4 * np.pi * (sigma**2)
    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (pixel_scale**2) * exp_time
    psf_peak = 1.0 / (2 * np.pi * sigma**2)
    
    # 2. Sample Magnitudes
    rc_loc = 15.5
    rc_scale = 0.35
    rc_enhancement = 10.0
    mags = sample_bulge_magnitudes(n_chunk_total, rc_loc, rc_scale, rc_enhancement, m_min=12.0, m_max=32.0)
    fluxes = exp_time * (10 ** (-0.4 * (mags - zp)))
    
    # 3. SNR Calculation (Worst case: high background from neighbors)
    # We'll assume a moderate crowding factor for calibration.
    # In a very dense field, the 'effective' sky is higher.
    # Let's use the average flux density as a background proxy.
    avg_flux_per_pix = fluxes.sum() / (256*256)
    effective_sky = sky_level + avg_flux_per_pix
    
    noise_var = fluxes + n_pix * (effective_sky + read_noise**2)
    snrs = fluxes / np.sqrt(noise_var)
    
    return np.sum(snrs >= 5.0)

def calibrate():
    # Sweep total stars per 256x256 chunk
    test_counts = [500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000]
    
    print(f"{'Total Stars':>12} | {'Detectable (SNR>=5)':>20} | {'Ratio':>8}")
    print("-" * 45)
    
    for n in test_counts:
        n_det = calculate_detectable_stars(n)
        print(f"{n:12,} | {n_det:20,} | {n_det/n:8.4f}")

if __name__ == "__main__":
    calibrate()
