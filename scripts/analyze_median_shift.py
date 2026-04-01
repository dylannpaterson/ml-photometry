import numpy as np
import matplotlib.pyplot as plt
from castor.data.stage0_gaussian import sample_bulge_magnitudes
from scipy.signal import fftconvolve
import os

def find_counts_for_target_medians():
    # 1. Setup Simulation Params
    img_size = 256
    sigma = 1.5
    exp_time = 60.0
    zp = 26.5
    sky_mag = 22.0
    pixel_scale = 0.11
    
    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (pixel_scale**2) * exp_time
    print(f"Instrumental Sky Level: {sky_level:.2f}")

    # 2. Iterative Search
    target_medians = [1000, 5000]
    found_counts = {}
    
    rc_mag, rc_sigma, rc_enhancement = 15.5, 0.4, 3.0
    gamma = 0.3
    
    star_grid = np.zeros((img_size, img_size), dtype=np.float64)
    total_stars = 0
    step_size = 100000 
    
    # Render PSF once
    k_size = 31
    half = k_size // 2
    gy, gx = np.meshgrid(np.arange(k_size), np.arange(k_size))
    kernel = np.exp(-((gx - half)**2 + (gy - half)**2) / (2 * sigma**2))
    kernel /= kernel.sum()

    print(f"Searching for star counts to hit medians: {target_medians}")
    
    # We'll run for up to 10M stars
    for i in range(100): 
        mags = sample_bulge_magnitudes(step_size, rc_mag, rc_sigma, rc_enhancement, 
                                       m_min=12.0, m_max=32.0, gamma=gamma)
        fluxes = exp_time * (10 ** (-0.4 * (mags - zp)))
        x = np.random.uniform(0, img_size, len(mags))
        y = np.random.uniform(0, img_size, len(mags))
        
        counts, _, _ = np.histogram2d(y, x, bins=img_size, range=[[0, img_size], [0, img_size]], weights=fluxes)
        star_grid += counts
        total_stars += step_size
        
        # Check current median
        star_signal = fftconvolve(star_grid, kernel, mode='same')
        current_median = np.median(star_signal + sky_level)
        
        for target in target_medians:
            if target not in found_counts and current_median >= target:
                found_counts[target] = total_stars
                print(f"🎯 Target {target} hit at {total_stars:,} stars (Median: {current_median:.2f})")
        
        if len(found_counts) == len(target_medians):
            break
            
    print("\n--- Summary ---")
    for target, count in found_counts.items():
        print(f"Median {target:4d}: ~{count:,} stars")

if __name__ == "__main__":
    find_counts_for_target_medians()
