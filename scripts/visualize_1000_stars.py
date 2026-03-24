import numpy as np
import matplotlib.pyplot as plt
from castor.data.stage0_gaussian import sample_bulge_magnitudes, AstroSpaceTransform
from castor.constants import GLOBAL_STRETCH_SCALE
from scipy.signal import fftconvolve
import os

def simulate_and_visualize_density_v2(n_target, title_suffix):
    # 1. Setup Simulation Params
    img_size = 256
    sigma = 1.5
    exp_time = 60.0
    zp = 26.5
    sky_mag = 22.0
    pixel_scale = 0.11
    
    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (pixel_scale**2) * exp_time
    rc_mag, rc_sigma, rc_enhancement = 15.5, 0.4, 3.0
    gamma = 0.3
    
    # 2. Render massive population
    print(f"Generating {n_target:,} stars...")
    mags = sample_bulge_magnitudes(n_target, rc_mag, rc_sigma, rc_enhancement, 
                                   m_min=12.0, m_max=32.0, gamma=gamma)
    n_actual = len(mags)
    fluxes = exp_time * (10 ** (-0.4 * (mags - zp)))
    x = np.random.uniform(0, img_size, n_actual)
    y = np.random.uniform(0, img_size, n_actual)
    
    star_grid, _, _ = np.histogram2d(y, x, bins=img_size, range=[[0, img_size], [0, img_size]], weights=fluxes)
    
    # PSF Kernel
    k_size = 31
    half = k_size // 2
    gy, gx = np.meshgrid(np.arange(k_size), np.arange(k_size))
    kernel = np.exp(-((gx - half)**2 + (gy - half)**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    
    # Peak value of the normalized PSF (for sigma=1.5, peak is ~0.07)
    psf_peak = 1.0 / (2 * np.pi * sigma**2)
    
    star_signal = fftconvolve(star_grid, kernel, mode='same')
    raw_image = star_signal + sky_level
    img_noisy = raw_image + np.random.normal(0, np.sqrt(np.maximum(raw_image, 0) + 25.0))
    
    # --- 3. LOCAL Confusion SNR Analysis (FIX) ---
    # Sample the local clean photon background at each star center
    ix = np.clip(x.astype(int), 0, img_size - 1)
    iy = np.clip(y.astype(int), 0, img_size - 1)
    total_local_light = star_signal[iy, ix]
    
    # Local Background = Total light at center - This star's own peak contribution
    local_background = np.maximum(0, total_local_light - (fluxes * psf_peak))
    
    n_pix = 4 * np.pi * (sigma**2)
    # SNR = Flux / sqrt(Flux + n_pix * (Sky + LocalBackground + ReadNoise^2))
    noise_variance = fluxes + n_pix * (sky_level + local_background + 25.0)
    snrs = fluxes / np.sqrt(noise_variance)
    
    detectable_mask = snrs >= 5.0
    n_detectable = np.sum(detectable_mask)
    
    chunk_median = np.median(img_noisy)
    print(f"  Final Median: {chunk_median:,.2f}")
    print(f"  Detectable (Local Confusion SNR >= 5): {n_detectable:,}")

    # 4. Visualization
    transform = AstroSpaceTransform(stretch_scale=GLOBAL_STRETCH_SCALE)
    network_input = transform.image_to_network(img_noisy, chunk_median)
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Telescope View
    im0 = axes[0].imshow(img_noisy, vmin=np.percentile(img_noisy, 1), vmax=np.percentile(img_noisy, 99), cmap='viridis')
    axes[0].set_title(f"Telescope View ({title_suffix})\nMedian: {chunk_median:,.0f}")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Model View
    im1 = axes[1].imshow(network_input, cmap='magma')
    axes[1].set_title(f"Model View\n{n_detectable:,} Targets (Local SNR)")
    
    d_indices = np.where(detectable_mask)[0]
    if len(d_indices) > 5000:
        d_indices = np.random.choice(d_indices, 5000, replace=False)
    axes[1].scatter(x[d_indices], y[d_indices], s=20, edgecolors='cyan', facecolors='none', alpha=0.5)

    # Magnitude Histogram
    hist_mask = mags <= 26.0
    axes[2].hist(mags[hist_mask], bins=50, color='gray', alpha=0.3, label='All Stars')
    axes[2].hist(mags[detectable_mask & hist_mask], bins=50, color='cyan', alpha=0.7, label='Local Detectable')
    axes[2].set_xlabel("Magnitude")
    axes[2].set_ylabel("Count")
    axes[2].set_title(f"Detectability Curve\n(Crowding-Limited SNR)")
    axes[2].invert_xaxis()
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"density_check_local_snr_{title_suffix.replace(' ', '_')}.png"
    plt.savefig(out_path, dpi=150)
    print(f"✅ Saved to {out_path}")

if __name__ == "__main__":
    simulate_and_visualize_density_v2(2800000, "2.8M Stars")
    simulate_and_visualize_density_v2(8700000, "8.7M Stars")
