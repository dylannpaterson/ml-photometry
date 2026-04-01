import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from castor.data.stage0_gaussian import sample_bulge_magnitudes

def generate_completeness_grid():
    # 1. Setup Parameters
    exp_times = [30, 60, 120]
    densities = np.linspace(100000, 10000000, 20) # 100k to 10M stars
    mags = np.linspace(18, 28, 50)
    
    sigma = 1.5
    zp = 26.5
    sky_mag = 22.0
    pixel_scale = 0.11
    n_pix = 4 * np.pi * (sigma ** 2)
    img_size = 256
    area = img_size ** 2

    # 2. Build the model
    fig, axes = plt.subplots(1, len(exp_times), figsize=(20, 6), sharey=True)
    
    for ax, exp_time in zip(axes, exp_times):
        sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (pixel_scale**2) * exp_time
        instrumental_variance = sky_level + 25.0
        
        completeness_grid = np.zeros((len(densities), len(mags)))
        
        for i, n_stars in enumerate(densities):
            # Estimate Confusion Noise for this density
            # Using our same LF logic
            test_mags = sample_bulge_magnitudes(100000, 15.5, 0.4, 3.0, m_min=12, m_max=32, gamma=0.3)
            test_fluxes = exp_time * (10 ** (-0.4 * (test_mags - zp)))
            
            # Find the visibility boundary for the "sea"
            snr_boundary = 3.0
            a_q, b_q, c_q = 1, -(snr_boundary**2), -(snr_boundary**2) * n_pix * instrumental_variance
            flux_boundary = (-b_q + np.sqrt(b_q**2 - 4*a_q*c_q)) / 2.0
            
            faint_fluxes = test_fluxes[test_fluxes < flux_boundary]
            # Scale the sea flux/variance by the actual requested density ratio
            scale_factor = n_stars / 100000.0
            mean_sea_flux = (np.sum(faint_fluxes) / area) * scale_factor
            var_conf = (np.sum(faint_fluxes**2) / (area * n_pix)) * scale_factor
            
            total_per_pixel_var = instrumental_variance + mean_sea_flux + var_conf
            
            # Calculate completeness for every mag in the grid
            for j, m in enumerate(mags):
                f = exp_time * (10 ** (-0.4 * (m - zp)))
                snr = f / np.sqrt(f + n_pix * total_per_pixel_var)
                # Sigmoid mapping
                comp = 1.0 / (1.0 + np.exp(-2.0 * (snr - 5.0)))
                completeness_grid[i, j] = comp

        im = ax.imshow(completeness_grid, extent=[mags.max(), mags.min(), densities.min()/1e6, densities.max()/1e6], 
                       aspect='auto', cmap='RdYlGn', origin='lower')
        ax.set_title(f"Exp Time: {exp_time}s")
        ax.set_xlabel("Magnitude")
        if exp_time == exp_times[0]:
            ax.set_ylabel("Density (Millions of Stars / 256x256)")

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Completeness Probability')
    
    plt.suptitle("Analytical Completeness Model: Detectability vs. Crowding", fontsize=16)
    plt.savefig("completeness_model_analysis.png", dpi=150)
    print("✅ Completeness model visualization saved to completeness_model_analysis.png")

if __name__ == "__main__":
    generate_completeness_grid()
