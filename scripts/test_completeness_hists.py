import numpy as np
import matplotlib.pyplot as plt
from castor.data.stage0_gaussian import GaussianPretrainingProvider
from castor.constants import GLOBAL_STRETCH_SCALE

def test_completeness_histograms():
    # 1. Initialize Provider
    # Use a high star density to get good statistics for the histograms
    provider = GaussianPretrainingProvider(
        min_stars=2000000, 
        max_stars=4000000, 
        image_size=512, # Larger size for better stats
        min_snr=5.0
    )
    
    print("Generating test chunk...")
    sample = provider.generate_chunk()
    
    full_mags = sample["full_mags"]
    full_snrs = sample["full_snrs"]
    full_comps = sample["full_comps"]
    
    # Filtering for SNR >= 5.0
    mask_detected = full_snrs >= 5.0
    detected_mags = full_mags[mask_detected]
    detected_comps = full_comps[mask_detected]
    
    # 2. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # --- Top Plot: Histograms ---
    bins = np.linspace(14, 28, 60)
    
    # 1) All stars
    ax1.hist(full_mags, bins=bins, histtype='step', label='All Stars (Truth)', color='black', lw=2)
    
    # 2) All stars with SNR >= 5
    ax1.hist(detected_mags, bins=bins, histtype='step', label='Detected Stars (SNR >= 5)', color='red', lw=1.5)
    
    # 3) All stars with SNR >= 5 weighted by 1/c
    weights = 1.0 / np.maximum(detected_comps, 0.01)
    ax1.hist(detected_mags, bins=bins, weights=weights, histtype='step', 
             label='Detected Stars (Weighted by 1/c)', color='blue', ls='--', lw=2)
    
    ax1.set_xlabel('Magnitude')
    ax1.set_ylabel('Count')
    ax1.set_yscale('log')
    ax1.set_title('Completeness Recovery Verification')
    ax1.legend()
    ax1.grid(True, which='both', alpha=0.3)
    
    # --- Bottom Plot: Completeness Curve ---
    # Sort by magnitude to plot a clean line
    sort_idx = np.argsort(full_mags)
    ax2.scatter(full_mags[::10], full_comps[::10], s=1, alpha=0.2, color='gray', label='Individual Stars (sample)')
    
    # Plot the binned mean completeness
    bin_means = []
    bin_centers = (bins[:-1] + bins[1:]) / 2
    for i in range(len(bins)-1):
        mask = (full_mags >= bins[i]) & (full_mags < bins[i+1])
        if mask.any():
            bin_means.append(np.mean(full_comps[mask]))
        else:
            bin_means.append(np.nan)
            
    ax2.plot(bin_centers, bin_means, color='blue', lw=2, label='Mean Completeness c(mag)')
    
    ax2.set_xlabel('Magnitude')
    ax2.set_ylabel('Completeness (c)')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    output_path = "completeness_test_analysis.png"
    plt.savefig(output_path)
    print(f"✅ Analysis plots saved to {output_path}")

if __name__ == "__main__":
    test_completeness_histograms()
