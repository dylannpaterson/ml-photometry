import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from castor.data.stage0_gaussian import GaussianMosaicDataset
from castor.constants import GLOBAL_STRETCH_SCALE

def test_mosaic_completeness_recovery():
    data_dir = "data/local_fast/mosaics"
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print(f"❌ Error: {data_dir} is empty or missing. Run generation script first.")
        return

    # 1. Initialize Dataset (which now pre-calculates the spline per mosaic)
    dataset = GaussianMosaicDataset(
        data_dir=data_dir,
        num_samples=100,
        image_size=256
    )
    

    # 2. Sample multiple crops and aggregate stars
    all_local_mags = []
    all_local_snrs = []
    all_local_comps = []
    
    print("Sampling 50 crops from the dataset...")
    for i in range(50):
        # We need to reach into the internal state to get the actual SNR/Mag for verification
        # __getitem__ normally only returns the target grid
        dataset.__getitem__(i) 
        
        # Pull the last sliced band/mask data
        # We'll use the active_cat and the calculated snrs/comps from the last __getitem__ call
        # Note: In a real test, we might modify the dataset to return these for verification
        # For now, let's look at the active_cat we just loaded
        pass

    # Actually, a better way to test is to compare the GLOBAL distribution 
    # since we calculate the spline ONCE for the whole mosaic.
    
    # Let's force a load
    dataset._load_mosaic_to_ram(0)
    
    full_mags = dataset.active_cat['mag']
    full_snrs = dataset.active_snrs
    full_comps = dataset.active_comps
    
    mask_detected = full_snrs >= 5.0
    detected_mags = full_mags[mask_detected]
    detected_comps = full_comps[mask_detected]
    
    # 3. Plotting Verification
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    bins = np.linspace(14, 28, 60)
    
    # recovery check: Sum(1/c) for detected stars should match the Truth (All stars)
    ax1.hist(full_mags, bins=bins, histtype='step', label='Truth (Whole Mosaic)', color='black', lw=2)
    ax1.hist(detected_mags, bins=bins, histtype='step', label='Detected (SNR >= 5)', color='red', lw=1.5)
    
    weights = 1.0 / np.maximum(detected_comps, 0.05)
    ax1.hist(detected_mags, bins=bins, weights=weights, histtype='step', 
             label='Recovered (Weighted by 1/Global Spline)', color='blue', ls='--', lw=2)
    
    ax1.set_xlabel('Magnitude')
    ax1.set_ylabel('Count')
    ax1.set_yscale('log')
    ax2.set_xlim(13.5,28.5)
    ax1.set_title('Mosaic-Level Completeness Recovery (Verification of Global Spline Optimization)')
    ax1.legend()
    ax1.grid(True, which='both', alpha=0.3)
    
    # Completeness Curve check
    bin_means = []
    bin_centers = (bins[:-1] + bins[1:]) / 2
    for i in range(len(bins)-1):
        mask = (full_mags >= bins[i]) & (full_mags < bins[i+1])
        if mask.any():
            # The EMPIRICAL completeness is (Detected / Total) in this bin
            total = mask.sum()
            det = (mask & mask_detected).sum()
            bin_means.append(det / (total + 1e-9))
        else:
            bin_means.append(np.nan)
            
    ax2.plot(bin_centers, bin_means, 'ro', alpha=0.5, label='Empirical (Detected/Total)')
    ax2.plot(full_mags[::100], full_comps[::100], 'b.', markersize=1, alpha=0.1, label='Spline Mapping')
    
    ax2.set_xlabel('Magnitude')
    ax2.set_ylabel('Completeness (c)')
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlim(13.5,28.5)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    output_path = "mosaic_completeness_verification.png"
    plt.savefig(output_path)
    print(f"✅ Mosaic analysis plots saved to {output_path}")

if __name__ == "__main__":
    test_mosaic_completeness_recovery()
