import h5py
import numpy as np
import os
import yaml

def debug_star_density(h5_path):
    if not os.path.exists(h5_path):
        print(f"❌ Error: {h5_path} not found.")
        return

    with h5py.File(h5_path, 'r') as f:
        images = f['images']
        targets = f['targets']
        
        num_samples = len(images)
        print(f"Inspecting {num_samples} samples in {h5_path}...")
        
        all_counts_snr2 = []
        all_counts_total = []
        
        # Sample 100 random chunks
        indices = np.random.choice(num_samples, min(100, num_samples), replace=False)
        
        for idx in indices:
            target = targets[idx]
            # Target structure: [64, 64, K*5 + 1]
            # Star slot: [p, dx, dy, flux, snr]
            grid_size = target.shape[0]
            K = 3 # Based on MAX_CAPACITY_PER_CELL
            
            star_data = target[:, :, :-1].reshape(grid_size, grid_size, K, 5)
            
            # snr is at index 4
            snrs = star_data[:, :, :, 4]
            p_vals = star_data[:, :, :, 0]
            
            # Active stars have p > 0 (or SNR > 1 by definition in generation)
            active_mask = p_vals > 0
            snr2_mask = (p_vals > 0) & (snrs > 2.0)
            
            all_counts_total.append(np.sum(active_mask))
            all_counts_snr2.append(np.sum(snr2_mask))
            
        print(f"\nResults over {len(indices)} sampled chunks (256x256):")
        print(f"  Avg Total Active Stars (SNR > 1): {np.mean(all_counts_total):.1f}")
        print(f"  Avg Stars with SNR > 2:           {np.mean(all_counts_snr2):.1f}")
        print(f"  Min Stars SNR > 2:                {np.min(all_counts_snr2)}")
        print(f"  Max Stars SNR > 2:                {np.max(all_counts_snr2)}")
        
        # Scaling back to 1024x1024
        print(f"\nExtrapolated to 1024x1024 (x16):")
        print(f"  Avg Stars with SNR > 2:           {np.mean(all_counts_snr2) * 16:.1f}")

if __name__ == "__main__":
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    val_h5 = os.path.join(config['curriculum']['stage0']['data_dir'], "stage0_val.h5")
    debug_star_density(val_h5)
