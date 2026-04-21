import h5py
import numpy as np
import os
from castor.constants import MAX_CAPACITY_PER_CELL

def check_occupancy(h5_path):
    if not os.path.exists(h5_path):
        print(f"Error: {h5_path} not found.")
        return
    
    with h5py.File(h5_path, 'r') as f:
        targets = f['targets']
        metas = f['metas']
        
        num_samples = len(targets)
        print(f"Checking occupancy for {num_samples} samples in {h5_path}...\n")
        
        grid_size = targets.shape[1]
        K = MAX_CAPACITY_PER_CELL
        max_cap = (grid_size ** 2) * K
        
        print(f"{'Idx':<4} | {'Target Occ':<12} | {'Actual Active':<15} | {'Actual Occ':<12}")
        print("-" * 55)
        
        for i in range(num_samples):
            # Actual occupancy is stored in meta[4] in generate_single_sample_stage0
            actual_occ_meta = metas[i][4]
            fwhm_meta = metas[i][3]

            # Actual active count: stars with p > 0.05
            tgt = targets[i, :, :, :-1].reshape(grid_size, grid_size, K, 5)
            p_vals = tgt[..., 0]
            actual_active = np.sum(p_vals > 0.05)
            actual_occ_calc = actual_active / max_cap

            print(f"{i:<4} | {actual_occ_meta*100:<10.1f}% | {actual_active:<15} | {actual_occ_calc*100:<10.1f}% | FWHM: {fwhm_meta:.2f}")
if __name__ == "__main__":
    check_occupancy("data/stage0_test_batch/stage0_train.h5")
