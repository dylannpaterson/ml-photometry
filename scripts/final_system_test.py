import numpy as np
import h5py
import os
import yaml
from scipy.ndimage import center_of_mass
from castor.data.stage0_gaussian import run_stage0_parallel_generation
from castor.constants import MAX_CAPACITY_PER_CELL

def final_system_test():
    # 1. Setup config for a single sparse sample
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    test_dir = "data/final_test"
    os.makedirs(test_dir, exist_ok=True)
    
    config["data_params"]["num_train_samples"] = 1
    config["curriculum"]["stage0"]["data_dir"] = test_dir
    config["data_params"]["min_stars"] = 50 # Sparse
    config["data_params"]["max_stars"] = 100
    
    # Remove physical density overrides so sparse test works
    if "density_26_min" in config["data_params"]:
        del config["data_params"]["density_26_min"]
    if "density_26_max" in config["data_params"]:
        del config["data_params"]["density_26_max"]
    
    print("🚀 Generating production-style test sample...")
    run_stage0_parallel_generation(config, split='train')
    
    h5_path = os.path.join(test_dir, "stage0_train.h5")
    
    # 2. Extract truth and image
    with h5py.File(h5_path, 'r') as f:
        img = f['images'][0][0] # (256, 256)
        target = f['targets'][0] # (64, 64, K*5+1)
        med = f['chunk_medians'][0]

    # 3. Find the brightest isolated star in the target grid
    grid_size = target.shape[0]
    K = MAX_CAPACITY_PER_CELL
    cell_size = 256 // grid_size
    
    star_targets = target[:, :, :-1].reshape(grid_size, grid_size, K, 5)
    p_map = star_targets[:, :, :, 0]
    dx_map = star_targets[:, :, :, 1]
    dy_map = star_targets[:, :, :, 2]
    flux_map = star_targets[:, :, :, 3]
    
    # Get all active stars
    active = p_map > 0.9
    y_idx, x_idx, k_idx = np.where(active)
    
    if len(x_idx) == 0:
        print("❌ Error: No stars found in generated sample.")
        return

    # Pick the brightest one
    fluxes = flux_map[active]
    brightest_idx = np.argmax(fluxes)
    
    # Calculate truth px
    truth_x = x_idx[brightest_idx] * cell_size + dx_map[y_idx[brightest_idx], x_idx[brightest_idx], k_idx[brightest_idx]]
    truth_y = y_idx[brightest_idx] * cell_size + dy_map[y_idx[brightest_idx], x_idx[brightest_idx], k_idx[brightest_idx]]
    
    # 4. Measure centroid in image
    # Crop a 15x15 window around truth
    ix, iy = int(round(truth_x)), int(round(truth_y))
    window = img[iy-7:iy+8, ix-7:ix+8]
    # Subtract local median to remove background/neighbor influence
    window -= np.median(window)
    window = np.maximum(window, 0)
    
    wy, wx = center_of_mass(window)
    # Map back to global
    meas_x = wx + (ix - 7)
    meas_y = wy + (iy - 7)
    
    offset_x = meas_x - truth_x
    offset_y = meas_y - truth_y
    
    print(f"\n--- Final System Validation ---")
    print(f"🌟 Star Location:  ({truth_x:.4f}, {truth_y:.4f})")
    print(f"🔭 Measured Peak: ({meas_x:.4f}, {meas_y:.4f})")
    print(f"⚠️  Total System Shift: ({offset_x:.6f}, {offset_y:.6f})")
    
    if abs(offset_x) < 1e-3 and abs(offset_y) < 1e-3:
        print("\n✅ SYSTEM VERIFIED: Data generation logic is perfectly aligned.")
    else:
        print("\n❌ SYSTEM FAILURE: Sub-pixel shift detected in production pipeline.")

if __name__ == "__main__":
    final_system_test()
