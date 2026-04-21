import yaml
import os
import numpy as np
import h5py
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from castor.data.stage0_gaussian import generate_single_sample_stage0
from castor.constants import DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL, GLOBAL_STRETCH_SCALE
from castor.data.transforms import AstroSpaceTransform

def verify_occupancy(occ_value, name, output_h5):
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    d_cfg = config['data_params']
    chunk_size = d_cfg.get('image_size', 256)
    
    params = {
        'image_size': chunk_size,
        'min_stars_chunk': 0, # Not used when force_occupancy is present
        'max_stars_chunk': 0,
        'force_occupancy': occ_value
    }
    
    os.makedirs(os.path.dirname(output_h5), exist_ok=True)
    
    print(f"🧪 Generating verification sample for {name} ({occ_value*100:.0f}% occupancy)...")
    img, tgt, med, meta = generate_single_sample_stage0(0, params)
    
    with h5py.File(output_h5, 'w') as f:
        f.create_dataset("images", data=img[None, None, ...])
        f.create_dataset("targets", data=tgt[None, ...])
        f.create_dataset("chunk_medians", data=[med])
        f.create_dataset("metas", data=meta[None, ...])
    
    # Visualize
    transform = AstroSpaceTransform(stretch_scale=GLOBAL_STRETCH_SCALE)
    network_input = transform.image_to_network(img, med)
    
    grid_size = tgt.shape[0]
    K = MAX_CAPACITY_PER_CELL
    cell_size = chunk_size // grid_size
    
    star_targets = tgt[:, :, :-1].reshape(grid_size, grid_size, K, 5)
    p_map = star_targets[:, :, :, 0]
    dx_map = star_targets[:, :, :, 1]
    dy_map = star_targets[:, :, :, 2]
    snr_map = star_targets[:, :, :, 4]
    
    active_mask = p_map > 0.05
    cy, cx, ck = np.where(active_mask)
    target_px = cx * cell_size + dx_map[active_mask]
    target_py = cy * cell_size + dy_map[active_mask]
    det_mask = snr_map[active_mask] >= 5.0

    plt.figure(figsize=(10, 10))
    plt.imshow(network_input, cmap='magma', origin='lower')
    plt.scatter(target_px[det_mask], target_py[det_mask], s=20, edgecolors='cyan', facecolors='none', alpha=0.5, label='SNR >= 5')
    plt.title(f"Stage 0 Verification: {name} ({occ_value*100:.0f}% Occupancy)\nStars SNR>1: {len(target_px)}")
    plt.legend()
    out_img = f"verify_{name.lower().replace(' ', '_')}.png"
    plt.savefig(out_img)
    plt.close()
    print(f"✅ Visualization saved to {out_img}")

if __name__ == "__main__":
    verify_occupancy(0.05, "Min Occupancy", "data/verify_min.h5")
    verify_occupancy(0.90, "Max Occupancy", "data/verify_max.h5")
