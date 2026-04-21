import numpy as np
import yaml
import os
import h5py
from tqdm import tqdm
from castor.data.stage0_gaussian import generate_single_sample_stage0
from castor.constants import DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL

def test_stage0_predictive_init(num_samples=20):
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    d_cfg = config['data_params']
    chunk_size = d_cfg['image_size']
    
    params = {
        'image_size': chunk_size,
        'exp_time_min': d_cfg['physics_params']['exp_time_min'],
        'exp_time_max': d_cfg['physics_params']['exp_time_max'],
        'zp': d_cfg['physics_params']['zp'],
        'sky_mag': d_cfg['physics_params']['sky_mag']
    }

    grid_size = chunk_size // DEFAULT_CELL_SIZE
    K = MAX_CAPACITY_PER_CELL
    output_path = "data/test_predictive_init.h5"
    os.makedirs("data", exist_ok=True)

    print(f"Generating {num_samples} samples to {output_path}...")
    
    with h5py.File(output_path, 'w') as f:
        dset_img = f.create_dataset("images", (num_samples, 256, 256), dtype='f4')
        dset_tgt = f.create_dataset("targets", (num_samples, 64, 64, K*5 + 1), dtype='f4')
        dset_med = f.create_dataset("chunk_medians", (num_samples,), dtype='f4')
        dset_meta = f.create_dataset("metas", (num_samples, 17), dtype='f4')

        for i in tqdm(range(num_samples)):
            img, target, median, meta = generate_single_sample_stage0(i, params)
            dset_img[i] = img
            dset_tgt[i] = target
            dset_med[i] = median
            dset_meta[i] = meta

    print(f"✅ Saved to {output_path}. Now run visualization.")

if __name__ == "__main__":
    test_stage0_predictive_init()
