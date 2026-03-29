import os
import h5py
import numpy as np
import torch
from tqdm import tqdm
import gc
import shutil
from castor.data.stage0_gaussian import fast_paint_grid
from castor.data.transforms import AstroSpaceTransform
from castor.constants import DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL, SHAPE_SIZE, GLOBAL_STRETCH_SCALE, N_PCA_COMPONENTS

def create_hdf5_datasets_combined(data_dir, train_path, val_path, train_samples=50000, val_samples=2000, img_size=256):
    cell_size = DEFAULT_CELL_SIZE
    grid_size = img_size // cell_size
    K = MAX_CAPACITY_PER_CELL
    N_PCA = N_PCA_COMPONENTS
    transform = AstroSpaceTransform(stretch_scale=GLOBAL_STRETCH_SCALE)
    
    target_shape = (grid_size, grid_size, K * (5 + N_PCA) + 1)
    
    # Discover mosaics
    mosaics = []
    mosaic_dir = os.path.join(data_dir, "mosaics")
    if not os.path.exists(mosaic_dir):
        print(f"❌ Error: Mosaic directory {mosaic_dir} not found.")
        return False

    for f in sorted(os.listdir(mosaic_dir)):
        if f.endswith("_img.npy"):
            base = f.replace("_img.npy", "")
            meta_path = os.path.join(mosaic_dir, f"{base}_meta.npy")
            lib_path = os.path.join(mosaic_dir, f"{base}_psf_lib.npy")
            cat_path = os.path.join(mosaic_dir, f"{base}_cat.npy")
            
            if not os.path.exists(meta_path) or not os.path.exists(cat_path):
                continue

            meta = np.load(meta_path)
            mosaics.append({
                'img': os.path.join(mosaic_dir, f),
                'cat': cat_path,
                'lib': lib_path if os.path.exists(lib_path) else None,
                'exp_time': meta[0], 'zp': meta[1], 'sky_mag': meta[2]
            })

    if not mosaics:
        print(f"❌ Error: No mosaics found in {mosaic_dir}")
        return False

    print(f"Found {len(mosaics)} mosaics. Creating combined HDF5 databases...")
    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    with h5py.File(train_path, 'w') as h5_train, h5py.File(val_path, 'w') as h5_val:
        # 1. Setup Train Datasets
        tr_imgs = h5_train.create_dataset("images", (train_samples, 1, img_size, img_size), dtype='float32', chunks=(1, 1, img_size, img_size), compression="lzf")
        tr_tgts = h5_train.create_dataset("targets", (train_samples, *target_shape), dtype='float16', chunks=(1, *target_shape), compression="lzf")
        tr_meds = h5_train.create_dataset("chunk_medians", (train_samples,), dtype='float32')
        tr_psfs = None # Allocated dynamically
        
        # 2. Setup Val Datasets
        val_imgs = h5_val.create_dataset("images", (val_samples, 1, img_size, img_size), dtype='float32', chunks=(1, 1, img_size, img_size), compression="lzf")
        val_tgts = h5_val.create_dataset("targets", (val_samples, *target_shape), dtype='float16', chunks=(1, *target_shape), compression="lzf")
        val_meds = h5_val.create_dataset("chunk_medians", (val_samples,), dtype='float32')
        val_psfs = None

        tr_per_mos = train_samples // len(mosaics)
        val_per_mos = val_samples // len(mosaics)
        
        tr_idx, v_idx = 0, 0
        
        for m_idx, mosaic in enumerate(mosaics):
            print(f"Processing mosaic {m_idx + 1}/{len(mosaics)}: {os.path.basename(mosaic['img'])}")
            
            img_data = np.load(mosaic['img'])
            cat_data = np.load(mosaic['cat'])
            psf_lib = np.load(mosaic['lib']) if mosaic['lib'] else np.zeros((N_PCA + 1, SHAPE_SIZE * SHAPE_SIZE), dtype=np.float32)

            if tr_psfs is None:
                tr_psfs = h5_train.create_dataset("psf_libraries", (train_samples, *psf_lib.shape), dtype='float32', chunks=(1, *psf_lib.shape), compression="lzf")
                val_psfs = h5_val.create_dataset("psf_libraries", (val_samples, *psf_lib.shape), dtype='float32', chunks=(1, *psf_lib.shape), compression="lzf")

            pixel_scale = 0.11
            sky_level = (10 ** (-0.4 * (mosaic['sky_mag'] - mosaic['zp']))) * (pixel_scale**2) * mosaic['exp_time']
            my, mx = img_data.shape
            snrs, comps = cat_data['snr'], cat_data['comp']

            # Determine number of samples for this mosaic
            this_tr = tr_per_mos if m_idx < len(mosaics)-1 else (train_samples - tr_idx)
            this_val = val_per_mos if m_idx < len(mosaics)-1 else (val_samples - v_idx)

            # --- Unified Sampling Loop ---
            for is_val, num_samples in [(False, this_tr), (True, this_val)]:
                ds_imgs = val_imgs if is_val else tr_imgs
                ds_tgts = val_tgts if is_val else tr_tgts
                ds_meds = val_meds if is_val else tr_meds
                ds_psfs = val_psfs if is_val else tr_psfs
                
                for _ in range(num_samples):
                    curr = v_idx if is_val else tr_idx
                    
                    py = np.random.randint(0, my - img_size)
                    px = np.random.randint(0, mx - img_size)
                    
                    # 1. Physics
                    star_signal = img_data[py:py+img_size, px:px+img_size]
                    chunk_median = np.median(star_signal) + sky_level
                    signal_tensor = np.expand_dims(np.clip(star_signal + sky_level, 0, None), axis=0).astype(np.float32)
                    
                    # 2. Target Painting
                    y_start = np.searchsorted(cat_data['y'], py)
                    y_end = np.searchsorted(cat_data['y'], py + img_size)
                    band_cat = cat_data[y_start:y_end]
                    mask_x = (band_cat['x'] >= px) & (band_cat['x'] < px + img_size)
                    
                    target_buffer = np.zeros(target_shape, dtype=np.float32)
                    if mask_x.any():
                        local_cat = band_cat[mask_x]
                        lx, ly = local_cat['x'] - px, local_cat['y'] - py
                        local_snrs, local_comps = snrs[y_start:y_end][mask_x], comps[y_start:y_end][mask_x]
                        psf_weights = np.column_stack([local_cat[f'w{i}'] for i in range(N_PCA)])
                        
                        sort_idx = np.argsort(local_cat['flux'])[::-1]
                        grid_stars = fast_paint_grid(lx, ly, local_cat['flux'], local_snrs, local_comps, psf_weights, sort_idx, 5.0, grid_size, cell_size, K)
                        target_buffer[:, :, :-1] = grid_stars.reshape(grid_size, grid_size, -1)
                    
                    target_buffer[:, :, -1] = transform.target_bg_to_network(sky_level - chunk_median)
                    
                    # 3. Write
                    ds_imgs[curr] = signal_tensor
                    ds_tgts[curr] = target_buffer
                    ds_meds[curr] = chunk_median
                    ds_psfs[curr] = psf_lib
                    
                    if is_val: v_idx += 1
                    else: tr_idx += 1

            # --- CLEANUP THIS MOSAIC ---
            for f_path in [mosaic['img'], mosaic['cat'], mosaic['lib']]:
                if f_path and os.path.exists(f_path): os.remove(f_path)
            meta_f = mosaic['img'].replace("_img.npy", "_meta.npy")
            if os.path.exists(meta_f): os.remove(meta_f)
            
            del img_data, cat_data
            gc.collect()

    print(f"✅ Combined HDF5 datasets complete!")
    if os.path.exists(mosaic_dir):
        shutil.rmtree(mosaic_dir)
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert .npy mosaics to HDF5 database (Combined Pass)")
    parser.add_argument("--data_dir", default="data/bulge_stage0_full", help="Directory containing mosaics")
    parser.add_argument("--train_samples", type=int, default=50000, help="Number of training samples")
    parser.add_argument("--val_samples", type=int, default=2000, help="Number of validation samples")
    args = parser.parse_args()

    train_h5 = os.path.join(args.data_dir, "stage0_train.h5")
    val_h5 = os.path.join(args.data_dir, "stage0_val.h5")
    
    create_hdf5_datasets_combined(args.data_dir, train_h5, val_h5, train_samples=args.train_samples, val_samples=args.val_samples)
