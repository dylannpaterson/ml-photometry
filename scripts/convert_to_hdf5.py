import os
import h5py
import numpy as np
import torch
from tqdm import tqdm
import gc
from castor.data.stage0_gaussian import fast_paint_grid
from castor.data.transforms import AstroSpaceTransform
from castor.constants import DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL, SHAPE_SIZE, GLOBAL_STRETCH_SCALE, N_PCA_COMPONENTS

def create_hdf5_dataset(data_dir, output_path, total_samples=50000, img_size=256):
    cell_size = DEFAULT_CELL_SIZE
    grid_size = img_size // cell_size
    K = MAX_CAPACITY_PER_CELL
    N_PCA = N_PCA_COMPONENTS
    transform = AstroSpaceTransform(stretch_scale=GLOBAL_STRETCH_SCALE)
    
    # Target buffer shape: (grid_size, grid_size, K * (5 + N_PCA) + 1)
    target_shape = (grid_size, grid_size, K * (5 + N_PCA) + 1)
    
    # Discover mosaics
    mosaics = []
    mosaic_dir = os.path.join(data_dir, "mosaics")
    if not os.path.exists(mosaic_dir):
        print(f"❌ Error: Mosaic directory {mosaic_dir} not found.")
        return

    for f in os.listdir(mosaic_dir):
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
        return

    print(f"Found {len(mosaics)} mosaics. Creating HDF5 database at {output_path}...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, 'w') as h5f:
        images_ds = h5f.create_dataset("images", (total_samples, 1, img_size, img_size), dtype='float32', chunks=(1, 1, img_size, img_size))
        targets_ds = h5f.create_dataset("targets", (total_samples, *target_shape), dtype='float32', chunks=(1, *target_shape))
        
        psf_ds = None
        medians_ds = h5f.create_dataset("chunk_medians", (total_samples,), dtype='float32')

        samples_per_mosaic = total_samples // len(mosaics)
        current_idx = 0
        
        for m_idx, mosaic in enumerate(mosaics):
            print(f"Processing mosaic {m_idx + 1}/{len(mosaics)}: {os.path.basename(mosaic['img'])}")
            
            img_data = np.load(mosaic['img'])
            cat_data = np.load(mosaic['cat'])
            
            # Library stores [N_PCA + 1, 961]
            if mosaic['lib']:
                psf_lib = np.load(mosaic['lib'])
            else:
                # Fallback should not happen with new mosaics
                psf_lib = np.zeros((N_PCA + 1, SHAPE_SIZE * SHAPE_SIZE), dtype=np.float32)

            if psf_ds is None:
                psf_library_shape = psf_lib.shape 
                psf_ds = h5f.create_dataset(
                    "psf_libraries", 
                    (total_samples, *psf_library_shape),
                    dtype='float32',
                    chunks=(1, *psf_library_shape)
                )
                print(f"Allocated PSF library dataset with shape: {psf_ds.shape}")
                
            snrs, comps = cat_data['snr'], cat_data['comp']
            my, mx = img_data.shape
            pixel_scale = 0.11
            sky_level = (10 ** (-0.4 * (mosaic['sky_mag'] - mosaic['zp']))) * (pixel_scale**2) * mosaic['exp_time']
            
            this_mosaic_samples = samples_per_mosaic
            if m_idx == len(mosaics) - 1:
                this_mosaic_samples = total_samples - current_idx

            for _ in tqdm(range(this_mosaic_samples), leave=False):
                if current_idx >= total_samples: break
                    
                py = np.random.randint(0, my - img_size)
                px = np.random.randint(0, mx - img_size)
                
                star_signal = img_data[py:py+img_size, px:px+img_size]
                chunk_median = np.median(star_signal) + sky_level
                signal_tensor = np.expand_dims(np.clip(star_signal + sky_level, 0, None), axis=0).astype(np.float32)
                
                y_start = np.searchsorted(cat_data['y'], py)
                y_end = np.searchsorted(cat_data['y'], py + img_size)
                band_cat = cat_data[y_start:y_end]
                mask_x = (band_cat['x'] >= px) & (band_cat['x'] < px + img_size)
                
                target_buffer = np.zeros(target_shape, dtype=np.float32)
                
                if mask_x.any():
                    local_cat = band_cat[mask_x]
                    lx, ly = local_cat['x'] - px, local_cat['y'] - py
                    fluxes = local_cat['flux']
                    local_snrs = snrs[y_start:y_end][mask_x]
                    local_comps = comps[y_start:y_end][mask_x]
                    
                    # Extract continuous PCA weights from catalog
                    psf_weights = np.column_stack([local_cat[f'w{i}'] for i in range(N_PCA)])
                    
                    sort_idx = np.argsort(fluxes)[::-1]
                    grid_stars_np = fast_paint_grid(
                        lx, ly, fluxes, local_snrs, local_comps, psf_weights, sort_idx, 
                        5.0, grid_size, cell_size, K
                    )
                    target_buffer[:, :, :-1] = grid_stars_np.reshape(grid_size, grid_size, -1)
                
                target_buffer[:, :, -1] = transform.target_bg_to_network(sky_level - chunk_median)
                
                images_ds[current_idx] = signal_tensor
                targets_ds[current_idx] = target_buffer
                psf_ds[current_idx] = psf_lib
                medians_ds[current_idx] = chunk_median
                
                current_idx += 1
                
            del img_data, cat_data
            gc.collect()

    print(f"✅ HDF5 dataset complete! Saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert .npy mosaics to HDF5 database")
    parser.add_argument("--data_dir", default="data/bulge_stage0_full", help="Directory containing mosaics")
    parser.add_argument("--train_samples", type=int, default=50000, help="Number of training samples")
    parser.add_argument("--val_samples", type=int, default=2000, help="Number of validation samples")
    args = parser.parse_args()

    create_hdf5_dataset(args.data_dir, os.path.join(args.data_dir, "stage0_train.h5"), total_samples=args.train_samples)
    create_hdf5_dataset(args.data_dir, os.path.join(args.data_dir, "stage0_val.h5"), total_samples=args.val_samples)
