import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from castor.data.transforms import AstroSpaceTransform
from castor.constants import DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL, GLOBAL_STRETCH_SCALE

# Architecture Reference from generate_stage1_mosaics.py
ARCHETYPE_PARAMS = {
    'roman':       {'scale': 0.11,  'jitter': 0.015, 'full_well': 100000, 'read_noise': 5.0},
    'hubble':      {'scale': 0.128, 'jitter': 0.008, 'full_well': 80000,  'read_noise': 3.0},
    'ideal_space': {'scale': 0.10,  'jitter': 0.002, 'full_well': 1000000,'read_noise': 1.0},
    'ground':      {'scale': 0.34,  'jitter': 0.050, 'full_well': 200000, 'read_noise': 15.0}
}

class Stage1MacroSparseDataset(Dataset):
    """
    Implements the 'Cached Physics, Live Noise' pipeline for Stage 1.
    Loads clean physics mosaics and injects noise/detector effects on the fly.
    """
    def __init__(self, data_dir, num_samples=50000, image_size=256, cell_size=DEFAULT_CELL_SIZE, K=MAX_CAPACITY_PER_CELL, global_stretch_scale=GLOBAL_STRETCH_SCALE):
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.img_size = image_size
        self.cell_size = cell_size
        self.grid_size = image_size // cell_size
        self.K = K
        self.transform = AstroSpaceTransform(stretch_scale=global_stretch_scale)

        
        # 1. Discover Mosaics
        self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])
        self.catalog_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".parquet")])
        
        if not self.image_files:
            raise FileNotFoundError(f"No Stage 1 mosaics found in {data_dir}. Run scripts/generate_stage1_mosaics.py first.")
            
        print(f"🔗 Stage 1 Macro-Sparse: Memory-mapping {len(self.image_files)} physics mosaics...")
        self.mosaics = []
        for img_f, cat_f in zip(self.image_files, self.catalog_files):
            # Infer archetype from filename (e.g., mosaic_00_roman.npy)
            # Find the archetype part
            for arch in ARCHETYPE_PARAMS.keys():
                if arch in img_f:
                    archetype = arch
                    break
            else:
                archetype = 'roman' # Fallback
                
            img_mmap = np.load(os.path.join(data_dir, img_f), mmap_mode='r')
            catalog = pd.read_parquet(os.path.join(data_dir, cat_f))
            
            # Extract metadata from the catalog if available
            if 'exp_time' in catalog.columns:
                exp_time = catalog['exp_time'].iloc[0]
                zp = catalog['zp'].iloc[0]
                sky_mag = catalog['sky_mag'].iloc[0]
                scale = ARCHETYPE_PARAMS[archetype]['scale']
                # sky_counts_per_pixel = (exp_time * 10**(-0.4*(sky_mag - zp))) * scale^2
                sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (scale**2) * exp_time
            else:
                sky_level = 30.0 # Legacy fallback
                
            self.mosaics.append({
                'image': img_mmap, 
                'catalog': catalog, 
                'params': ARCHETYPE_PARAMS[archetype],
                'archetype': archetype,
                'sky_level': sky_level
            })

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Select random mosaic
        m_idx = np.random.randint(0, len(self.mosaics))
        mosaic = self.mosaics[m_idx]
        params = mosaic['params']
        
        # 2. Slice random clean crop
        full_h, full_w = mosaic['image'].shape
        py = np.random.randint(0, full_h - self.img_size)
        px = np.random.randint(0, full_w - self.img_size)
        
        clean_physics = mosaic['image'][py:py+self.img_size, px:px+self.img_size].copy()
        
        # --- NOISE PIPELINE (Live Injection) ---
        
        # 3. Add Sky Background
        sky_level = mosaic['sky_level'] * np.random.uniform(0.8, 1.2) # Randomized sky
        img_with_sky = clean_physics + sky_level
        
        # 4. Apply Poisson Noise (Photon Noise)
        # Poisson is only valid for positive values
        img_poisson = np.random.poisson(np.maximum(img_with_sky, 0)).astype(np.float32)
        
        # 5. Add Gaussian Read Noise
        read_noise = np.random.normal(0, params['read_noise'], size=img_poisson.shape)
        img_noisy = img_poisson + read_noise
        
        # 6. Detector Saturation (Hardware Clamp)
        img_saturated = np.clip(img_noisy, 0, params['full_well'])
        
        # 7. Normalize Image -> Network Space
        chunk_median = np.median(img_saturated)
        normalized_image = self.transform.image_to_network(img_saturated, chunk_median)
        image_tensor = torch.from_numpy(normalized_image).unsqueeze(0).float()

        # --- TARGET GENERATION (JIT Painting) ---
        
        # 8. Filter catalog for stars in crop
        cat = mosaic['catalog']
        mask = (cat['x'] >= px) & (cat['x'] < px + self.img_size) & \
               (cat['y'] >= py) & (cat['y'] < py + self.img_size)
        local_stars = cat[mask].copy()
        
        # Adjust coordinates to local crop
        local_stars['lx'] = local_stars['x'] - px
        local_stars['ly'] = local_stars['y'] - py
        
        # 9. Build Target Grid
        # NEW: Shape size is now determined by model constants
        # In this dataset, we use 81 (9x9) which matches SHAPE_SIZE=9 in constants
        grid_stars = torch.zeros((self.grid_size, self.grid_size, self.K, 5 + 81), dtype=torch.float32)
        
        # Assign to slots (Brightest-to-Faint)
        cell_assignments = {}
        for _, star in local_stars.iterrows():
            cx, cy = int(star['lx'] // self.cell_size), int(star['ly'] // self.cell_size)
            if (cy, cx) not in cell_assignments: cell_assignments[(cy, cx)] = []
            cell_assignments[(cy, cx)].append(star)
            
        for (cy, cx), stars in cell_assignments.items():
            if cy >= self.grid_size or cx >= self.grid_size: continue
            
            # Sort local cell stars by flux
            sorted_stars = sorted(stars, key=lambda s: s['flux'], reverse=True)
            
            for slot in range(min(self.K, len(sorted_stars))):
                star = sorted_stars[slot]
                
                # Dynamic Completeness Calculation (Penalized SNR)
                snr = star['flux'] / np.sqrt(star['flux'] + sky_level + params['read_noise']**2)
                completeness = 1.0 / (1.0 + np.exp(-2.0 * (snr - 5.0)))
                
                # [p, dx, dy, flux, c, shape...]
                grid_stars[cy, cx, slot, 0] = 1.0 # p
                grid_stars[cy, cx, slot, 1] = star['lx'] % self.cell_size # dx
                grid_stars[cy, cx, slot, 2] = star['ly'] % self.cell_size # dy
                grid_stars[cy, cx, slot, 3] = star['flux']
                grid_stars[cy, cx, slot, 4] = completeness
                grid_stars[cy, cx, slot, 5:] = torch.from_numpy(star['shape'].copy())

        # 10. Background Target
        bg_target_linear = sky_level - chunk_median
        bg_grid_stretched = self.transform.target_bg_to_network(np.full((self.grid_size, self.grid_size), bg_target_linear, dtype=np.float32))
        
        # Flatten
        flattened_stars = grid_stars.view(self.grid_size, self.grid_size, -1)
        target = torch.cat([flattened_stars, torch.from_numpy(bg_grid_stretched).unsqueeze(-1).float()], dim=-1)
        
        return image_tensor, target
