import torch
import numpy as np
from astropy.io import fits
from castor.data.stage0_gaussian import GaussianPretrainingProvider
from castor.cloud.config_utils import load_config
import os

def test_generation(config_path="config/config.yaml", output_path="test_stage0_sample.fits"):
    config = load_config(config_path)
    data_cfg = config["data_params"]
    
    print("Testing Stage 0 Generation with new Speed-Hack & Astrophysical Priors...")
    
    provider = GaussianPretrainingProvider(
        min_stars=data_cfg["min_stars"],
        max_stars=data_cfg["max_stars"],
        image_size=data_cfg["image_size"],
        max_capacity_per_cell=data_cfg["max_capacity_per_cell"],
        shape_size=data_cfg["shape_size"]
    )
    
    # Generate a single chunk
    sample = provider.generate_chunk()
    
    # 1. Authentic Image (Stretched for Network)
    img_network = sample["image"].squeeze().numpy()
    
    # 2. Raw Physical Image (Absolute counts)
    img_raw = sample["raw_image"].numpy()
    
    # 3. Physics-only image (Noiseless)
    img_physics = sample["physics_image"].numpy()
    
    # 4. Target Star Mask
    base_grid = sample["base_grid"].numpy()
    star_mask = (np.sum(base_grid[..., 0], axis=-1) > 0).astype(np.float32)
    
    # Save to multi-extension FITS
    primary_hdu = fits.PrimaryHDU(img_raw)
    primary_hdu.header['EXTNAME'] = 'RAW_IMAGE'
    primary_hdu.header['SKY'] = sample['sky_level']
    primary_hdu.header['EXP'] = sample['exp_time']
    primary_hdu.header['ZP'] = sample['zp']
    
    hdul = fits.HDUList([
        primary_hdu,
        fits.ImageHDU(img_physics, name="CLEAN_PHYSICS"),
        fits.ImageHDU(img_network, name="NETWORK_INPUT"),
        fits.ImageHDU(star_mask, name="STAR_MASK_GRID")
    ])
    
    hdul.writeto(output_path, overwrite=True)
    print(f"✅ Test sample saved to {output_path}")
    print(f"   Stats: Sky={sample['sky_level']:.2f}, Median={sample['chunk_median']:.2f}")

if __name__ == "__main__":
    test_generation()
