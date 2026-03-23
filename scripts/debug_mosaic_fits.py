import torch
import numpy as np
from astropy.io import fits
from castor.data.stage0_gaussian import GaussianPretrainingProvider
from castor.cloud.config_utils import load_config
import os

def generate_fits_mosaic(config_path="config/config.yaml", output_path="mosaic_debug.fits"):
    config = load_config(config_path)
    data_cfg = config["data_params"]
    stage_cfg = config["curriculum"]["stage0"]
    
    # Scale stars for a full mosaic size (4088x4088)
    mosaic_size = 4088
    training_size = data_cfg['image_size']
    area_ratio = (mosaic_size / training_size)**2
    
    sca_min_stars = int(data_cfg['min_stars'] * area_ratio * 0.8)
    sca_max_stars = int(data_cfg['max_stars'] * area_ratio * 0.8)
    
    print(f"Generating high-density mosaic ({mosaic_size}x{mosaic_size})...")
    print(f"Targeting approx {sca_max_stars} stars.")
    
    provider = GaussianPretrainingProvider(
        min_stars=sca_min_stars,
        max_stars=sca_max_stars,
        image_size=mosaic_size,
        max_capacity_per_cell=data_cfg['max_capacity_per_cell'],
        shape_size=data_cfg['shape_size']
    )
    
    sample = provider.generate_chunk()
    img = sample["raw_image"].numpy()
    
    # Create a simple mask of where stars are in the target grid
    # base_grid is [G, G, K, 5]
    base_grid = sample["base_grid"].numpy()
    star_mask = (np.sum(base_grid[..., 0], axis=-1) > 0).astype(np.float32)
    
    # Save to FITS
    primary_hdu = fits.PrimaryHDU(img)
    primary_hdu.header['EXTNAME'] = 'RAW_IMAGE'
    
    hdul = fits.HDUList([
        primary_hdu,
        fits.ImageHDU(star_mask, name="STAR_MASK_GRID")
    ])
    
    hdul.writeto(output_path, overwrite=True)
    print(f"✅ Mosaic saved to {output_path}")

if __name__ == "__main__":
    generate_fits_mosaic()
