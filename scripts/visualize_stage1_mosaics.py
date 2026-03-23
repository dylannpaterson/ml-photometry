import numpy as np
import pandas as pd
from astropy.io import fits
import os
import argparse

# Instrument Params for Authentic Pass
ARCHETYPE_PARAMS = {
    'roman':       {'sky': 30.0,  'read_noise': 5.0,  'full_well': 100000},
    'hubble':      {'sky': 20.0,  'read_noise': 3.0,  'full_well': 80000},
    'ideal_space': {'sky': 5.0,   'read_noise': 1.0,  'full_well': 1000000},
    'ground':      {'sky': 150.0, 'read_noise': 15.0, 'full_well': 200000}
}

def convert_mosaic_to_fits(npy_path, parquet_path, output_path):
    # Infer archetype
    archetype = npy_path.split("_")[-1].replace(".npy", "")
    params = ARCHETYPE_PARAMS[archetype]
    
    print(f"Loading {npy_path} ({archetype.upper()})...")
    image_data = np.load(npy_path)
    
    # --- AUTHENTIC DETECTOR PASS ---
    print(f"Applying Authentic Detector Layer (Sky={params['sky']}, RN={params['read_noise']})...")
    
    # 1. Sky + Poisson Noise
    img_with_sky = image_data + params['sky']
    img_poisson = np.random.poisson(np.maximum(img_with_sky, 0)).astype(np.float32)
    
    # 2. Gaussian Read Noise
    read_noise = np.random.normal(0, params['read_noise'], size=img_poisson.shape)
    img_noisy = img_poisson + read_noise
    
    # 3. Saturation
    img_authentic = np.clip(img_noisy, 0, params['full_well'])
    
    print(f"Loading catalog {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Create star density map
    h, w = image_data.shape
    star_map, _, _ = np.histogram2d(
        df['y'], df['x'], 
        bins=[h, w], 
        range=[[0, h], [0, w]]
    )
    
    # Save to FITS with extensions
    primary_hdu = fits.PrimaryHDU(img_authentic)
    primary_hdu.header['EXTNAME'] = 'AUTHENTIC_IMAGE'
    primary_hdu.header['ARCHTYP'] = archetype
    
    physics_hdu = fits.ImageHDU(image_data, name='CLEAN_PHYSICS')
    star_hdu = fits.ImageHDU(star_map.astype(np.float32), name='STAR_DENSITY')
    
    hdul = fits.HDUList([primary_hdu, physics_hdu, star_hdu])
    hdul.writeto(output_path, overwrite=True)
    print(f"✅ Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert Stage 1 Macro-Sparse mosaics to FITS for visualization.")
    parser.add_argument("--dir", default="data/stage1_mosaics", help="Directory containing mosaics")
    parser.add_argument("--idx", type=int, default=0, help="Index of the mosaic to convert")
    
    args = parser.parse_args()
    
    files = os.listdir(args.dir)
    img_file = [f for f in files if f.startswith(f"mosaic_{args.idx:02d}") and f.endswith(".npy")]
    cat_file = [f for f in files if f.startswith(f"mosaic_{args.idx:02d}") and f.endswith(".parquet")]
    
    if not img_file or not cat_file:
        print(f"❌ Error: Could not find mosaic files for index {args.idx} in {args.dir}")
        return
        
    img_path = os.path.join(args.dir, img_file[0])
    cat_path = os.path.join(args.dir, cat_file[0])
    output_path = img_path.replace(".npy", ".fits")
    
    convert_mosaic_to_fits(img_path, cat_path, output_path)

if __name__ == "__main__":
    main()
