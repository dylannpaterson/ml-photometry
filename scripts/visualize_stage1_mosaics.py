import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
import os
import argparse

# Instrument Params for Authentic Pass
ARCHETYPE_PARAMS = {
    'roman':       {'scale': 0.11, 'read_noise': 5.0,  'full_well': 100000},
    'hubble':      {'scale': 0.128, 'read_noise': 3.0,  'full_well': 80000},
    'ideal_space': {'scale': 0.10, 'read_noise': 1.0,  'full_well': 1000000},
    'ground':      {'scale': 0.34, 'read_noise': 15.0, 'full_well': 200000}
}

def get_bogus_wcs(img_shape, pixel_scale, idx):
    """Generates a realistic WCS header centered in the Galactic Bulge."""
    rng = np.random.RandomState(42 + idx) # Deterministic per mosaic
    h, w = img_shape
    
    # Random center in the Galactic Bulge (RA ~ 266, Dec ~ -29)
    ra_cent = rng.uniform(264.0, 268.0)
    dec_cent = rng.uniform(-32.0, -26.0)
    
    # Random rotation (roll angle)
    theta = rng.uniform(0, 2*np.pi)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    w = WCS(naxis=2)
    w.wcs.crpix = [img_shape[1]/2, img_shape[0]/2]
    w.wcs.crval = [ra_cent, dec_cent]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    
    # Scale in degrees/pixel
    s = pixel_scale / 3600.0
    
    # CD matrix (incorporates scale and rotation)
    # CD1_1 CD1_2
    # CD2_1 CD2_2
    w.wcs.cd = np.array([
        [-s * cos_theta,  s * sin_theta],
        [ s * sin_theta,  s * cos_theta]
    ])
    
    return w

def convert_mosaic_to_fits(npy_path, parquet_path, output_path, idx):
    # Infer archetype
    for arch in ARCHETYPE_PARAMS.keys():
        if arch in npy_path:
            archetype = arch
            break
    else:
        archetype = 'roman'
        
    params = ARCHETYPE_PARAMS[archetype]
    
    print(f"Loading {npy_path} ({archetype.upper()})...")
    image_data = np.load(npy_path)
    
    print(f"Loading catalog {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # --- AUTHENTIC DETECTOR PASS ---
    if 'exp_time' in df.columns:
        exp_time = df['exp_time'].iloc[0]
        zp = df['zp'].iloc[0]
        sky_mag = df['sky_mag'].iloc[0]
        scale = params['scale']
        sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (scale**2) * exp_time
        print(f"Applying Metadata-driven Detector Layer (Sky={sky_level:.2f}, exp={exp_time:.1f}s, RN={params['read_noise']})...")
    else:
        sky_level = 30.0 # Legacy fallback
        print(f"Applying Legacy Detector Layer (Sky={sky_level}, RN={params['read_noise']})...")
    
    # 1. Sky + Poisson Noise
    img_with_sky = image_data + sky_level
    img_poisson = np.random.poisson(np.maximum(img_with_sky, 0)).astype(np.float32)
    
    # 2. Gaussian Read Noise
    read_noise = np.random.normal(0, params['read_noise'], size=img_poisson.shape)
    img_noisy = img_poisson + read_noise
    
    # 3. Saturation
    img_authentic = np.clip(img_noisy, 0, params['full_well'])
    
    # Create star density map
    h, w = image_data.shape
    star_map, _, _ = np.histogram2d(
        df['y'], df['x'], 
        bins=[h, w], 
        range=[[0, h], [0, w]]
    )
    
    # Generate WCS
    wcs = get_bogus_wcs(image_data.shape, params['scale'], idx)
    header = wcs.to_header()
    
    # Save to FITS with extensions
    primary_hdu = fits.PrimaryHDU(img_authentic, header=header)
    primary_hdu.header['EXTNAME'] = 'AUTHENTIC_IMAGE'
    primary_hdu.header['ARCHTYP'] = archetype
    if 'exp_time' in df.columns:
        primary_hdu.header['EXPTIME'] = exp_time
        primary_hdu.header['ZP'] = zp
        primary_hdu.header['SKYMAG'] = sky_mag
    
    physics_hdu = fits.ImageHDU(image_data, header=header, name='CLEAN_PHYSICS')
    star_hdu = fits.ImageHDU(star_map.astype(np.float32), header=header, name='STAR_DENSITY')
    
    hdul = fits.HDUList([primary_hdu, physics_hdu, star_hdu])
    hdul.writeto(output_path, overwrite=True)
    print(f"✅ Saved to {output_path} with realistic WCS")

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
    
    convert_mosaic_to_fits(img_path, cat_path, output_path, args.idx)

if __name__ == "__main__":
    main()
