import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Headless Backend
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import argparse
from scipy.ndimage import map_coordinates
from castor.data.transforms import AstroSpaceTransform
from castor.constants import GLOBAL_STRETCH_SCALE, SHAPE_SIZE, N_PCA_COMPONENTS

def visualize_mosaic_optimized(mosaic_idx=0, data_dir="data/bulge_stage0_full/mosaics"):
    img_path = os.path.join(data_dir, f"mosaic_{mosaic_idx:03d}_img.npy")
    bg_path = os.path.join(data_dir, f"mosaic_{mosaic_idx:03d}_bg.npy")
    cat_path = os.path.join(data_dir, f"mosaic_{mosaic_idx:03d}_cat.npy")
    meta_path = os.path.join(data_dir, f"mosaic_{mosaic_idx:03d}_meta.npy")
    lib_path = os.path.join(data_dir, f"mosaic_{mosaic_idx:03d}_psf_lib.npy")
    
    if not os.path.exists(img_path):
        print(f"Error: Mosaic not found at {img_path}")
        return

    # 1. Load Data
    star_signal = np.load(img_path)
    bg_image = np.load(bg_path) if os.path.exists(bg_path) else np.zeros_like(star_signal)
    structured_cat = np.load(cat_path)
    meta = np.load(meta_path)
    
    O = 4 # Assume oversampling factor
    half = SHAPE_SIZE // 2

    # Load PSF Library to get actual peak and area
    if os.path.exists(lib_path):
        psf_lib = np.load(lib_path)
        # Handle new format (single flattened PSF) or old format
        if psf_lib.size == SHAPE_SIZE * SHAPE_SIZE:
            mean_psf_1x = psf_lib.reshape(SHAPE_SIZE, SHAPE_SIZE)
        else:
            # Fallback for older multi-component format
            mean_psf_1x = psf_lib[-1].reshape(SHAPE_SIZE, SHAPE_SIZE)
        
        sum_p = np.sum(mean_psf_1x)
        eff_area = (sum_p**2) / (np.sum(mean_psf_1x**2) + 1e-9)
        print(f"📊 Using PSF from library: eff_area={eff_area:.2f}")
    else:
        eff_area = 12.0
        print(f"⚠️ PSF library not found, using default eff_area.")

    exp_time, zp, sky_mag = meta[0], meta[1], meta[2]
    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (0.11**2) * exp_time
    
    img_noisy = np.random.poisson(np.maximum(star_signal + sky_level, 0)).astype(np.float32)
    img_noisy += np.random.normal(0, 5.0, size=img_noisy.shape)
    
    chunk_median = np.median(img_noisy)
    transform = AstroSpaceTransform(stretch_scale=GLOBAL_STRETCH_SCALE)
    network_input = transform.image_to_network(img_noisy, chunk_median)
    bg_stretched = transform.target_bg_to_network(bg_image + sky_level - chunk_median)

    # --- 2. Rigorous SNR Extraction ---
    # We use the SNR from the catalog if available
    if 'snr' in structured_cat.dtype.names:
        snrs = structured_cat['snr']
    else:
        # Fallback to an approximation
        fluxes = structured_cat['flux']
        px, py = structured_cat['x'], structured_cat['y']
        x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
        actual_pixel_values = star_signal[y0, x0]
        local_background = np.maximum(0, actual_pixel_values - fluxes * 0.2) 
        noise_var = fluxes + eff_area * (sky_level + local_background + 25.0)
        snrs = fluxes / np.sqrt(noise_var)
    
    # Identify model targets
    px, py = structured_cat['x'], structured_cat['y']
    target_mask = snrs >= 5.0
    targets_x = px[target_mask]
    targets_y = py[target_mask]
    targets_mag = structured_cat['mag'][target_mask]
    
    snr2_mask = snrs > 2.0
    snr2_mag = structured_cat['mag'][snr2_mask]
    
    print(f"Total stars in culled catalog: {len(structured_cat):,}")
    print(f"Detectable targets (SNR >= 5):  {len(targets_x):,}")
    print(f"Visible sources (SNR > 2):      {len(snr2_mag):,}")

    # 3. Create Plot
    fig = plt.figure(figsize=(24, 12))
    
    ax0 = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)
    im0 = ax0.imshow(network_input, cmap='magma', vmin=np.percentile(network_input, 1), vmax=np.percentile(network_input, 99.9))
    ax0.set_title(f"Full Mosaic (1024x1024)\n{len(targets_x):,} Targets (SNR >= 5)")
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    ax_bg = plt.subplot2grid((2, 4), (0, 2), rowspan=1, colspan=1)
    im_bg = ax_bg.imshow(bg_stretched, cmap='magma', vmin=np.percentile(bg_stretched, 1), vmax=np.percentile(bg_stretched, 99.9))
    ax_bg.set_title("Truth Background Map\n(Unresolved Stars)")
    plt.colorbar(im_bg, ax=ax_bg, fraction=0.046, pad=0.04)
    
    zoom_size = 256
    zx, zy = 400, 400
    ax1 = plt.subplot2grid((2, 4), (1, 2))
    crop = network_input[zy:zy+zoom_size, zx:zx+zoom_size]
    im1 = ax1.imshow(crop, cmap='magma')
    ax1.set_title("Target Selection (SNR >= 5)\nZoomed 256x256")
    
    zoom_mask = (targets_x >= zx) & (targets_x < zx + zoom_size) & \
                (targets_y >= zy) & (targets_y < zy + zoom_size)
    ax1.scatter(targets_x[zoom_mask] - zx, targets_y[zoom_mask] - zy, s=30, edgecolors='cyan', facecolors='none', alpha=0.6)
    
    ax2 = plt.subplot2grid((2, 4), (0, 3), rowspan=2)
    hist_range = (12, 30)
    ax2.hist(structured_cat['mag'], bins=50, range=hist_range, color='gray', alpha=0.3, label='Culled Pop')
    ax2.hist(snr2_mag, bins=50, range=hist_range, color='orange', alpha=0.5, label='SNR > 2')
    ax2.hist(targets_mag, bins=50, range=hist_range, color='cyan', alpha=0.7, edgecolor='black', label='Targets (SNR >= 5)')
    
    ax2.set_title("Target Selection LF")
    ax2.set_xlabel("Magnitude")
    ax2.invert_xaxis()
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("optimized_mosaic_validation.png", dpi=150)
    print(f"✅ Optimized mosaic validation saved to optimized_mosaic_validation.png")

    # --- 4. Save to FITS ---
    from astropy.io import fits
    from astropy.wcs import WCS
    fits_path = "optimized_mosaic_validation.fits"
    
    w = WCS(naxis=2)
    w.wcs.crpix = [512.5, 512.5]
    w.wcs.crval = [266.417, -29.008] 
    scale = 0.11 / 3600.0
    w.wcs.cdelt = [-scale, scale]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    
    header = w.to_header()
    header['EXTNAME'] = 'NOISY_OBS'
    header['EXPTIME'] = exp_time
    header['ZP'] = zp
    header['SKYMAG'] = sky_mag
    header['MEDIAN'] = chunk_median

    primary_hdu = fits.PrimaryHDU(img_noisy, header=header)
    clean_hdu = fits.ImageHDU(star_signal, name='CLEAN_PHYSICS', header=header)
    stretched_hdu = fits.ImageHDU(network_input, name='NETWORK_INPUT', header=header)
    bg_hdu = fits.ImageHDU(bg_image, name='TRUTH_BG', header=header)
    
    h, w_size = star_signal.shape
    star_map, _, _ = np.histogram2d(
        py, px, 
        bins=[h, w_size], 
        range=[[0, h], [0, w_size]]
    )
    star_hdu = fits.ImageHDU(star_map.astype(np.float32), name='STAR_DENSITY', header=header)
    
    hdul = fits.HDUList([primary_hdu, clean_hdu, stretched_hdu, bg_hdu, star_hdu])
    hdul.writeto(fits_path, overwrite=True)
    print(f"✅ Mosaic saved to FITS with WCS: {fits_path}")
    
    hdul = fits.HDUList([primary_hdu, clean_hdu, stretched_hdu, star_hdu])
    hdul.writeto(fits_path, overwrite=True)
    print(f"✅ Mosaic saved to FITS with WCS: {fits_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize optimized mosaic")
    parser.add_argument("--mosaic_idx", type=int, default=0, help="Index of the mosaic to visualize")
    parser.add_argument("--data_dir", type=str, default="data/bulge_stage0_full/mosaics", help="Directory containing the mosaics")
    args = parser.parse_args()
    
    visualize_mosaic_optimized(mosaic_idx=args.mosaic_idx, data_dir=args.data_dir)
