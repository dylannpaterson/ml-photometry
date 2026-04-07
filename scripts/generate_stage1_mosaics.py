import galsim
import numpy as np
import pandas as pd
import multiprocessing as mp
import os
import time
from scipy.signal import fftconvolve

# --- Configuration ---
NUM_MOSAICS = 500              # Up from 20 to 500 unique parameter states
IMAGE_SIZE = 1024              # Down from 4088 (1/16th the area)
MIN_STARS = 12500              # Scaled down by 16 (matches 200k on 4088)
MAX_STARS = 218750             # Scaled down by 16 (matches 3.5m on 4088)
OUTPUT_DIR = "data/stage1_mosaics"

# Native scales and detector limits
ARCHETYPE_PARAMS = {
    'roman': {
        'scale': 0.11, 'jitter': 0.015, 'charge_diffusion': 0.035,
        'full_well': 100000, 'sky_mag_arcsec2': 22.0,
        'zp': 26.5, # Realistic Roman WFI wide-band ZP
        'exp_time_range': (20.0, 100.0) # Simulate snapshots vs deep stares
    },
    'hubble': {
        'scale': 0.128, 'jitter': 0.008, 'charge_diffusion': 0.020,
        'full_well': 80000, 'sky_mag_arcsec2': 21.0,
        'zp': 25.5,
        'exp_time_range': (100.0, 1000.0)
    },
    'ideal_space': {
        'scale': 0.10, 'jitter': 0.002, 'charge_diffusion': 0.0,
        'full_well': 1000000, 'sky_mag_arcsec2': 24.0,
        'zp': 27.0,
        'exp_time_range': (50.0, 500.0)
    },
    'ground': {
        'scale': 0.34, 'jitter': 0.050, 'charge_diffusion': 0.0,
        'full_well': 200000, 'sky_mag_arcsec2': 16.0, # Bright ground sky
        'zp': 25.0, # e.g., VISTA 4m telescope
        'exp_time_range': (5.0, 30.0) # Short IR ground exposures to avoid sky saturation
    }
}

# --- 1. The Dynamic Bulge LF Sampler ---
def sample_bulge_magnitudes(n_stars, rc_mag, rc_sigma, rc_fraction, m_min=10.0, m_max=26.0):
    """Samples apparent magnitudes (m). Smaller m = brighter star."""
    n_rc = int(n_stars * rc_fraction)
    n_bg = n_stars - n_rc

    # 1. Background (Power law in linear space becomes linear in magnitude space)
    # Most stars are faint (high magnitude)
    u = np.random.uniform(0, 1, n_bg)
    # Background slope randomization
    m_bg = np.interp(u, [0.0, 0.95, 1.0], [m_max, 18.0, m_min])

    # 2. Red Clump (Gaussian in magnitude space)
    m_rc = np.random.normal(loc=rc_mag, scale=rc_sigma, size=n_rc)

    # 3. Combine and clip
    m_all = np.concatenate([m_bg, m_rc])
    m_all = np.clip(m_all, m_min, m_max)

    return m_all # Return raw magnitudes

# --- 2. The Multi-Telescope Optical Archetypes ---
def generate_archetype_psf(archetype, lam, params):
    """Generates a realistic PSF including optics, jitter, and detector diffusion."""
    pixel_scale = params['scale']
    jitter_sigma = params['jitter']
    diffusion_sigma = params.get('charge_diffusion', 0.0)
    
    if archetype == 'roman':
        # Roman: 6 thick struts -> 12 spikes
        opt_psf = galsim.OpticalPSF(
            lam=lam, diam=2.4, obscuration=0.32, 
            nstruts=6, strut_angle=15*galsim.degrees, strut_thick=0.032,
            defocus=np.random.normal(0, 0.05), astig1=np.random.normal(0, 0.03), oversampling=1.5
        )
    elif archetype == 'hubble':
        # Hubble: 4 thick struts -> 4 spikes
        opt_psf = galsim.OpticalPSF(
            lam=lam, diam=2.4, obscuration=0.33, 
            nstruts=4, strut_angle=np.random.uniform(0, 90)*galsim.degrees, strut_thick=0.032,
            defocus=np.random.normal(0, 0.04), coma1=np.random.normal(0, 0.02), oversampling=1.5
        )
    elif archetype == 'ideal_space':
        opt_psf = galsim.OpticalPSF(lam=lam, diam=2.4, obscuration=0.0, nstruts=0, oversampling=1.5)
    elif archetype == 'ground':
        seeing_fwhm = np.random.uniform(0.6, 1.5)
        opt_psf = galsim.Moffat(beta=3.5, fwhm=seeing_fwhm)
    else:
        raise ValueError("Unknown archetype")

    components = [opt_psf]
    
    # Add spacecraft pointing jitter
    if jitter_sigma > 0:
        components.append(galsim.Gaussian(sigma=jitter_sigma))
        
    # Add detector charge diffusion/IPC smear
    if diffusion_sigma > 0:
        components.append(galsim.Gaussian(sigma=diffusion_sigma))
        
    # Integrate over the physical pixel area
    components.append(galsim.Pixel(pixel_scale))
    
    return galsim.Convolve(components)

# --- 3. The Mosaic Rendering Engine ---
def render_single_mosaic(idx):
    np.random.seed(42 + idx) 
    
    # 1. Setup Instrument Params
    archetypes = ['roman', 'hubble', 'ideal_space', 'ground']
    archetype = archetypes[idx % 4] 
    params = ARCHETYPE_PARAMS[archetype]
    pixel_scale = params['scale']
    wavelength = np.random.uniform(500, 2000) 
    
    log_min, log_max = np.log10(MIN_STARS), np.log10(MAX_STARS)
    n_detectable = int(10 ** np.random.uniform(log_min, log_max))
    
    # NEW: Draw the specific exposure parameters for this mosaic
    exp_time = np.random.uniform(*params['exp_time_range'])
    zp = params['zp']
    
    # NEW: The Red Clump is now defined by its Apparent Magnitude (e.g., ~14.5 to 16.5 depending on dust)
    rc_mag = np.random.uniform(14.5, 16.5)
    rc_sigma = np.random.uniform(0.2, 0.5)
    rc_fraction = np.random.uniform(0.05, 0.20)
    
    print(f"[{idx+1}/{NUM_MOSAICS}] {archetype.upper()} | t={exp_time:.1f}s | RC_Mag={rc_mag:.1f} | {n_detectable:,} stars")
    start_time = time.time()

    # 2. Initialize the clean, empty image
    image = galsim.ImageF(IMAGE_SIZE, IMAGE_SIZE, scale=pixel_scale)
    image.setOrigin(0, 0)
    
    # 3. Generate PSFs
    # Lower folding_threshold to capture wide wings
    gs_params = galsim.GSParams(folding_threshold=1e-3, maximum_fft_size=16384)
    base_psf = generate_archetype_psf(archetype, wavelength, params).withGSParams(gs_params)
    
    # 4. Pre-render Kernel (Large 127x127 to avoid box artifacts)
    kernel_size = 127
    psf_kernel_image = galsim.ImageF(kernel_size, kernel_size, scale=pixel_scale)
    base_psf.drawImage(image=psf_kernel_image, method='no_pixel')
    psf_kernel = psf_kernel_image.array
    
    # 5. Extract Canonical 9x9 Shape for catalog
    shape_stamp = galsim.ImageF(9, 9, scale=pixel_scale)
    base_psf.drawImage(image=shape_stamp, method='no_pixel')
    shape_vector = shape_stamp.array.flatten().tolist()
    
    # 6. Generate Star Catalog (Detectable)
    # 1. Sample purely in Astrophysics (Magnitudes)
    mags = sample_bulge_magnitudes(n_detectable, rc_mag, rc_sigma, rc_fraction)
    
    # 2. Convert to Instrument Physics (Flux Counts)
    fluxes = exp_time * (10 ** (-0.4 * (mags - zp)))
    
    # Sort brightest to faintest (highest flux to lowest flux)
    sort_idx = np.argsort(fluxes)[::-1]
    fluxes = fluxes[sort_idx]
    mags = mags[sort_idx]
    
    x_coords = np.random.uniform(0, IMAGE_SIZE - 1, n_detectable)
    y_coords = np.random.uniform(0, IMAGE_SIZE - 1, n_detectable)
    
    catalog_data = {
        'x': x_coords, 'y': y_coords, 'flux': fluxes, 
        'mag': mags, 'shape': [shape_vector] * n_detectable,
        'exp_time': [exp_time] * n_detectable,
        'zp': [zp] * n_detectable,
        'sky_mag': [params['sky_mag_arcsec2']] * n_detectable
    }

    # 7. Hybrid Rendering
    monster_cutoff = int(n_detectable * 0.02)
    
    # 7a. Render Monsters (Preserving sub-pixel phase)
    for i in range(monster_cutoff):
        star = base_psf.withFlux(fluxes[i])
        pos = galsim.PositionD(x_coords[i], y_coords[i])
        bounds = galsim.BoundsI(int(pos.x)-256, int(pos.x)+255, int(pos.y)-256, int(pos.y)+255)
        overlap = bounds & image.bounds
        if overlap.isDefined():
            star.drawImage(image=image[overlap], center=pos, method='no_pixel', add_to_image=True)
    
    # 7b. Render Crowd (Fast FFT Convolution)
    crowd_map = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    cx, cy, cf = x_coords[monster_cutoff:], y_coords[monster_cutoff:], fluxes[monster_cutoff:]
    x0, y0 = np.floor(cx).astype(int), np.floor(cy).astype(int)
    dx, dy = cx - x0, cy - y0
    mask = (x0 >= 0) & (x0 < IMAGE_SIZE - 1) & (y0 >= 0) & (y0 < IMAGE_SIZE - 1)
    x0, y0, dx, dy, cf = x0[mask], y0[mask], dx[mask], dy[mask], cf[mask]
    np.add.at(crowd_map, (y0, x0), cf * (1-dx) * (1-dy))
    np.add.at(crowd_map, (y0, x0+1), cf * dx * (1-dy))
    np.add.at(crowd_map, (y0+1, x0), cf * (1-dx) * dy)
    np.add.at(crowd_map, (y0+1, x0+1), cf * dx * dy)
    
    # 7c. The "Stellar Sea" (Unresolved mottled background)
    n_unresolved = 2000000
    # NEW: Sample unresolved sea magnitudes and convert to fluxes
    unresolved_mags = sample_bulge_magnitudes(n_unresolved, rc_mag, rc_sigma, rc_fraction, m_min=26.0, m_max=32.0)
    unresolved_fluxes = exp_time * (10 ** (-0.4 * (unresolved_mags - zp)))
    
    ux = np.random.uniform(0, IMAGE_SIZE-1, n_unresolved)
    uy = np.random.uniform(0, IMAGE_SIZE-1, n_unresolved)
    ux0, uy0 = np.floor(ux).astype(int), np.floor(uy).astype(int)
    umask = (ux0 >= 0) & (ux0 < IMAGE_SIZE-1) & (uy0 >= 0) & (uy0 < IMAGE_SIZE-1)
    np.add.at(crowd_map, (uy0[umask], ux0[umask]), unresolved_fluxes[umask])

    # Fast global FFT convolution
    image.array[:] += fftconvolve(crowd_map, psf_kernel, mode='same')

    # 8. Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, f"mosaic_{idx:02d}_{archetype}.npy"), image.array)
    pd.DataFrame(catalog_data).to_parquet(os.path.join(OUTPUT_DIR, f"mosaic_{idx:02d}_{archetype}.parquet"), index=False)
    
    print(f"[{idx+1}/{NUM_MOSAICS}] Completed in {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    print(f"🚀 Starting Stage 1 Mosaic Generation...")
    num_workers = max(1, mp.cpu_count() - 2) 
    with mp.Pool(num_workers) as pool:
        pool.map(render_single_mosaic, range(NUM_MOSAICS))
    print(f"✅ All {NUM_MOSAICS} Macro-Sparse mosaics successfully generated.")
