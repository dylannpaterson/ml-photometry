import galsim
import numpy as np
import pandas as pd
import multiprocessing as mp
import os
import time
from scipy.signal import fftconvolve

# --- Configuration ---
NUM_MOSAICS = 20
MIN_STARS = 200000      
MAX_STARS = 3500000     
IMAGE_SIZE = 4088
OUTPUT_DIR = "data/stage1_mosaics"

# Native scales and detector limits
ARCHETYPE_PARAMS = {
    'roman': {
        'scale': 0.11, 
        'jitter': 0.015,
        'charge_diffusion': 0.035, # Emulate NIR detector smear
        'full_well': 100000, 
        'sky': 30.0
    },
    'hubble': {
        'scale': 0.128, 
        'jitter': 0.008, 
        'charge_diffusion': 0.020,
        'full_well': 80000,  
        'sky': 20.0
    },
    'ideal_space': {
        'scale': 0.10, 
        'jitter': 0.002, 
        'charge_diffusion': 0.0,
        'full_well': 1000000, 
        'sky': 5.0
    },
    'ground': {
        'scale': 0.34, 
        'jitter': 0.050, 
        'charge_diffusion': 0.0,
        'full_well': 200000, 
        'sky': 150.0 
    }
}

# --- 1. The Fast Bulge LF Sampler ---
def sample_bulge_fluxes(n_stars, f_min=10, f_max=10**6.5):
    """Realistic LF sampling with faint tail extension."""
    u = np.random.uniform(0, 1, n_stars)
    fluxes = 10 ** (np.interp(u, [0.0, 0.90, 1.0], [np.log10(f_min), 4.0, np.log10(f_max)])) 
    return np.sort(fluxes)[::-1] 

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
            nstruts=6, strut_angle=15*galsim.degrees, strut_width=0.032,
            defocus=np.random.normal(0, 0.05), astig1=np.random.normal(0, 0.03), oversampling=1.5
        )
    elif archetype == 'hubble':
        # Hubble: 4 thick struts -> 4 spikes
        opt_psf = galsim.OpticalPSF(
            lam=lam, diam=2.4, obscuration=0.33, 
            nstruts=4, strut_angle=np.random.uniform(0, 90)*galsim.degrees, strut_width=0.032,
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
    
    print(f"[{idx+1}/{NUM_MOSAICS}] {archetype.upper()} | {n_detectable:,} stars + Stellar Sea...")
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
    fluxes = sample_bulge_fluxes(n_detectable)
    x_coords = np.random.uniform(0, IMAGE_SIZE - 1, n_detectable)
    y_coords = np.random.uniform(0, IMAGE_SIZE - 1, n_detectable)
    
    catalog_data = {'x': x_coords, 'y': y_coords, 'flux': fluxes, 'shape': [shape_vector] * n_detectable}

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
    ux = np.random.uniform(0, IMAGE_SIZE-1, n_unresolved)
    uy = np.random.uniform(0, IMAGE_SIZE-1, n_unresolved)
    uf = sample_bulge_fluxes(n_unresolved, f_min=0.1, f_max=5.0)
    ux0, uy0 = np.floor(ux).astype(int), np.floor(uy).astype(int)
    umask = (ux0 >= 0) & (ux0 < IMAGE_SIZE-1) & (uy0 >= 0) & (uy0 < IMAGE_SIZE-1)
    np.add.at(crowd_map, (uy0[umask], ux0[umask]), uf[umask])

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
    print("✅ All 20 Macro-Sparse mosaics successfully generated.")
