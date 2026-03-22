import galsim
import numpy as np
import pandas as pd
import multiprocessing as mp
import os
import time

# --- Configuration ---
NUM_MOSAICS = 20
MIN_STARS = 200000      # Simulates high-latitude or high-extinction regions
MAX_STARS = 3500000     # Simulates deep bulge core windows
IMAGE_SIZE = 4088
PIXEL_SCALE = 0.11  # Roman WFI scale (arcsec/pixel)
OUTPUT_DIR = "data/stage1_mosaics"

# --- 1. The Fast Bulge LF Sampler ---
def sample_bulge_fluxes(n_stars):
    """
    Approximates a Bulge Luminosity Function. 
    """
    # Generates a power law distribution favoring faint sources
    u = np.random.uniform(0, 1, n_stars)
    # Stretch roughly corresponds to Roman magnitude limits (approx 10^2 to 10^6 photons)
    fluxes = 10 ** (np.interp(u, [0.0, 0.95, 1.0], [2.0, 4.0, 6.5])) 
    return np.sort(fluxes)[::-1] # Sort bright to faint for canonical slotting

# --- 2. The Multi-Telescope Optical Archetypes ---
def generate_archetype_psf(archetype, lam):
    """Generates a randomized PSF based on the selected archetype."""
    if archetype == 'roman':
        return galsim.OpticalPSF(
            lam=lam, diam=2.4, obscuration=0.32, nstruts=6, strut_angle=np.random.uniform(0, 60)*galsim.degrees,
            defocus=np.random.normal(0, 0.05), astig1=np.random.normal(0, 0.03), oversampling=1.5
        )
    elif archetype == 'hubble':
        return galsim.OpticalPSF(
            lam=lam, diam=2.4, obscuration=0.33, nstruts=4, strut_angle=np.random.uniform(0, 90)*galsim.degrees,
            defocus=np.random.normal(0, 0.04), coma1=np.random.normal(0, 0.02), oversampling=1.5
        )
    elif archetype == 'ideal_space':
        # Unobscured, no struts, just pure Airy disk with slight aberrations
        return galsim.OpticalPSF(
            lam=lam, diam=2.4, obscuration=0.0, nstruts=0,
            astig2=np.random.normal(0, 0.06), defocus=np.random.normal(0, 0.02), oversampling=1.5
        )
    elif archetype == 'ground':
        # Seeing-limited Moffat profile
        seeing_fwhm = np.random.uniform(0.6, 1.5) # Typical ground seeing in arcsec
        return galsim.Moffat(beta=3.5, fwhm=seeing_fwhm)
    else:
        raise ValueError("Unknown archetype")

# --- 3. The Mosaic Rendering Engine ---
def render_single_mosaic(idx):
    np.random.seed(42 + idx) # Ensure reproducible but unique mosaics
    
    # 1. Determine Mosaic Properties
    archetypes = ['roman', 'hubble', 'ideal_space', 'ground']
    archetype = archetypes[idx % 4] # Evenly distribute the 20 mosaics
    wavelength = np.random.uniform(500, 2000) # Continuous domain randomization (nm)
    
    # NEW: Log-uniform density sampling
    log_min = np.log10(MIN_STARS)
    log_max = np.log10(MAX_STARS)
    n_stars = int(10 ** np.random.uniform(log_min, log_max))
    
    print(f"[{idx+1}/{NUM_MOSAICS}] {archetype.upper()} @ {wavelength:.0f}nm | {n_stars:,} stars")
    start_time = time.time()

    # 2. Initialize the clean, empty image
    image = galsim.ImageF(IMAGE_SIZE, IMAGE_SIZE, scale=PIXEL_SCALE)
    image.setOrigin(0, 0)
    
    # 3. Generate the Base PSF and Stars
    # Use GSParm to speed up rendering by being slightly less aggressive on folding threshold
    gs_params = galsim.GSParams(folding_threshold=1e-2, maximum_fft_size=16384)
    base_psf = generate_archetype_psf(archetype, wavelength).withGSParams(gs_params)
    
    fluxes = sample_bulge_fluxes(n_stars)
    x_coords = np.random.uniform(0, IMAGE_SIZE - 1, n_stars)
    y_coords = np.random.uniform(0, IMAGE_SIZE - 1, n_stars)
    
    # 4. Extract the canonical 9x9 shape prior for the catalog
    shape_stamp = galsim.ImageF(9, 9, scale=PIXEL_SCALE)
    base_psf.drawImage(image=shape_stamp, method='real_space') # Force real-space
    shape_vector = shape_stamp.array.flatten().tolist()
    
    catalog_data = {
        'x': x_coords, 'y': y_coords, 'flux': fluxes,
        'shape': [shape_vector] * n_stars 
    }

    # 5. The Two-Tier Drawing Loop
    # Using real-space drawing (real_space) is MUCH more memory efficient for bright stars
    # and bypasses the 20GB FFT allocation issue.
    bright_cutoff = int(n_stars * 0.05)
    
    for i in range(n_stars):
        star = base_psf.withFlux(fluxes[i])
        pos = galsim.PositionD(x_coords[i], y_coords[i])
        
        if i < bright_cutoff:
            # Use real-space drawing for bright stars to handle broad wings without FFT blowup
            star.drawImage(image=image, center=pos, method='real_space', add_to_image=True)
        else:
            # For faint stars, stay extremely local (16x16) for speed
            bounds = galsim.BoundsI(
                int(pos.x) - 8, int(pos.x) + 7, 
                int(pos.y) - 8, int(pos.y) + 7
            )
            overlap = bounds & image.bounds
            if overlap.isDefined():
                stamp = image[overlap]
                star.drawImage(image=stamp, center=pos, method='real_space', add_to_image=True)

    # 6. Save Macro-Sparse Outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save the 66MB Clean Physics Image
    img_path = os.path.join(OUTPUT_DIR, f"mosaic_{idx:02d}_{archetype}.npy")
    np.save(img_path, image.array)
    
    # Save the Parquet Catalog
    cat_path = os.path.join(OUTPUT_DIR, f"mosaic_{idx:02d}_{archetype}.parquet")
    df = pd.DataFrame(catalog_data)
    df.to_parquet(cat_path, engine='pyarrow', index=False)
    
    print(f"[{idx+1}/{NUM_MOSAICS}] Completed in {time.time() - start_time:.1f}s")

# --- Execution ---
if __name__ == "__main__":
    print(f"🚀 Starting Stage 1 Mosaic Generation...")
    
    # Leave 2 cores free to maintain system responsiveness
    num_workers = max(1, mp.cpu_count() - 2) 
    print(f"Using {num_workers} parallel workers...")
    
    with mp.Pool(num_workers) as pool:
        pool.map(render_single_mosaic, range(NUM_MOSAICS))
        
    print("✅ All 20 Macro-Sparse mosaics successfully generated.")
