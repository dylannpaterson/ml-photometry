import argparse
import numpy as np
import asdf
import galsim
import time
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u
from romanisim import gaia, bandpass, catalog, log, wcs, persistence, parameters, ris_make_utils as ris
import romanisim.image as sim_image

def generate_pollux_prototype(output_path="pollux_prototype_l2.asdf"):
    """
    Generates a high-fidelity Roman L2 prototype image of a Galactic Bulge field.
    Uses the official gaia2romanisimcat conversion logic from the tutorial.
    """
    print(f"Starting tutorial-aligned Roman L2 simulation of the Galactic Bulge...")
    start_time = time.time()
    
    # 1. Gaia Catalog Query
    # RA: 17h 45m 40s -> 266.4167 deg, Dec: -29d 00m 28s -> -29.0078 deg
    ra_bulge, dec_bulge = 266.4167, -29.0078
    radius = 0.18 # Covers SCA footprint diagonal
    
    print(f"Querying Gaia around Bulge center (RA={ra_bulge}, Dec={dec_bulge}, Limit=20,000)...")
    query = f'''
    SELECT TOP 20000 ra, dec, pmra, pmdec, parallax, 
           phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag
    FROM gaiadr3.gaia_source 
    WHERE distance({ra_bulge}, {dec_bulge}, ra, dec) < {radius}
    AND phot_g_mean_mag IS NOT NULL AND phot_bp_mean_mag IS NOT NULL AND phot_rp_mean_mag IS NOT NULL
    ORDER BY phot_g_mean_mag ASC
    '''
    result = Gaia.launch_job_async(query).get_results()
    
    # 2. Tutorial-aligned Catalog Transformation
    obs_time, band = '2026-10-31T00:00:00', 'F146'
    print(f"Converting Gaia sources to Roman-compatible catalog...")
    # Tutorial uses set(bandpass.galsim2roman_bandpass.values()) to get all bands
    all_bands = set(bandpass.galsim2roman_bandpass.values())
    gaia_catalog = gaia.gaia2romanisimcat(result, Time(obs_time), fluxfields=all_bands)
    
    # Clean out bad sources (matching tutorial logic)
    bad = np.zeros(len(gaia_catalog), dtype='bool')
    for b in all_bands:
        if b in gaia_catalog.dtype.names:
            bad |= ~np.isfinite(gaia_catalog[b])
            if hasattr(gaia_catalog[b], 'mask'):
                bad |= gaia_catalog[b].mask
    
    gaia_catalog = gaia_catalog[~bad]
    gaia_catalog = gaia_catalog[np.isfinite(gaia_catalog['ra'])]
    
    # We still need the AB magnitudes for our ground truth
    # 3631.0 Jy is the AB zero point flux
    m_f146_ab_clean = -2.5 * np.log10(gaia_catalog[band] / 3631.0)

    # 3. Setup Metadata
    sca, seed = 1, 7
    rng = galsim.UniformDeviate(seed)
    persist = persistence.Persistence()
    
    # Using official GBTDS MA table 1002
    metadata = ris.set_metadata(date=obs_time, bandpass=band, sca=sca, ma_table_number=1002, usecrds=True)
    wcs.fill_in_parameters(metadata, SkyCoord(ra_bulge, dec_bulge, unit='deg', frame='icrs'), boresight=False, pa_aper=0.0)

    # 4. RUN SIMULATION
    print(f"Executing simulation ({len(gaia_catalog)} stars)...")
    image_obj, _ = sim_image.simulate(
        metadata, 
        gaia_catalog, 
        rng=rng, 
        persistence=persist, 
        usecrds=True, 
        psftype='stpsf', 
        level=2
    )

    # 5. DERIVE GOLDEN TRUTH FROM PHYSICS
    # official STScI zero point for F146
    zp_f146 = 27.61
    true_counts = 10**(-0.4 * (m_f146_ab_clean - zp_f146))
    log10_flux = np.log10(np.maximum(1e-15, true_counts))
    
    wcs_obj = image_obj.meta.wcs
    world_coords = SkyCoord(ra=gaia_catalog['ra'], dec=gaia_catalog['dec'], unit='deg')
    x_pix, y_pix = wcs_obj.world_to_pixel(world_coords)
    
    # Keep only stars within the detector footprint
    mask = (x_pix >= 0) & (x_pix < 4088) & (y_pix >= 0) & (y_pix < 4088)
    
    # 6. Save to ASDF
    print(f"Saving prototype to {output_path}...")
    tree = {
        'roman': image_obj,
        'ground_truth': {
            'ra': np.ascontiguousarray(gaia_catalog['ra'][mask], dtype=np.float64),
            'dec': np.ascontiguousarray(gaia_catalog['dec'][mask], dtype=np.float64),
            'x': np.ascontiguousarray(x_pix[mask], dtype=np.float64),
            'y': np.ascontiguousarray(y_pix[mask], dtype=np.float64),
            'ab_mag': np.ascontiguousarray(m_f146_ab_clean[mask], dtype=np.float64),
            'counts_per_sec': np.ascontiguousarray(true_counts[mask], dtype=np.float64),
            'log10_flux': np.ascontiguousarray(log10_flux[mask], dtype=np.float64),
            'num_stars': int(np.sum(mask))
        }
    }
    
    with asdf.AsdfFile(tree) as af:
        af.write_to(output_path)

    print(f"Done! Successfully generated prototype in {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    generate_pollux_prototype()
