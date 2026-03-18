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

def convert_gaia_to_roman_f146(g, bp, rp):
    """Official Roman-STScI-000825 transformation."""
    color = bp - rp
    c0, c1, c2, c3, c4 = -0.1463, -0.5841, -0.1048, 0.0201, -0.0015
    return g + c0 + c1*color + c2*(color**2) + c3*(color**3) + c4*(color**4)

def generate_pollux_prototype(output_path="pollux_prototype_l2.asdf"):
    """
    Generates a high-fidelity Roman L2 prototype image of 47 Tuc.
    Calibrated using official STScI zero points.
    """
    print(f"Starting high-fidelity Roman L2 simulation of 47 Tuc...")
    start_time = time.time()
    
    # 1. Gaia Catalog Query
    ra_47tuc, dec_47tuc = 6.0236, -72.0813
    radius = 0.15 
    query = f'''
    SELECT ra, dec, pmra, pmdec, parallax, 
           phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag
    FROM gaiadr3.gaia_source 
    WHERE distance({ra_47tuc}, {dec_47tuc}, ra, dec) < {radius}
    AND phot_g_mean_mag IS NOT NULL AND phot_bp_mean_mag IS NOT NULL AND phot_rp_mean_mag IS NOT NULL
    '''
    result = Gaia.launch_job_async(query).get_results()
    
    # 2. Photometry Transformation (Input to Simulator)
    obs_time, band = '2026-10-31T00:00:00', 'F146'
    m_f146_ab = convert_gaia_to_roman_f146(result['phot_g_mean_mag'], result['phot_bp_mean_mag'], result['phot_rp_mean_mag'])
    
    # romanisim expects flux in Janskys
    gaia_catalog = gaia.gaia2romanisimcat(result, Time(obs_time), fluxfields=[band])
    gaia_catalog[band] = 3631.0 * 10**(-0.4 * m_f146_ab)
    
    # Clean bad sources
    bad = ~np.isfinite(m_f146_ab) | ~np.isfinite(gaia_catalog['ra'])
    full_catalog = gaia_catalog[~bad]
    m_f146_ab_clean = m_f146_ab[~bad]

    # 3. Setup Metadata
    sca, seed = 1, 7
    rng = galsim.UniformDeviate(seed)
    persist = persistence.Persistence()
    metadata = ris.set_metadata(date=obs_time, bandpass=band, sca=sca, ma_table_number=1018, usecrds=True)
    wcs.fill_in_parameters(metadata, SkyCoord(ra_47tuc, dec_47tuc, unit='deg', frame='icrs'), boresight=False, pa_aper=0.0)

    # 4. RUN SIMULATION
    print("Executing simulation (STPSF + CRDS + Persistence)...")
    image_obj, _ = sim_image.simulate(
        metadata, 
        full_catalog, 
        rng=rng, 
        persistence=persist, 
        usecrds=True, 
        psftype='stpsf', 
        level=2
    )

    # 5. DERIVE GOLDEN TRUTH FROM PHYSICS
    print("Deriving machine flux from official zero points...")
    # m_ab = -2.5 * log10(counts/s) + 27.61
    zp_f146 = 27.61
    true_counts = 10**(-0.4 * (m_f146_ab_clean - zp_f146))
    log10_flux = np.log10(np.maximum(1e-15, true_counts))
    
    # Filter for stars that are actually on the detector
    wcs_obj = image_obj.meta.wcs
    world_coords = SkyCoord(ra=full_catalog['ra'], dec=full_catalog['dec'], unit='deg')
    x_pix, y_pix = wcs_obj.world_to_pixel(world_coords)
    
    # Keep only stars within the 4088x4088 SCA footprint
    mask = (x_pix >= 0) & (x_pix < 4088) & (y_pix >= 0) & (y_pix < 4088)
    
    # 6. Save to ASDF
    print(f"Saving high-fidelity image and filtered truth to {output_path}...")
    tree = {
        'roman': image_obj,
        'ground_truth': {
            'ra': np.array(full_catalog['ra'])[mask],
            'dec': np.array(full_catalog['dec'])[mask],
            'x': x_pix[mask],
            'y': y_pix[mask],
            'ab_mag': m_f146_ab_clean[mask],
            'counts_per_sec': true_counts[mask],
            'log10_flux': log10_flux[mask],
            'num_stars': np.sum(mask)
        }
    }
    
    with asdf.AsdfFile(tree) as af:
        af.write_to(output_path)

    print(f"Done! Successfully generated {output_path} in {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    generate_pollux_prototype()
