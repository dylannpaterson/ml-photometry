import numpy as np
from astropy.coordinates import SkyCoord
from romanisim import wcs, ris_make_utils as ris
import romanisim.image as sim_image
from romanisim import persistence
import galsim
import os

def generate_bulge_exposure(output_path="bulge_empty.asdf"):
    """
    Generates a mission-fidelity empty Roman image pointing at the Galactic Bulge.
    Uses romanisim to populate the complex WCS and metadata tree.
    """
    print(f"🛰️ Initializing Roman metadata for Bulge pointing...")
    
    # 1. Setup Pointing (Galactic Bulge Center)
    ra, dec = 266.4167, -29.0078
    obs_time, band = '2026-10-31T00:00:00', 'F146'
    sca = 1 # Simulation for SCA 1
    
    # 2. Get official metadata template
    metadata = ris.set_metadata(date=obs_time, bandpass=band, sca=sca, ma_table_number=1002, usecrds=True)
    wcs.fill_in_parameters(metadata, SkyCoord(ra, dec, unit='deg', frame='icrs'), boresight=False, pa_aper=0.0)

    # 3. Minimal/Empty Catalog
    # Using an astropy Table ensures romanisim parses the source list correctly.
    from astropy.table import Table
    minimal_catalog = Table()
    minimal_catalog['ra'] = [ra]
    minimal_catalog['dec'] = [dec]
    minimal_catalog[band] = [0.0001] # Effectively zero flux

    # 4. RUN SIMULATION
    # level=2 produces an ImageModel (L2 data) which is what Castor/Pollux ingest.
    print(f"🚀 Simulating minimal exposure to generate valid headers...")
    res, _ = sim_image.simulate(
        metadata, 
        minimal_catalog, 
        rng=galsim.UniformDeviate(42), 
        persistence=persistence.Persistence(), 
        usecrds=True, 
        psftype='gauss', # Fast gaussian PSF
        level=2
    )

    # 5. Export to ASDF
    # Wrap in the high-level DataModel to expose the .save() method
    from roman_datamodels import datamodels
    model = datamodels.ImageModel(res)
    
    print(f"💾 Saving to {output_path}...")
    model.save(output_path)
    print(f"✅ Successfully created {output_path} with full Bulge headers.")

if __name__ == "__main__":
    generate_bulge_exposure()
