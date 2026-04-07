import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# Set STPSF_DATA if not in environment
if 'STPSF_DATA' not in os.environ:
    os.environ['STPSF_DATA'] = '/home/dylan/data/stpsf-data'

import stpsf

def generate_realistic_prior():
    print("🛰️ Generating Realistic Roman PSF Prior using stpsf...")
    
    try:
        # 1. Initialize the Wide Field Instrument
        wfi = stpsf.roman.WFI()

        # 2. Set parameters (F146 is the standard Bulge survey filter)
        wfi.filter = 'F146' 
        wfi.detector = 'WFI01'
        wfi.detector_position = (2048, 2048) # Center of detector

        # 3. Add realistic jitter (~7 mas)
        wfi.options['jitter'] = 'gaussian'
        wfi.options['jitter_sigma'] = 0.007 

        # 4. Calculate the PSF
        # fov_pixels=127 to capture full diffraction spikes
        # oversample=1 to get it at detector pixel resolution
        print("🔍 Calculating 127x127 PSF (this may take a few seconds)...")
        psf_hdulist = wfi.calc_psf(fov_pixels=127, oversample=1)

        # 5. Get the raw numpy array (HDU 0 is detector sampled)
        realistic_psf = psf_hdulist[0].data
        
        # Ensure it's exactly 127x127
        S = 127
        if realistic_psf.shape != (S, S):
            print(f"⚠️ PSF shape {realistic_psf.shape} != (127, 127). Resizing...")
            curr_h, curr_w = realistic_psf.shape
            pad_h = (S - curr_h) // 2
            pad_w = (S - curr_w) // 2
            if pad_h >= 0 and pad_w >= 0:
                realistic_psf = np.pad(realistic_psf, ((pad_h, S-curr_h-pad_h), (pad_w, S-curr_w-pad_w)))
            else:
                h_start = (curr_h - S) // 2
                w_start = (curr_w - S) // 2
                realistic_psf = realistic_psf[h_start:h_start+S, w_start:w_start+S]

        realistic_psf = realistic_psf / (realistic_psf.sum() + 1e-9)
        
        # Save it
        torch.save(torch.from_numpy(realistic_psf.astype(np.float32)), "roman_psf_prior.pt")
        
        plt.figure(figsize=(8, 8))
        # Use log scale for visualization to see the spikes
        plt.imshow(np.log10(realistic_psf + 1e-6), cmap='inferno')
        plt.title("Realistic Roman PSF Prior (stpsf F146, 127x127)")
        plt.colorbar(label="log10(Intensity)")
        plt.savefig("roman_psf_prior.png")
        print("✅ Saved 127x127 realistic prior to roman_psf_prior.pt and roman_psf_prior.png")
        
    except Exception as e:
        print(f"❌ Failed to generate realistic PSF: {e}")
        import traceback
        traceback.print_exc()
        print("Falling back to a high-quality analytical model.")
        
        S = 127
        half = S // 2
        y, x = np.meshgrid(np.arange(S) - half, np.arange(S) - half, indexing='ij')
        r = np.sqrt(x**2 + y**2)
        
        psf = np.exp(-(r**2) / (2 * 1.5**2))
        
        angles = [0, np.pi/3, 2*np.pi/3]
        for angle in angles:
            dist_to_line = np.abs(x * np.sin(angle) - y * np.cos(angle))
            # Wider spikes for the larger box
            psf += np.exp(-dist_to_line / 0.4) * 0.05 * np.exp(-r**2 / (2 * 40**2))
            
        psf /= psf.sum()
        torch.save(torch.from_numpy(psf.astype(np.float32)), "roman_psf_prior.pt")
        print("✅ Saved analytical high-quality 127x127 prior to roman_psf_prior.pt")

if __name__ == "__main__":
    generate_realistic_prior()
