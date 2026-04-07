import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
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
        # oversample=4 to capture sub-pixel structure for intra-pixel integration
        print("🔍 Calculating 127x127 PSF (oversampled 4x)...")
        psf_hdulist = wfi.calc_psf(fov_pixels=127, oversample=4)

        # 5. Get the oversampled numpy array (HDU 0 is oversampled)
        realistic_psf_oversampled = psf_hdulist[0].data
        
        # Ensure it's exactly 127*4 x 127*4
        S_orig = 127
        O = 4
        S = S_orig * O
        if realistic_psf_oversampled.shape != (S, S):
            print(f"⚠️ PSF shape {realistic_psf_oversampled.shape} != ({S}, {S}). Resizing...")
            curr_h, curr_w = realistic_psf_oversampled.shape
            pad_h = (S - curr_h) // 2
            pad_w = (S - curr_w) // 2
            if pad_h >= 0 and pad_w >= 0:
                realistic_psf_oversampled = np.pad(realistic_psf_oversampled, ((pad_h, S-curr_h-pad_h), (pad_w, S-curr_w-pad_w)))
            else:
                h_start = (curr_h - S) // 2
                w_start = (curr_w - S) // 2
                realistic_psf_oversampled = realistic_psf_oversampled[h_start:h_start+S, w_start:w_start+S]

        realistic_psf_oversampled = realistic_psf_oversampled / (realistic_psf_oversampled.sum() + 1e-9)
        
        # Save it
        torch.save(torch.from_numpy(realistic_psf_oversampled.astype(np.float32)), "roman_psf_prior_4x.pt")
        
        # Also save a 1x version for legacy/visualization
        realistic_psf_1x = realistic_psf_oversampled.reshape(S_orig, O, S_orig, O).mean(axis=(1, 3))
        torch.save(torch.from_numpy(realistic_psf_1x.astype(np.float32)), "roman_psf_prior.pt")
        
        plt.figure(figsize=(8, 8))
        # Use log scale for visualization to see the spikes
        plt.imshow(np.log10(realistic_psf_1x + 1e-6), cmap='inferno')
        plt.title("Realistic Roman PSF Prior (stpsf F146, 127x127)")
        plt.colorbar(label="log10(Intensity)")
        plt.savefig("roman_psf_prior.png")
        print("✅ Saved 508x508 realistic prior to roman_psf_prior_4x.pt and 127x127 to roman_psf_prior.pt")
        
    except Exception as e:
        print(f"❌ Failed to generate realistic PSF: {e}")
        import traceback
        traceback.print_exc()
        print("Falling back to a high-quality analytical model.")
        
        S_orig = 127
        O = 4
        S = S_orig * O
        half = S // 2
        y, x = np.meshgrid(np.arange(S) - half, np.arange(S) - half, indexing='ij')
        r = np.sqrt(x**2 + y**2)
        
        # Scale parameters for oversampled grid
        s_base = 1.5 * O
        psf = np.exp(-(r**2) / (2 * s_base**2))
        
        angles = [0, np.pi/3, 2*np.pi/3]
        for angle in angles:
            dist_to_line = np.abs(x * np.sin(angle) - y * np.cos(angle))
            # Wider spikes for the larger box
            psf += np.exp(-dist_to_line / (0.4 * O)) * 0.05 * np.exp(-r**2 / (2 * (40 * O)**2))
            
        psf /= psf.sum()
        torch.save(torch.from_numpy(psf.astype(np.float32)), "roman_psf_prior_4x.pt")
        
        # 1x version
        psf_1x = psf.reshape(S_orig, O, S_orig, O).mean(axis=(1, 3))
        torch.save(torch.from_numpy(psf_1x.astype(np.float32)), "roman_psf_prior.pt")
        print(f"✅ Saved analytical high-quality {S}x{S} prior to roman_psf_prior_4x.pt and {S_orig}x{S_orig} to roman_psf_prior.pt")

if __name__ == "__main__":
    generate_realistic_prior()
