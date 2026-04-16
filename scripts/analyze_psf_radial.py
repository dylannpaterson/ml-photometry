import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def moffat(r, alpha, beta, amplitude):
    return amplitude * (1 + (r / alpha)**2)**(-beta)

def analyze_psf():
    data = torch.load('master_psf_library.pt', map_location='cpu', weights_only=False)
    mean_psf = data['mean_psf']
    
    O = 4 # 4x oversampled
    size = mean_psf.shape[0]
    center = (size - 1) / 2.0
    
    y, x = np.ogrid[:size, :size]
    r = np.sqrt((x - center)**2 + (y - center)**2)
    
    # Normalize
    mean_psf /= mean_psf.sum()
    
    r_flat = r.ravel() / O # 1x pixel units
    psf_flat = mean_psf.ravel()
    
    # Radial binning (High resolution for stable fit)
    r_bins = np.linspace(0, 15, 200)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    psf_profile = []
    for i in range(len(r_bins)-1):
        mask = (r_flat >= r_bins[i]) & (r_flat < r_bins[i+1])
        if np.any(mask):
            psf_profile.append(np.mean(psf_flat[mask]))
        else:
            psf_profile.append(0)
    psf_profile = np.array(psf_profile)
    
    # --- STABLE LOG-LOG CUBIC FIT ---
    # r_core handles the flattening at r=0
    r_core = 0.5 
    log_r = np.log10(r_centers + r_core)
    log_p = np.log10(np.maximum(psf_profile, 1e-10))
    
    # Weight the fit by intensity to prioritize the core
    weights = np.sqrt(psf_profile / psf_profile.max())
    
    # Fit region (r < 12px)
    fit_mask = r_centers <= 12.0
    coeffs = np.polyfit(log_r[fit_mask], log_p[fit_mask], 3, w=weights[fit_mask])
    poly = np.poly1d(coeffs)
    
    print(f"✨ Log-Log Cubic Fit Coefficients (log10(r+{r_core})):")
    print(f"   {coeffs}")

    # Model values
    psf_model = 10**poly(np.log10(r_centers + r_core))

    # --- PIECEWISE LINEAR (LOG-LOG) / BROKEN POWER LAW ---
    # This captures the "linear then linear again" behavior
    def broken_log_pl(r, r_b, slope1, slope2, amp):
        log_r = np.log10(r + 0.1)
        log_rb = np.log10(r_b + 0.1)
        # Linear in log-log is amp + slope * (log_r - log_rb)
        y = np.where(r <= r_b, 
                     amp + slope1 * (log_r - log_rb),
                     amp + slope2 * (log_r - log_rb))
        return 10**y

    try:
        popt_bp, _ = curve_fit(broken_log_pl, r_centers[fit_mask], psf_profile[fit_mask], 
                               p0=[3.5, -1.0, -4.0, np.log10(psf_profile[np.abs(r_centers-3.5).argmin()])])
        print(f"✨ Broken Power Law (Piecewise Log-Log):")
        print(f"   Break Radius: {popt_bp[0]:.2f}")
        print(f"   Inner Slope:  {popt_bp[1]:.2f}")
        print(f"   Outer Slope:  {popt_bp[2]:.2f}")
    except:
        popt_bp = None

    # Plotting
    plt.figure(figsize=(16, 7))
    
    # 1. Log-Log Plot
    plt.subplot(1, 2, 1)
    plt.loglog(r_centers, psf_profile, 'ko', markersize=2, alpha=0.3, label='Data')
    plt.loglog(r_centers, psf_model, 'r-', linewidth=3, label='Cubic Fit')
    if popt_bp is not None:
        plt.loglog(r_centers, broken_log_pl(r_centers, *popt_bp), 'b--', linewidth=2, label='Piecewise Fit')
    
    plt.axvline(3.5, color='gray', linestyle=':', label='Observed Break (3.5)')
    plt.xlabel("Radius (1x pixels)")
    plt.ylabel("Intensity")
    plt.title("PSF Profile Comparison (Log-Log)")
    plt.ylim(1e-7, psf_profile.max()*2)
    plt.xlim(0.1, 15)
    plt.legend()
    plt.grid(True, which='both', alpha=0.1)
    
    # 2. Linear Core Comparison
    plt.subplot(1, 2, 2)
    plt.plot(r_centers, psf_profile, 'ko', markersize=2, alpha=0.3, label='Data')
    plt.plot(r_centers, psf_model, 'r-', linewidth=2, label='Cubic Fit')
    if popt_bp is not None:
        plt.plot(r_centers, broken_log_pl(r_centers, *popt_bp), 'b--', label='Piecewise Fit')
    plt.xlim(0, 6)
    plt.ylim(0, psf_profile.max()*1.1)
    plt.xlabel("Radius (1x pixels)")
    plt.ylabel("Intensity")
    plt.title("PSF Core Detail (Linear)")
    plt.legend()
    plt.grid(alpha=0.2)
    
    plt.tight_layout()
    plt.savefig("psf_radial_diagnostic.png")
    print("✨ Diagnostic saved to psf_radial_diagnostic.png")

if __name__ == "__main__":
    analyze_psf()
