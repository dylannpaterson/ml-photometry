import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm import tqdm

def gaussian_2d(coords, amplitude, xo, yo, sigma, background):
    """2D Gaussian function with a constant background."""
    x, y = coords
    g = amplitude * np.exp(-((x - xo)**2 + (y - yo)**2) / (2 * sigma**2)) + background
    return g.ravel()

def run_simulation(n_trials=1000, img_size=25, sigma=1.5, amplitude=100.0, background=10.0):
    """
    Runs a Monte Carlo simulation to estimate the precision of 2D Gaussian fitting.
    
    Args:
        n_trials: Number of Monte Carlo iterations.
        img_size: Size of the image stamp (pixels).
        sigma: Standard deviation of the Gaussian (pixels).
        amplitude: Peak amplitude of the Gaussian (counts above background).
        background: Constant background level (counts).
    """
    x = np.arange(0, img_size)
    y = np.arange(0, img_size)
    x, y = np.meshgrid(x, y)
    
    # True parameters: amplitude, xo, yo, sigma, background
    # We'll place the star at a random sub-pixel position near the center
    true_xo_base = img_size / 2.0
    true_yo_base = img_size / 2.0
    
    dx_errors = []
    dy_errors = []
    
    print(f"Running {n_trials} trials...")
    print(f"Parameters: Sigma={sigma}, Amplitude={amplitude}, Background={background}")
    print(f"Expected Peak SNR (approx): {amplitude / np.sqrt(amplitude + background):.2f}")

    for _ in tqdm(range(n_trials)):
        # Random sub-pixel shift within [-0.5, 0.5]
        xo = true_xo_base + np.random.uniform(-0.5, 0.5)
        yo = true_yo_base + np.random.uniform(-0.5, 0.5)
        
        # 1. Generate clean image
        clean_img = amplitude * np.exp(-((x - xo)**2 + (y - yo)**2) / (2 * sigma**2)) + background
        
        # 2. Add Poisson noise
        # Note: np.random.poisson expects the mean. 
        # For very high counts, this might overflow if not careful, but 100-1000 is fine.
        noisy_img = np.random.poisson(clean_img).astype(float)
        
        # 3. Fit the model
        # Initial guess: [amplitude, xo, yo, sigma, background]
        initial_guess = [amplitude, true_xo_base, true_yo_base, sigma, background]
        
        # Bounds to keep it stable: (min_vals, max_vals)
        bounds = ([0, 0, 0, 0.1, 0], [np.inf, img_size, img_size, img_size, np.inf])
        
        try:
            popt, pcov = curve_fit(gaussian_2d, (x, y), noisy_img.ravel(), p0=initial_guess, bounds=bounds)
            
            fit_xo, fit_yo = popt[1], popt[2]
            dx_errors.append(fit_xo - xo)
            dy_errors.append(fit_yo - yo)
        except Exception as e:
            # print(f"Fit failed: {e}")
            continue

    dx_errors = np.array(dx_errors)
    dy_errors = np.array(dy_errors)
    
    std_x = np.std(dx_errors)
    std_y = np.std(dy_errors)
    
    print("\nResults:")
    print(f"Precision dx (std): {std_x:.4f} pixels")
    print(f"Precision dy (std): {std_y:.4f} pixels")
    print(f"Combined precision: {np.sqrt(std_x**2 + std_y**2):.4f} pixels")
    
    # Optional: Plotting the last trial
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.title("Noisy Image")
    plt.imshow(noisy_img, origin='lower')
    plt.colorbar()
    
    plt.subplot(132)
    plt.title("Errors in X")
    plt.hist(dx_errors, bins=30, density=True)
    plt.xlabel("dx (fitted - true)")
    
    plt.subplot(133)
    plt.title("Errors in Y")
    plt.hist(dy_errors, bins=30, density=True)
    plt.xlabel("dy (fitted - true)")
    
    plt.tight_layout()
    plt.savefig("gaussian_fit_precision.png")
    print("\nPlot saved to gaussian_fit_precision.png")

if __name__ == "__main__":
    # You can adjust these parameters to match your typical SNR
    # SNR ~ Flux / sqrt(Flux + Area * (Background + Noise^2))
    # Here we use Poisson noise, so Noise^2 = Background.
    run_simulation(n_trials=2000, img_size=25, sigma=1.5, amplitude=100.0, background=20.0)
