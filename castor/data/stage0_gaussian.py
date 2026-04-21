import numpy as np
import torch
from torch.utils.data import Dataset
import os
import h5py
from scipy.signal import fftconvolve
from scipy.ndimage import map_coordinates
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

from castor.data.transforms import AstroSpaceTransform
from castor.constants import DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL, GLOBAL_STRETCH_SCALE, SHAPE_SIZE

# Ensure STPSF_PATH is set for stpsf tool
if 'STPSF_PATH' not in os.environ:
    # Try common cluster and local paths
    candidate_paths = [
        os.path.expanduser('~/project_amber/stpsf-data'),
        os.path.expanduser('~/data/stpsf-data'),
        '/fs/project/gaudi.1/amber/stpsf-data',
    ]
    for path in candidate_paths:
        if os.path.exists(os.path.join(path, 'version.txt')):
            os.environ['STPSF_PATH'] = path
            print(f"🛰️ Auto-detected STPSF_PATH: {path}")
            break

# --- CENTRALIZED RENDERING LOGIC ---

def get_gaussian_psf(sigma=0.405, kernel_size=25):
    """
    Generates a perfectly centered analytical 2D Gaussian PSF kernel.
    Ensures the peak is at the exact center.
    Sigma 0.405 derived from F146 FWHM=0.105" and scale=0.11"/px.
    """
    center = (kernel_size - 1) / 2.0
    yy, xx = np.indices((kernel_size, kernel_size)) - center
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return (psf / (psf.sum() + 1e-9)).astype(np.float32)

def get_oversampled_gaussian_psf(sigma_detector, grid_size=129, oversample=4):
    """
    Generates a Gaussian PSF on an oversampled grid and block-reduces to 1x.
    Accepts sigma in detector pixel units.
    """
    S_oversampled = grid_size * oversample
    sigma_oversampled = sigma_detector * oversample
    
    # Generate in oversampled space (e.g. 516x516 for 129x129 @ 4x)
    psf_oversampled = get_gaussian_psf(sigma=sigma_oversampled, kernel_size=S_oversampled)
    
    # Block reduce to 1x detector resolution
    psf_1x = psf_oversampled.reshape(grid_size, oversample, grid_size, oversample).mean(axis=(1, 3))
    psf_1x /= (psf_1x.sum() + 1e-9)
    return psf_1x.astype(np.float32)

def render_gaussian_stars(height, width, px, py, fluxes, sigma=0.405, psf_kernel=None):
    """
    Vectorized star renderer using bilinear splatting followed by convolution.
    """
    if psf_kernel is None:
        psf_kernel = get_gaussian_psf(sigma=sigma)
        
    grid = np.zeros((height, width), dtype=np.float32)
    if len(fluxes) == 0:
        return grid
        
    x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
    dx, dy = px - x0, py - y0
    w00, w10, w01, w11 = (1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy
    
    def paint(gr, xi, yi, wi, fi):
        mask = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
        flat = yi[mask] * width + xi[mask]
        gr.flat += np.bincount(flat, weights=fi[mask] * wi[mask], minlength=gr.size)
    
    paint(grid, x0, y0, w00, fluxes)
    paint(grid, x0+1, y0, w10, fluxes)
    paint(grid, x0, y0+1, w01, fluxes)
    paint(grid, x0+1, y0+1, w11, fluxes)
    
    return fftconvolve(grid, psf_kernel, mode='same')

def fast_paint_grid(lx, ly, fluxes, snrs, sort_idx, min_snr, grid_size, cell_size, K):
    """
    Highly optimized target grid painter.

    Parameters
    ----------
    lx : numpy.ndarray
        Local x coordinates within cells.
    ly : numpy.ndarray
        Local y coordinates within cells.
    fluxes : numpy.ndarray
        Star fluxes.
    snrs : numpy.ndarray
        Signal-to-noise ratios.
    sort_idx : numpy.ndarray
        Indices for sorting stars.
    min_snr : float
        Minimum SNR threshold (currently not used directly in logic but passed).
    grid_size : int
        Size of the target grid.
    cell_size : int
        Size of each grid cell.
    K : int
        Maximum number of stars per cell.

    Returns
    -------
    numpy.ndarray
        The painted grid of shape [grid_size, grid_size, K, 5].
    """
    grid_stars = np.zeros((grid_size, grid_size, K, 5), dtype=np.float32)
    counts = np.zeros((grid_size, grid_size), dtype=np.int32)

    for idx in range(len(sort_idx)):
        i = sort_idx[idx]
        snr = snrs[i]

        if snr <= 1.0:
            target_p = 0.0
        elif snr >= 5.0:
            target_p = 1.0
        else:
            target_p = np.log10(snr) / 0.69897000433

        if target_p <= 0.0:
            continue

        cx = int(lx[i] // cell_size)
        cy = int(ly[i] // cell_size)

        if 0 <= cx < grid_size and 0 <= cy < grid_size:
            slot = counts[cy, cx]
            if slot < K:
                grid_stars[cy, cx, slot, 0] = target_p
                grid_stars[cy, cx, slot, 1] = lx[i] % cell_size
                grid_stars[cy, cx, slot, 2] = ly[i] % cell_size
                grid_stars[cy, cx, slot, 3] = fluxes[i]
                grid_stars[cy, cx, slot, 4] = snr
                counts[cy, cx] += 1

    return grid_stars

def sample_bulge_magnitudes(n_total, rc_mag, rc_sigma, rc_enhancement=3.0, m_min=12.0, m_max=32.0, gamma=0.3):
    """
    Samples from a power-law luminosity function with Red Clump enhancement.

    Parameters
    ----------
    n_total : int
        Total number of stars to sample.
    rc_mag : float
        Magnitude of the Red Clump.
    rc_sigma : float
        Sigma for the Red Clump Gaussian enhancement.
    rc_enhancement : float, optional
        Enhancement factor for Red Clump stars, by default 3.0.
    m_min : float, optional
        Minimum magnitude, by default 12.0.
    m_max : float, optional
        Maximum magnitude, by default 32.0.
    gamma : float, optional
        Power-law index, by default 0.3.

    Returns
    -------
    numpy.ndarray
        The sampled magnitudes.
    """
    u = np.random.uniform(0, 1, n_total)
    a = 10**(gamma * m_min)
    b = 10**(gamma * m_max)
    m_base = (1.0 / gamma) * np.log10(u * (b - a) + a)
    
    local_rgb_count = np.sum((m_base >= rc_mag - 0.5) & (m_base <= rc_mag + 0.5))
    n_rc = int(local_rgb_count * rc_enhancement)
    
    if n_rc > 0:
        m_rc = np.random.normal(loc=rc_mag, scale=rc_sigma, size=n_rc)
        m_all = np.concatenate([m_base, m_rc])
    else:
        m_all = m_base
        
    m_all = np.clip(m_all, m_min, m_max)
    np.random.shuffle(m_all)
    return m_all

def calculate_safe_magnitude_cutoff(exp_time, zp, sky_mag, read_noise=5.0, sigma=0.405, snr_cutoff=1.0):
    """
    Calculate the magnitude cutoff for a given SNR.

    Parameters
    ----------
    exp_time : float
        Exposure time in seconds.
    zp : float
        Zero point magnitude.
    sky_mag : float
        Sky brightness in mag/arcsec^2.
    read_noise : float, optional
        Read noise in electrons, by default 5.0.
    sigma : float, optional
        Gaussian sigma for PSF, by default 0.405.
    snr_cutoff : float, optional
        SNR threshold for the cutoff, by default 1.0.

    Returns
    -------
    float
        The calculated magnitude cutoff.
    """
    pixel_scale = 0.11
    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (pixel_scale**2) * exp_time
    n_pix = 12.0 
    bg_variance = n_pix * (sky_level + read_noise**2)
    a, b, c = 1.0, -(snr_cutoff**2), -(snr_cutoff**2) * bg_variance
    min_flux = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    return zp - 2.5 * np.log10(min_flux / exp_time)

def generate_dust_cirrus(img_size, amplitude, exponent=None):
    """
    Generates fractal dust noise with randomized power spectrum P(k) ~ k^-beta.

    Parameters
    ----------
    img_size : int
        The size of the square image.
    amplitude : float
        The maximum amplitude of the dust map.
    exponent : float, optional
        The power-law exponent (beta). If None, randomized between 2.5 and 4.0.

    Returns
    -------
    numpy.ndarray
        The generated dust map.
    """
    fx = np.fft.fftfreq(img_size)
    fy = np.fft.fftfreq(img_size)
    kx, ky = np.meshgrid(fx, fy)
    k = np.sqrt(kx**2 + ky**2)
    k[0, 0] = 1e-9 
    
    noise_fft = (np.random.normal(size=(img_size, img_size)) + 
                 1j * np.random.normal(size=(img_size, img_size)))
    
    # Beta typically ranges from 2.5 to 3.5 for ISM cirrus. 
    # We randomize this to give the network different textures to learn.
    if exponent is None:
        beta = np.random.uniform(2.5, 4.0)
    else:
        beta = exponent
        
    noise_fft *= k**(-beta / 2.0)
    noise_fft[0, 0] = 0.0 
    
    dust_map = np.real(np.fft.ifft2(noise_fft))
    dust_map -= dust_map.min()
    dust_map /= (dust_map.max() + 1e-9)
    return dust_map * amplitude

def generate_single_sample_stage0(idx, params):
    """
    Worker function to generate a single 256x256 Stage 0 sample.
    Uses Predictive Initialization: Reverse-engineers the input density 
    required to hit a target 'active' occupancy (SNR > 1) naturally.
    """
    img_size = params['image_size']
    exp_time = np.random.uniform(params['exp_time_min'], params['exp_time_max'])
    zp, sky_mag = params['zp'], params['sky_mag']
    
    # 1. Physics & Limits
    mag_limit_snr1 = calculate_safe_magnitude_cutoff(exp_time, zp, sky_mag, snr_cutoff=1.0)
    
    # 2. ML Target Definition
    cell_size, K = DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL
    grid_size = img_size // cell_size
    
    # Target 0.2 to 3.0 stars per cell (average)
    # Sampling up to K ensures we cover the network's full capacity
    target_lambda = np.random.uniform(0.2, 3.0)
    target_n_active = int(target_lambda * (grid_size ** 2))
    
    # 3. Super-Population Predictive Initialization
    gamma = np.random.uniform(0.25, 0.35)
    m_min, m_max_total = 12.0, 32.0
    m_rc = np.random.uniform(14.5, 16.5)
    rc_sigma = np.random.uniform(0.2, 0.5)
    rc_enh = np.random.uniform(5.0, 15.0)

    # Analytic fraction calculation: N(<m) ~ 10^(gamma*m)
    def get_lf_frac(m_limit):
        return (10**(gamma * m_limit) - 10**(gamma * m_min)) / (10**(gamma * m_max_total) - 10**(gamma * m_min))
    
    f_active_base = get_lf_frac(mag_limit_snr1)
    # RC stars are in a +/- 0.5 mag bin and are always active (bright)
    f_rc_bin = get_lf_frac(m_rc + 0.5) - get_lf_frac(m_rc - 0.5)
    
    # expected_active_per_base_star = (prob base star is active) + (prob extra RC star is added)
    expected_active_per_star = f_active_base + (f_rc_bin * rc_enh)
    
    n_pool = int(target_n_active / np.maximum(expected_active_per_star, 1e-6))
    n_pool = np.clip(n_pool, 100, 1_000_000) 
    
    # 4. Population Generation (No Culling)
    mags = sample_bulge_magnitudes(n_pool, m_rc, rc_sigma, rc_enh, m_min=m_min, m_max=m_max_total, gamma=gamma)
    px, py = np.random.uniform(0, img_size, len(mags)), np.random.uniform(0, img_size, len(mags))
    fluxes = exp_time * (10 ** (-0.4 * (mags - zp)))

    # 5. PSF (Randomized per chunk)
    sigma_detector = np.random.uniform(0.25, 1.0)
    psf_kernel = get_gaussian_psf(sigma=sigma_detector, kernel_size=25)
    chunk_fwhm = sigma_detector * 2.355

    # 6. Dust & Sky
    sky_level = (10 ** (-0.4 * (sky_mag - zp))) * (0.11**2) * exp_time
    raw_dust = generate_dust_cirrus(img_size, 1.0)
    transmission = 10 ** (-0.4 * (raw_dust ** np.random.uniform(1.0, 5.0)) * np.random.uniform(0.2, 5.0))
    
    star_transmissions = map_coordinates(transmission, [py, px], order=1, mode='nearest')
    apparent_fluxes = fluxes * star_transmissions
    
    sky_map = (sky_level * 0.4) + (sky_level * 0.6 * transmission) + (raw_dust * np.random.uniform(20, 100))
    
    # 7. Final Rendering & SNR Calculation
    def get_snrs(f, x, y, sm):
        render = render_gaussian_stars(img_size, img_size, x, y, f, psf_kernel=psf_kernel) + sm
        half = psf_kernel.shape[0] // 2
        star_peak = psf_kernel[half, half]
        N_eff = 1.0 / (np.sum(psf_kernel**2) + 1e-9)
        l_sky = map_coordinates(sm, [y, x], order=1, mode='nearest')
        l_full = map_coordinates(render, [y, x], order=1, mode='nearest')
        conf = np.maximum(0.0, l_full - (f * star_peak))
        s = f / np.sqrt(np.maximum(1.0, f + N_eff * (l_sky + conf + 25.0)))
        return s, render

    final_snrs, full_img = get_snrs(apparent_fluxes, px, py, sky_map)
    fg_mask = mags < mag_limit_snr1

    # Render background image (unresolved population)
    bg_img = render_gaussian_stars(img_size, img_size, px[~fg_mask], py[~fg_mask], apparent_fluxes[~fg_mask], psf_kernel=psf_kernel) + sky_map

    target_grid = fast_paint_grid(px[fg_mask], py[fg_mask], apparent_fluxes[fg_mask], final_snrs[fg_mask], np.argsort(apparent_fluxes[fg_mask])[::-1], 5.0, grid_size, cell_size, K)
    
    transform = AstroSpaceTransform(stretch_scale=GLOBAL_STRETCH_SCALE)
    chunk_median = np.median(full_img)
    bg_downsampled = bg_img.reshape(grid_size, cell_size, grid_size, cell_size).mean(axis=(1, 3))
    
    target_grid_full = np.concatenate([
        target_grid.reshape(grid_size, grid_size, -1), 
        transform.target_bg_to_network(bg_downsampled - chunk_median)[:, :, None]
    ], axis=-1)
    
    meta = np.zeros(17, dtype=np.float32)
    meta[0], meta[1], meta[2], meta[3] = exp_time, zp, sky_mag, float(chunk_fwhm)
    meta[7], meta[12] = 0.11, 5.0 
    
    return full_img.astype(np.float32), target_grid_full.astype(np.float32), chunk_median.astype(np.float32), meta

def run_stage0_parallel_generation(config, num_samples=None, num_workers=None):
    s0_cfg = config['curriculum']['stage0']
    d_cfg = config['data_params']
    total_samples = num_samples if num_samples is not None else (d_cfg['num_train_samples'] + d_cfg['num_val_samples'])
    output_path = os.path.join(s0_cfg['data_dir'], "stage0_data.h5")
    os.makedirs(s0_cfg['data_dir'], exist_ok=True)
    
    params = {
        'image_size': d_cfg.get('image_size', 256),
        'exp_time_min': d_cfg['physics_params']['exp_time_min'],
        'exp_time_max': d_cfg['physics_params']['exp_time_max'],
        'zp': d_cfg['physics_params']['zp'],
        'sky_mag': d_cfg['physics_params']['sky_mag']
    }
    
    if num_workers is None:
        num_workers = mp.cpu_count()
        
    # --- 🏗️ HDF5 SERIALIZATION BUDGET ---
    # Leave 1 worker for the main process to handle HDF5 writing and UI
    n_rendering_workers = max(1, num_workers - 1)
        
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    print(f"🔥 Stage 0 Unified Generation: {n_rendering_workers} Rendering Workers + 1 HDF5 Writer")
    
    with h5py.File(output_path, 'w') as f:
        dset_img = f.create_dataset("images", (total_samples, 256, 256), dtype='f4', chunks=(1, 256, 256), compression="lzf")
        dset_tgt = f.create_dataset("targets", (total_samples, 64, 64, MAX_CAPACITY_PER_CELL*5 + 1), dtype='f4', chunks=(1, 64, 64, MAX_CAPACITY_PER_CELL*5 + 1), compression="lzf")
        dset_med = f.create_dataset("chunk_medians", (total_samples,), dtype='f4')
        dset_meta = f.create_dataset("metas", (total_samples, 17), dtype='f4')
        
        worker_func = partial(generate_single_sample_stage0, params=params)
        with mp.Pool(n_rendering_workers) as pool:
            # imap_unordered for maximum throughput while main process writes
            results = pool.imap_unordered(worker_func, range(total_samples), chunksize=4)
            for i, (img, tgt, med, meta) in enumerate(tqdm(results, total=total_samples)):
                dset_img[i] = img
                dset_tgt[i] = tgt
                dset_med[i] = med
                dset_meta[i] = meta

class HDF5ChunkDataset(Dataset):
    """
    PyTorch Dataset for reading simulated training chunks from an HDF5 file.

    Attributes
    ----------
    h5_path : str
        Path to the HDF5 file.
    file : h5py.File
        HDF5 file handle.
    img_size : int
        Size of the square images.
    cell_size : int
        Size of each grid cell.
    grid_size : int
        Number of cells along one dimension.
    K : int
        Maximum number of stars per cell.
    transform : AstroSpaceTransform
        Transform for data augmentation/normalization.
    length : int
        Total number of samples in the dataset.
    """

    def __init__(self, h5_path, image_size=256):
        """
        Initialize the HDF5ChunkDataset.

        Parameters
        ----------
        h5_path : str
            Path to the HDF5 file.
            image_size : int, optional
            Size of the square images, by default 256.
        """
        self.h5_path, self.file, self.img_size = h5_path, None, image_size
        self.cell_size, self.grid_size, self.K = DEFAULT_CELL_SIZE, image_size // DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL
        self.transform = AstroSpaceTransform(stretch_scale=GLOBAL_STRETCH_SCALE)
        if not os.path.exists(self.h5_path): raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")
        
        with h5py.File(self.h5_path, 'r') as f:
            self.length = len(f['images'])

    def __len__(self):
        """
        Get the total number of samples.

        Returns
        -------
        int
            Number of samples.
        """
        return self.length

    def __getitem__(self, idx):
        """
        Get a single sample by index.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        dict
            A dictionary containing 'image', 'target', 'chunk_median', and 'meta'.
        """
        if self.file is None: self.file = h5py.File(self.h5_path, 'r', swmr=True, libver='latest')
        img = torch.from_numpy(self.file['images'][idx]).float()
        target = torch.from_numpy(self.file['targets'][idx]).float()
        median = float(self.file['chunk_medians'][idx])
        meta = self.file['metas'][idx] if 'metas' in self.file else np.zeros(17, dtype=np.float32)
        # Return a single mean PSF or None if we don't need it per-sample here
        return {"image": img, "target": target, "chunk_median": median, "meta": meta}

if __name__ == "__main__":
    import yaml
    import argparse
    parser = argparse.ArgumentParser(description="Generate Unified Stage 0 Gaussian Data")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    run_stage0_parallel_generation(config, num_samples=args.num_samples, num_workers=args.num_workers)
