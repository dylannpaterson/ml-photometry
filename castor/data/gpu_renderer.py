import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import math
import time

def _get_jax_renderer_core():
    """
    PCA-optimized JAX renderer with Eigen-Jiggling.
    Draws all stars with the mean PSF, but only applies eigen-corrections
    to stars brighter than the magnitude limit.
    """
    def render_core(x, y, fluxes, mags, psf_indices, weights_lib, mean_psf, eigen_psfs, s_vals, mosaic_size, mag_limit, key):
        n_pca = eigen_psfs.shape[0]
        n_stars = fluxes.shape[0]
        
        # 1. Bilinear Sub-pixel Distribution (The 4-pixel footprint)
        x0 = jnp.floor(x).astype(jnp.int32)
        y0 = jnp.floor(y).astype(jnp.int32)
        dx = x - x0
        dy = y - y0
        
        # Boundary Masks
        mask00 = (x0 >= 0) & (x0 < mosaic_size) & (y0 >= 0) & (y0 < mosaic_size)
        mask10 = (x0+1 >= 0) & (x0+1 < mosaic_size) & (y0 >= 0) & (y0 < mosaic_size)
        mask01 = (x0 >= 0) & (x0 < mosaic_size) & (y0+1 >= 0) & (y0+1 < mosaic_size)
        mask11 = (x0+1 >= 0) & (x0+1 < mosaic_size) & (y0+1 >= 0) & (y0+1 < mosaic_size)
        
        # --- Base Pass (All Stars) ---
        # Bilinear scatter into a single grid for the mean PSF
        base_grid = jnp.zeros((1, mosaic_size, mosaic_size))
        base_grid = base_grid.at[0, y0, x0].add(jnp.where(mask00, fluxes * (1-dx) * (1-dy), 0.0))
        base_grid = base_grid.at[0, y0, x0+1].add(jnp.where(mask10, fluxes * dx * (1-dy), 0.0))
        base_grid = base_grid.at[0, y0+1, x0].add(jnp.where(mask01, fluxes * (1-dx) * dy, 0.0))
        base_grid = base_grid.at[0, y0+1, x0+1].add(jnp.where(mask11, fluxes * dx * dy, 0.0))
        
        # --- Eigen-Jiggling Pass (Bright Stars Only) ---
        is_bright = mags < mag_limit
        bright_fluxes = jnp.where(is_bright, fluxes, 0.0)
        
        # Assign base weights from library
        base_star_weights = weights_lib[psf_indices] # [N, n_pca]
        
        # Add perturbation proportional to singular values (Simulate local breathing/jitter)
        perturb_scale = 0.05
        noise = jax.random.normal(key, shape=(n_stars, n_pca)) * (s_vals * perturb_scale)
        star_weights = base_star_weights + noise
        
        # Scatter each eigen-component using bilinear weights
        # We use a scan here to handle 20 components efficiently
        def scatter_eigen(carry, i):
            w_f = bright_fluxes * star_weights[:, i]
            grid = jnp.zeros((mosaic_size, mosaic_size))
            grid = grid.at[y0, x0].add(jnp.where(mask00, w_f * (1-dx) * (1-dy), 0.0))
            grid = grid.at[y0, x0+1].add(jnp.where(mask10, w_f * dx * (1-dy), 0.0))
            grid = grid.at[y0+1, x0].add(jnp.where(mask01, w_f * (1-dx) * dy, 0.0))
            grid = grid.at[y0+1, x0+1].add(jnp.where(mask11, w_f * dx * dy, 0.0))
            return carry, grid

        _, eigen_grids = jax.lax.scan(scatter_eigen, None, jnp.arange(n_pca))
            
        # Convolutions
        k_h, k_w = mean_psf.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        
        # 11-Pass Eigen-Convolution (1 Base + N_PCA Eigen)
        base_kernel = mean_psf[::-1, ::-1].reshape((1, 1, k_h, k_w))
        base_rendered = lax.conv_general_dilated(
            base_grid[jnp.newaxis, :, :, :],
            base_kernel,
            window_strides=(1, 1),
            padding=[(pad_h, pad_h), (pad_w, pad_w)],
            dimension_numbers=('NCHW', 'OIHW', 'NCHW')
        )
        
        eigen_kernels = eigen_psfs[:, ::-1, ::-1].reshape((n_pca, 1, k_h, k_w))
        eigen_rendered = lax.conv_general_dilated(
            eigen_grids[jnp.newaxis, :, :, :],
            eigen_kernels,
            window_strides=(1, 1),
            padding=[(pad_h, pad_h), (pad_w, pad_w)],
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
            feature_group_count=n_pca
        )
        
        final_image = (base_rendered + jnp.sum(eigen_rendered, axis=1)).squeeze()
        return jnp.maximum(0, final_image), star_weights
    
    return render_core

def _get_fused_generator_renderer():
    render_core = _get_jax_renderer_core()
    
    def fused_op(key, fluxes, mags, weights_lib, mean_psf, eigen_psfs, s_vals, mosaic_size, mag_limit):
        n_stars = fluxes.shape[0]
        num_psfs = weights_lib.shape[0]
        k1, k2, k3, k4 = jax.random.split(key, 4)
        
        # Coordinate Generation
        x = jax.random.uniform(k1, shape=(n_stars,), minval=0.0, maxval=float(mosaic_size))
        y = jax.random.uniform(k2, shape=(n_stars,), minval=0.0, maxval=float(mosaic_size))
        
        # PSF Library Selection
        psf_indices = jax.random.randint(k3, shape=(n_stars,), minval=0, maxval=num_psfs)
        
        # Jiggled Rendering
        image, final_star_weights = render_core(x, y, fluxes, mags, psf_indices, weights_lib, mean_psf, eigen_psfs, s_vals, mosaic_size, mag_limit, k4)
        
        # Catalog Filter
        catalog_mask = mags < mag_limit
        
        return image, x, y, psf_indices, catalog_mask, final_star_weights

    return jax.jit(fused_op, static_argnums=(7, 8))

_FUSED_OP = None
_JAX_KEY = jax.random.PRNGKey(int(time.time())) if 'time' in globals() else jax.random.PRNGKey(42)

def render_generate_and_filter_gpu(fluxes, mags, weights_lib, mean_psf, eigen_psfs, s_vals, mosaic_size, mag_limit=27.0):
    global _FUSED_OP, _JAX_KEY
    if _FUSED_OP is None:
        _FUSED_OP = _get_fused_generator_renderer()
    
    _JAX_KEY, subkey = jax.random.split(_JAX_KEY)
    
    current_size = len(fluxes)
    chunk_size = 100_000
    padded_size = int(math.ceil(current_size / chunk_size)) * chunk_size
    pad_width = padded_size - current_size
    
    if pad_width > 0:
        fluxes_padded = np.pad(fluxes, (0, pad_width), constant_values=0.0)
        mags_padded = np.pad(mags, (0, pad_width), constant_values=99.0)
    else:
        fluxes_padded, mags_padded = fluxes, mags

    fluxes_jax = jnp.array(fluxes_padded)
    mags_jax = jnp.array(mags_padded)
    weights_jax = jnp.array(weights_lib)
    mean_psf_jax = jnp.array(mean_psf)
    eigen_psfs_jax = jnp.array(eigen_psfs)
    s_vals_jax = jnp.array(s_vals)
    
    image_jax, x_jax, y_jax, psf_jax, mask_jax, weights_jax_out = _FUSED_OP(
        subkey, fluxes_jax, mags_jax, weights_jax, mean_psf_jax, eigen_psfs_jax, s_vals_jax, mosaic_size, float(mag_limit)
    )
    
    image = np.array(image_jax)
    mask = np.array(mask_jax)[:current_size]
    
    x_v = np.array(x_jax)[:current_size][mask]
    y_v = np.array(y_jax)[:current_size][mask]
    psf_v = np.array(psf_jax)[:current_size][mask]
    flux_v = fluxes[mask]
    mag_v = mags[mask]
    weights_v = np.array(weights_jax_out)[:current_size][mask]
    
    return image, x_v, y_v, psf_v, flux_v, mag_v, weights_v
