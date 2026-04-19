import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import math
import time
from castor.constants import SHAPE_SIZE

def is_gpu_available():
    """
    Checks if a GPU is available for JAX.

    Returns
    -------
    bool
        True if a GPU is detected, False otherwise.
    """
    try:
        return any(d.platform == 'gpu' for d in jax.devices())
    except:
        return False

def _get_jax_renderer_core():
    """
    Returns the core JAX-based rendering function.

    Returns
    -------
    callable
        A function that performs the core rendering logic using JAX.
    """
    def render_core(x, y, fluxes, repr_psf_4x, mosaic_size, v_mask):
        O = 4 
        S = SHAPE_SIZE
        
        # 1. Bin the single PSF into 1x (Centered at S//2)
        psf_1x = repr_psf_4x.reshape(S, O, S, O).mean(axis=(1, 3))
        psf_1x = psf_1x / (jnp.sum(psf_1x) + 1e-9)
        
        # 2. Bi-linear placement
        x0, y0 = jnp.floor(x).astype(jnp.int32), jnp.floor(y).astype(jnp.int32)
        dx, dy = x - x0, y - y0
        
        valid00 = (x0 >= 0) & (x0 < mosaic_size) & (y0 >= 0) & (y0 < mosaic_size)
        valid10 = (x0+1 >= 0) & (x0+1 < mosaic_size) & (y0 >= 0) & (y0 < mosaic_size)
        valid01 = (x0 >= 0) & (x0 < mosaic_size) & (y0+1 >= 0) & (y0+1 < mosaic_size)
        valid11 = (x0+1 >= 0) & (x0+1 < mosaic_size) & (y0+1 >= 0) & (y0+1 < mosaic_size)
        
        # Separate grids for Resolved and Unresolved
        fg_grid = jnp.zeros((mosaic_size, mosaic_size))
        bg_grid = jnp.zeros((mosaic_size, mosaic_size))
        
        # Helper to add flux to a grid based on mask
        def add_flux(grid, mask, f, w):
            return grid.at[y0, x0].add(jnp.where(valid00 & mask, f * (1-dx) * (1-dy), 0.0))\
                       .at[y0, x0+1].add(jnp.where(valid10 & mask, f * dx * (1-dy), 0.0))\
                       .at[y0+1, x0].add(jnp.where(valid01 & mask, f * (1-dx) * dy, 0.0))\
                       .at[y0+1, x0+1].add(jnp.where(valid11 & mask, f * dx * dy, 0.0))

        fg_grid = add_flux(fg_grid, v_mask, fluxes, 1.0)
        bg_grid = add_flux(bg_grid, ~v_mask, fluxes, 1.0)
        
        # 3. Separate Convolutions
        kernel = psf_1x[::-1, ::-1].reshape((1, 1, S, S))
        pad = S // 2
        
        def convolve(g):
            return lax.conv_general_dilated(
                g[jnp.newaxis, jnp.newaxis, :, :],
                kernel,
                window_strides=(1, 1),
                padding=[(pad, pad), (pad, pad)],
                dimension_numbers=('NCHW', 'OIHW', 'NCHW')
            ).squeeze()
        
        fg_image = convolved_fg = convolve(fg_grid)
        bg_image = convolved_bg = convolve(bg_grid)
        
        return jnp.maximum(0, fg_image + bg_image), jnp.maximum(0, bg_image)
    return render_core

def _get_fused_generator_renderer():
    """
    Returns a JIT-compiled fused generator and renderer.

    Returns
    -------
    callable
        A JIT-compiled function that generates star positions and renders them.
    """
    render_core = _get_jax_renderer_core()
    def fused_op(key, fluxes, mags, single_psf, mosaic_size, mag_limit):
        n_stars = fluxes.shape[0]
        k1, k2 = jax.random.split(key, 2)
        x = jax.random.uniform(k1, shape=(n_stars,), minval=0.0, maxval=float(mosaic_size))
        y = jax.random.uniform(k2, shape=(n_stars,), minval=0.0, maxval=float(mosaic_size))
        
        v_mask = mags < mag_limit
        full_image, bg_image = render_core(x, y, fluxes, single_psf, mosaic_size, v_mask)
        return full_image, bg_image, x, y, v_mask
    return jax.jit(fused_op, static_argnums=(4, 5))

_FUSED_OP = None
_JAX_KEY = jax.random.PRNGKey(int(time.time())) if 'time' in globals() else jax.random.PRNGKey(42)

def render_generate_and_filter_gpu(fluxes, mags, single_psf, mosaic_size, mag_limit=27.0):
    """
    Accelerated GPU renderer for large mosaic simulations using JAX.

    Parameters
    ----------
    fluxes : numpy.ndarray
        Array of star fluxes.
    mags : numpy.ndarray
        Array of star magnitudes.
    single_psf : numpy.ndarray
        High-resolution PSF kernel.
    mosaic_size : int
        Size of the output mosaic image.
    mag_limit : float, optional
        Magnitude threshold for separating foreground and background, by default 27.0.

    Returns
    -------
    tuple
        A tuple (full_image, bg_image, x, y, fluxes_filtered, mags_filtered) 
        containing the rendered images and filtered star catalogs.
    """
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

    full_image_jax, bg_image_jax, x_jax, y_jax, mask_jax = _FUSED_OP(
        subkey, jnp.array(fluxes_padded), jnp.array(mags_padded), 
        jnp.array(single_psf), mosaic_size, float(mag_limit)
    )
    
    full_image = np.array(full_image_jax)
    bg_image = np.array(bg_image_jax)
    mask = np.array(mask_jax)[:current_size]
    
    return full_image, bg_image, np.array(x_jax)[:current_size][mask], np.array(y_jax)[:current_size][mask], fluxes[mask], mags[mask]
