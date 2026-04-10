import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import math
import time

def is_gpu_available():
    try:
        return any(d.platform == 'gpu' for d in jax.devices())
    except:
        return False

def _get_jax_renderer_core():
    def render_core(x, y, fluxes, single_psf, mosaic_size):
        x0 = jnp.floor(x).astype(jnp.int32)
        y0 = jnp.floor(y).astype(jnp.int32)
        dx = x - x0
        dy = y - y0
        
        mask00 = (x0 >= 0) & (x0 < mosaic_size) & (y0 >= 0) & (y0 < mosaic_size)
        mask10 = (x0+1 >= 0) & (x0+1 < mosaic_size) & (y0 >= 0) & (y0 < mosaic_size)
        mask01 = (x0 >= 0) & (x0 < mosaic_size) & (y0+1 >= 0) & (y0+1 < mosaic_size)
        mask11 = (x0+1 >= 0) & (x0+1 < mosaic_size) & (y0+1 >= 0) & (y0+1 < mosaic_size)
        
        base_grid = jnp.zeros((1, mosaic_size, mosaic_size))
        base_grid = base_grid.at[0, y0, x0].add(jnp.where(mask00, fluxes * (1-dx) * (1-dy), 0.0))
        base_grid = base_grid.at[0, y0, x0+1].add(jnp.where(mask10, fluxes * dx * (1-dy), 0.0))
        base_grid = base_grid.at[0, y0+1, x0].add(jnp.where(mask01, fluxes * (1-dx) * dy, 0.0))
        base_grid = base_grid.at[0, y0+1, x0+1].add(jnp.where(mask11, fluxes * dx * dy, 0.0))
        
        # Single fast convolution instead of PCA
        k_h, k_w = single_psf.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        
        base_kernel = single_psf[::-1, ::-1].reshape((1, 1, k_h, k_w))
        base_rendered = lax.conv_general_dilated(
            base_grid[jnp.newaxis, :, :, :],
            base_kernel,
            window_strides=(1, 1),
            padding=[(pad_h, pad_h), (pad_w, pad_w)],
            dimension_numbers=('NCHW', 'OIHW', 'NCHW')
        )
        return jnp.maximum(0, base_rendered.squeeze())
    return render_core

def _get_fused_generator_renderer():
    render_core = _get_jax_renderer_core()
    def fused_op(key, fluxes, mags, single_psf, mosaic_size, mag_limit):
        n_stars = fluxes.shape[0]
        k1, k2 = jax.random.split(key, 2)
        x = jax.random.uniform(k1, shape=(n_stars,), minval=0.0, maxval=float(mosaic_size))
        y = jax.random.uniform(k2, shape=(n_stars,), minval=0.0, maxval=float(mosaic_size))
        
        image = render_core(x, y, fluxes, single_psf, mosaic_size)
        catalog_mask = mags < mag_limit
        return image, x, y, catalog_mask
    return jax.jit(fused_op, static_argnums=(4, 5))

_FUSED_OP = None
_JAX_KEY = jax.random.PRNGKey(int(time.time())) if 'time' in globals() else jax.random.PRNGKey(42)

def render_generate_and_filter_gpu(fluxes, mags, single_psf, mosaic_size, mag_limit=27.0):
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

    image_jax, x_jax, y_jax, mask_jax = _FUSED_OP(
        subkey, jnp.array(fluxes_padded), jnp.array(mags_padded), 
        jnp.array(single_psf), mosaic_size, float(mag_limit)
    )
    
    image = np.array(image_jax)
    mask = np.array(mask_jax)[:current_size]
    
    return image, np.array(x_jax)[:current_size][mask], np.array(y_jax)[:current_size][mask], fluxes[mask], mags[mask]
