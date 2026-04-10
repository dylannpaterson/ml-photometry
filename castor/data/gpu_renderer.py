import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import math
import time
from castor.constants import SHAPE_SIZE

def is_gpu_available():
    try:
        return any(d.platform == 'gpu' for d in jax.devices())
    except:
        return False

def _get_jax_renderer_core():
    def render_core(x, y, fluxes, repr_psf_4x, mosaic_size):
        O = 4 
        S = SHAPE_SIZE
        
        # 1. Bin the single PSF into 1x (Centered at S//2)
        # Using reshaped mean for exact centering as in CPU version
        psf_1x = repr_psf_4x.reshape(S, O, S, O).mean(axis=(1, 3))
        psf_1x = psf_1x / (jnp.sum(psf_1x) + 1e-9)
        
        # 2. Bi-linear placement
        x0, y0 = jnp.floor(x).astype(jnp.int32), jnp.floor(y).astype(jnp.int32)
        dx, dy = x - x0, y - y0
        
        valid00 = (x0 >= 0) & (x0 < mosaic_size) & (y0 >= 0) & (y0 < mosaic_size)
        valid10 = (x0+1 >= 0) & (x0+1 < mosaic_size) & (y0 >= 0) & (y0 < mosaic_size)
        valid01 = (x0 >= 0) & (x0 < mosaic_size) & (y0+1 >= 0) & (y0+1 < mosaic_size)
        valid11 = (x0+1 >= 0) & (x0+1 < mosaic_size) & (y0+1 >= 0) & (y0+1 < mosaic_size)
        
        grid = jnp.zeros((mosaic_size, mosaic_size))
        grid = grid.at[y0, x0].add(jnp.where(valid00, fluxes * (1-dx) * (1-dy), 0.0))
        grid = grid.at[y0, x0+1].add(jnp.where(valid10, fluxes * dx * (1-dy), 0.0))
        grid = grid.at[y0+1, x0].add(jnp.where(valid01, fluxes * (1-dx) * dy, 0.0))
        grid = grid.at[y0+1, x0+1].add(jnp.where(valid11, fluxes * dx * dy, 0.0))
        
        # 3. Single Convolution
        kernel = psf_1x[::-1, ::-1].reshape((1, 1, S, S))
        pad = S // 2
        
        convolved = lax.conv_general_dilated(
            grid[jnp.newaxis, jnp.newaxis, :, :],
            kernel,
            window_strides=(1, 1),
            padding=[(pad, pad), (pad, pad)],
            dimension_numbers=('NCHW', 'OIHW', 'NCHW')
        )
        
        return jnp.maximum(0, convolved.squeeze())
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
