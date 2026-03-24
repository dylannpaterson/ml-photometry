import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import math
import time

def _get_jax_renderer_core():
    """Returns the core rendering logic as a JIT-able function."""
    
    def render_core(x, y, fluxes, kernel_bank, mosaic_size):
        n_sub = 4
        ix = jnp.floor(x).astype(jnp.int32)
        iy = jnp.floor(y).astype(jnp.int32)
        px = jnp.clip(jnp.floor((x - ix) * n_sub).astype(jnp.int32), 0, n_sub - 1)
        py = jnp.clip(jnp.floor((y - iy) * n_sub).astype(jnp.int32), 0, n_sub - 1)
        
        phase_idx = py * n_sub + px
        mask = (ix >= 0) & (ix < mosaic_size) & (iy >= 0) & (iy < mosaic_size)
        
        grids = jnp.zeros((n_sub * n_sub, mosaic_size, mosaic_size))
        grids = grids.at[phase_idx, iy, ix].add(jnp.where(mask, fluxes, 0.0))
        
        k_h, k_w = kernel_bank.shape[1], kernel_bank.shape[2]
        pad = k_h // 2
        
        kernels_flipped = kernel_bank[:, ::-1, ::-1]
        kernels_reshaped = kernels_flipped.reshape((n_sub * n_sub, 1, k_h, k_w))
        grids_reshaped = grids[jnp.newaxis, :, :, :] 
        
        rendered_phases = lax.conv_general_dilated(
            grids_reshaped,
            kernels_reshaped,
            window_strides=(1, 1),
            padding=[(pad, pad), (pad, pad)],
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'), 
            feature_group_count=n_sub * n_sub
        )
        
        return jnp.sum(rendered_phases, axis=1).squeeze()
    
    return render_core

def _get_fused_generator_renderer():
    """Returns a JIT-compiled function that generates coords and renders."""
    render_core = _get_jax_renderer_core()
    
    def fused_op(key, fluxes, mags, kernel_bank, mosaic_size, mag_limit):
        n_stars = fluxes.shape[0]
        k1, k2 = jax.random.split(key)
        
        # 1. Generate Coordinates directly on GPU
        x = jax.random.uniform(k1, shape=(n_stars,), minval=0.0, maxval=float(mosaic_size))
        y = jax.random.uniform(k2, shape=(n_stars,), minval=0.0, maxval=float(mosaic_size))
        
        # 2. Render
        image = render_core(x, y, fluxes, kernel_bank, mosaic_size)
        
        # 3. Filter Mask based on dynamic limit
        catalog_mask = mags < mag_limit
        
        return image, x, y, catalog_mask

    return jax.jit(fused_op, static_argnums=(4,))

_FUSED_OP = None
_JAX_KEY = jax.random.PRNGKey(int(time.time())) if 'time' in globals() else jax.random.PRNGKey(42)

def render_generate_and_filter_gpu(fluxes, mags, kernel_bank, mosaic_size, mag_limit=27.0):
    """
    Fused GPU entry point with dynamic magnitude limit.
    """
    import time
    global _FUSED_OP, _JAX_KEY
    if _FUSED_OP is None:
        _FUSED_OP = _get_fused_generator_renderer()
    
    _JAX_KEY, subkey = jax.random.split(_JAX_KEY)
    
    # --- PADDING TO PREVENT CONSTANT RECOMPILATION ---
    current_size = len(fluxes)
    chunk_size = 1_000_000 
    padded_size = int(math.ceil(current_size / chunk_size)) * chunk_size
    pad_width = padded_size - current_size
    
    if pad_width > 0:
        fluxes_padded = np.pad(fluxes, (0, pad_width), constant_values=0.0)
        mags_padded = np.pad(mags, (0, pad_width), constant_values=99.0)
    else:
        fluxes_padded, mags_padded = fluxes, mags

    # Convert to JAX arrays
    fluxes_jax = jnp.array(fluxes_padded)
    mags_jax = jnp.array(mags_padded)
    kernels_jax = jnp.array(kernel_bank)
    
    # Run Fused Op
    image_jax, x_jax, y_jax, mask_jax = _FUSED_OP(subkey, fluxes_jax, mags_jax, kernels_jax, mosaic_size, mag_limit)
    
    # Transfer back
    image = np.array(image_jax)
    mask = np.array(mask_jax)
    
    valid_x = np.array(x_jax)[mask]
    valid_y = np.array(y_jax)[mask]
    valid_flux = np.array(fluxes_jax)[mask]
    valid_mags = np.array(mags_jax)[mask]
    
    return image, valid_x, valid_y, valid_flux, valid_mags

def render_gpu(x, y, fluxes, kernel_bank, mosaic_size):
    """Legacy entry point."""
    global _RENDER_CORE
    if '_RENDER_CORE' not in globals():
        global _RENDER_CORE
        _RENDER_CORE = jax.jit(_get_jax_renderer_core(), static_argnums=(4,))
    
    current_size = len(x)
    chunk_size = 1_000_000 
    padded_size = int(math.ceil(current_size / chunk_size)) * chunk_size
    pad_width = padded_size - current_size
    
    if pad_width > 0:
        x_padded = np.pad(x, (0, pad_width), constant_values=-1.0)
        y_padded = np.pad(y, (0, pad_width), constant_values=-1.0)
        fluxes_padded = np.pad(fluxes, (0, pad_width), constant_values=0.0)
    else:
        x_padded, y_padded, fluxes_padded = x, y, fluxes

    return np.array(_RENDER_CORE(jnp.array(x_padded), jnp.array(y_padded), jnp.array(fluxes_padded), jnp.array(kernel_bank), mosaic_size))
