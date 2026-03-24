import jax
import jax.numpy as jnp
from jax import vmap, lax
import numpy as np
import math

def _get_jax_renderer():
    """Returns a JIT-compiled JAX renderer."""
    
    def render_mosaic_jax(x, y, fluxes, kernel_bank, mosaic_size):
        n_sub = 4
        
        # 1. Coordinate to grid index and phase
        ix = jnp.floor(x).astype(jnp.int32)
        iy = jnp.floor(y).astype(jnp.int32)
        px = jnp.clip(jnp.floor((x - ix) * n_sub).astype(jnp.int32), 0, n_sub - 1)
        py = jnp.clip(jnp.floor((y - iy) * n_sub).astype(jnp.int32), 0, n_sub - 1)
        
        phase_idx = py * n_sub + px
        
        # 2. Scatter stars into phase-specific grids
        # Mask out out-of-bounds stars AND padded dummy stars
        mask = (ix >= 0) & (ix < mosaic_size) & (iy >= 0) & (iy < mosaic_size)
        
        grids = jnp.zeros((n_sub * n_sub, mosaic_size, mosaic_size))
        grids = grids.at[phase_idx, iy, ix].add(jnp.where(mask, fluxes, 0.0))
        
        # 3. Batched Convolution
        k_h, k_w = kernel_bank.shape[1], kernel_bank.shape[2]
        pad = k_h // 2
        
        kernels_flipped = kernel_bank[:, ::-1, ::-1]
        kernels_reshaped = kernels_flipped.reshape((n_sub * n_sub, 1, k_h, k_w))
        grids_reshaped = grids[jnp.newaxis, :, :, :] # (1, 16, H, W)
        
        # Explicit dimension_numbers guarantee stable memory layout across hardware
        rendered_phases = lax.conv_general_dilated(
            grids_reshaped,
            kernels_reshaped,
            window_strides=(1, 1),
            padding=[(pad, pad), (pad, pad)],
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'), 
            feature_group_count=n_sub * n_sub
        )
        
        final_image = jnp.sum(rendered_phases, axis=1).squeeze()
        return final_image

    return jax.jit(render_mosaic_jax, static_argnums=(4,))

_RENDER_FUNC = None

def render_gpu(x, y, fluxes, kernel_bank, mosaic_size):
    """Entry point for GPU rendering."""
    global _RENDER_FUNC
    if _RENDER_FUNC is None:
        _RENDER_FUNC = _get_jax_renderer()
    
    # --- PADDING TO PREVENT CONSTANT RECOMPILATION ---
    # We round up the array size to the nearest million. 
    # JAX will only compile once for each unique padded size.
    current_size = len(x)
    chunk_size = 1_000_000 
    padded_size = int(math.ceil(current_size / chunk_size)) * chunk_size
    pad_width = padded_size - current_size
    
    if pad_width > 0:
        # Pad with dummy negative coordinates and zero flux so they are safely masked
        x_padded = np.pad(x, (0, pad_width), constant_values=-1.0)
        y_padded = np.pad(y, (0, pad_width), constant_values=-1.0)
        fluxes_padded = np.pad(fluxes, (0, pad_width), constant_values=0.0)
    else:
        x_padded, y_padded, fluxes_padded = x, y, fluxes

    # Convert to JAX arrays
    x_jax = jnp.array(x_padded)
    y_jax = jnp.array(y_padded)
    fluxes_jax = jnp.array(fluxes_padded)
    kernels_jax = jnp.array(kernel_bank)
    
    return np.array(_RENDER_FUNC(x_jax, y_jax, fluxes_jax, kernels_jax, mosaic_size))
