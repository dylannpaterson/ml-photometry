import jax
import jax.numpy as jnp
from jax import vmap, lax
import numpy as np

def _get_jax_renderer():
    """Returns a JIT-compiled JAX renderer."""
    
    @jax.jit(static_argnums=(4,))
    def render_mosaic_jax(x, y, fluxes, kernel_bank, mosaic_size):
        """
        Render stars into a mosaic using JAX.
        
        Args:
            x, y: (N,) arrays of coordinates.
            fluxes: (N,) array of fluxes.
            kernel_bank: (16, K, K) array of kernels for 16 sub-pixel phases.
            mosaic_size: int, size of the square mosaic.
            
        Returns:
            (mosaic_size, mosaic_size) rendered image.
        """
        n_sub = 4
        # 1. Coordinate to grid index and phase
        ix = jnp.floor(x).astype(jnp.int32)
        iy = jnp.floor(y).astype(jnp.int32)
        px = jnp.clip(jnp.floor((x - ix) * n_sub).astype(jnp.int32), 0, n_sub - 1)
        py = jnp.clip(jnp.floor((y - iy) * n_sub).astype(jnp.int32), 0, n_sub - 1)
        
        phase_idx = py * n_sub + px
        
        # 2. Scatter stars into phase-specific grids
        # We handle out-of-bounds stars by masking
        mask = (ix >= 0) & (ix < mosaic_size) & (iy >= 0) & (iy < mosaic_size)
        
        grids = jnp.zeros((n_sub * n_sub, mosaic_size, mosaic_size))
        grids = grids.at[phase_idx, iy, ix].add(jnp.where(mask, fluxes, 0.0))
        
        # 3. Batched Convolution
        # JAX lax.conv_general_dilated is efficient for grouped/batched convs.
        # We need to reshape kernels to (16, 1, K, K) and grids to (1, 16, H, W).
        # Wait, to use feature_group_count, we want (1, 16, H, W) and (16, 1, K, K).
        
        k_h, k_w = kernel_bank.shape[1], kernel_bank.shape[2]
        pad = k_h // 2
        
        # Flip kernels because lax.conv is correlation, but we want convolution (or match fftconvolve)
        # Actually, if kernel_bank is symmetric it doesn't matter, but they are shifted.
        # To match scipy.signal.convolve2d(grid, kernel, mode='same'):
        # We flip both axes.
        kernels_flipped = kernel_bank[:, ::-1, ::-1]
        
        kernels_reshaped = kernels_flipped.reshape((n_sub * n_sub, 1, k_h, k_w))
        grids_reshaped = grids[jnp.newaxis, :, :, :] # (1, 16, H, W)
        
        rendered_phases = lax.conv_general_dilated(
            grids_reshaped,
            kernels_reshaped,
            window_strides=(1, 1),
            padding=[(pad, pad), (pad, pad)],
            feature_group_count=n_sub * n_sub
        )
        
        # Sum across phases (which are channels here)
        final_image = jnp.sum(rendered_phases, axis=1).squeeze()
        return final_image

    return render_mosaic_jax

_RENDER_FUNC = None

def render_gpu(x, y, fluxes, kernel_bank, mosaic_size):
    """Entry point for GPU rendering."""
    global _RENDER_FUNC
    if _RENDER_FUNC is None:
        _RENDER_FUNC = _get_jax_renderer()
    
    # Convert inputs to jax arrays if they aren't already
    x_jax = jnp.array(x)
    y_jax = jnp.array(y)
    fluxes_jax = jnp.array(fluxes)
    kernels_jax = jnp.array(kernel_bank)
    
    return np.array(_RENDER_FUNC(x_jax, y_jax, fluxes_jax, kernels_jax, mosaic_size))
