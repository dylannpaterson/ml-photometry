import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import math
import time

def _get_jax_renderer_core():
    def render_core(x, y, fluxes, psf_indices, kernel_bank, mosaic_size):
        num_kernels = kernel_bank.shape[0]
        
        # 1. Bilinear Sub-pixel Distribution (The Fix)
        x0 = jnp.floor(x).astype(jnp.int32)
        y0 = jnp.floor(y).astype(jnp.int32)
        dx = x - x0
        dy = y - y0
        
        # Boundary Masks for 2x2 grid
        mask00 = (x0 >= 0) & (x0 < mosaic_size) & (y0 >= 0) & (y0 < mosaic_size)
        mask10 = (x0+1 >= 0) & (x0+1 < mosaic_size) & (y0 >= 0) & (y0 < mosaic_size)
        mask01 = (x0 >= 0) & (x0 < mosaic_size) & (y0+1 >= 0) & (y0+1 < mosaic_size)
        mask11 = (x0+1 >= 0) & (x0+1 < mosaic_size) & (y0+1 >= 0) & (y0+1 < mosaic_size)
        
        grids = jnp.zeros((num_kernels, mosaic_size, mosaic_size))
        
        # Scatter 4-pixel footprint for every star
        grids = grids.at[psf_indices, y0, x0].add(jnp.where(mask00, fluxes * (1-dx) * (1-dy), 0.0))
        grids = grids.at[psf_indices, y0, x0+1].add(jnp.where(mask10, fluxes * dx * (1-dy), 0.0))
        grids = grids.at[psf_indices, y0+1, x0].add(jnp.where(mask01, fluxes * (1-dx) * dy, 0.0))
        grids = grids.at[psf_indices, y0+1, x0+1].add(jnp.where(mask11, fluxes * dx * dy, 0.0))
        
        k_h, k_w = kernel_bank.shape[1], kernel_bank.shape[2]
        pad_h, pad_w = k_h // 2, k_w // 2
        
        kernels_flipped = kernel_bank[:, ::-1, ::-1]
        kernels_reshaped = kernels_flipped.reshape((num_kernels, 1, k_h, k_w))
        grids_reshaped = grids[jnp.newaxis, :, :, :] 
        
        # Convolve all grids simultaneously using grouped convolution
        rendered_phases = lax.conv_general_dilated(
            grids_reshaped,
            kernels_reshaped,
            window_strides=(1, 1),
            padding=[(pad_h, pad_h), (pad_w, pad_w)],
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'), 
            feature_group_count=num_kernels
        )
        
        return jnp.sum(rendered_phases, axis=1).squeeze()
    
    return render_core

def _get_fused_generator_renderer():
    render_core = _get_jax_renderer_core()
    
    def fused_op(key, fluxes, mags, kernel_bank, mosaic_size, mag_limit):
        n_stars = fluxes.shape[0]
        num_kernels = kernel_bank.shape[0]
        k1, k2, k3 = jax.random.split(key, 3)
        
        # 1. Generate Coordinates directly on GPU
        x = jax.random.uniform(k1, shape=(n_stars,), minval=0.0, maxval=float(mosaic_size))
        y = jax.random.uniform(k2, shape=(n_stars,), minval=0.0, maxval=float(mosaic_size))
        
        # 2. Assign a random PSF index from the library
        psf_indices = jax.random.randint(k3, shape=(n_stars,), minval=0, maxval=num_kernels)
        
        # 3. Render
        image = render_core(x, y, fluxes, psf_indices, kernel_bank, mosaic_size)
        
        # 4. Filter Mask
        catalog_mask = mags < mag_limit
        
        return image, x, y, psf_indices, catalog_mask

    return jax.jit(fused_op, static_argnums=(4,))

_FUSED_OP = None
_JAX_KEY = jax.random.PRNGKey(int(time.time())) if 'time' in globals() else jax.random.PRNGKey(42)

def render_generate_and_filter_gpu(fluxes, mags, kernel_bank, mosaic_size, mag_limit=27.0):
    global _FUSED_OP, _JAX_KEY
    if _FUSED_OP is None:
        _FUSED_OP = _get_fused_generator_renderer()
    
    _JAX_KEY, subkey = jax.random.split(_JAX_KEY)
    
    current_size = len(fluxes)
    chunk_size = 1_000_000 
    padded_size = int(math.ceil(current_size / chunk_size)) * chunk_size
    pad_width = padded_size - current_size
    
    if pad_width > 0:
        fluxes_padded = np.pad(fluxes, (0, pad_width), constant_values=0.0)
        mags_padded = np.pad(mags, (0, pad_width), constant_values=99.0)
    else:
        fluxes_padded, mags_padded = fluxes, mags

    fluxes_jax = jnp.array(fluxes_padded)
    mags_jax = jnp.array(mags_padded)
    kernels_jax = jnp.array(kernel_bank)
    
    image_jax, x_jax, y_jax, psf_jax, mask_jax = _FUSED_OP(subkey, fluxes_jax, mags_jax, kernels_jax, mosaic_size, mag_limit)
    
    image = np.array(image_jax)
    mask = np.array(mask_jax)
    
    valid_x = np.array(x_jax)[mask]
    valid_y = np.array(y_jax)[mask]
    valid_psf = np.array(psf_jax)[mask]
    valid_flux = np.array(fluxes_jax)[mask]
    valid_mags = np.array(mags_jax)[mask]
    
    return image, valid_x, valid_y, valid_psf, valid_flux, valid_mags
