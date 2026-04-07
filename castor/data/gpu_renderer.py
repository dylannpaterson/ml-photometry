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
    """
    PCA-optimized JAX renderer with oversampled stamping and spatial weights.
    """
    def render_core(x, y, fluxes, mags, weights_lib, mean_psf_4x, eigen_psfs_4x, jitter_kernel_4x, mosaic_size, mag_limit):
        n_pca = eigen_psfs_4x.shape[0]
        O = 4 # Oversampling
        
        # 1. Spatially Correlated Weights (Simplified JAX version)
        # We'll use a few anchors for interpolation
        num_anchors = 5
        # Static anchors for JAX (could be passed in)
        anchors_x = jnp.array([0, 0, mosaic_size, mosaic_size, mosaic_size/2])
        anchors_y = jnp.array([0, mosaic_size, 0, mosaic_size, mosaic_size/2])
        
        # We'll just use the first 5 entries of weights_lib as anchors for this JITted op
        anchor_weights = weights_lib[:num_anchors]
        
        dist_sq = (x[:, jnp.newaxis] - anchors_x)**2 + (y[:, jnp.newaxis] - anchors_y)**2
        w_rbf = jnp.exp(-dist_sq / (2 * (mosaic_size/1.5)**2))
        star_weights = jnp.dot(w_rbf, anchor_weights) / (jnp.sum(w_rbf, axis=1, keepdims=True) + 1e-9)
        
        # 2. Apply Jitter to PSFs before binning
        def convolve_psf(psf, kernel):
            k_h, k_w = kernel.shape
            pad_h, pad_w = k_h // 2, k_w // 2
            return lax.conv_general_dilated(
                psf[jnp.newaxis, jnp.newaxis, :, :],
                kernel[::-1, ::-1][jnp.newaxis, jnp.newaxis, :, :],
                window_strides=(1, 1),
                padding=[(pad_h, pad_h), (pad_w, pad_w)],
                dimension_numbers=('NCHW', 'OIHW', 'NCHW')
            ).squeeze()

        mean_psf_jit = convolve_psf(mean_psf_4x, jitter_kernel_4x)
        
        # 3. Oversampled Stamping (via 16-channel convolution)
        x0 = jnp.floor(x).astype(jnp.int32)
        y0 = jnp.floor(y).astype(jnp.int32)
        dx_idx = jnp.clip(jnp.floor((x - x0) * O).astype(jnp.int32), 0, O-1)
        dy_idx = jnp.clip(jnp.floor((y - y0) * O).astype(jnp.int32), 0, O-1)
        
        # Create 16 grids for 16 sub-pixel shifts
        def get_grid(dyi, dxi):
            mask = (x0 >= 0) & (x0 < mosaic_size) & (y0 >= 0) & (y0 < mosaic_size) & (dx_idx == dxi) & (dy_idx == dyi)
            grid = jnp.zeros((mosaic_size, mosaic_size))
            return grid.at[y0, x0].add(jnp.where(mask, fluxes, 0.0))

        # We can use lax.scan or just a loop since it's only 16
        grids = []
        for dyi in range(O):
            for dxi in range(O):
                grids.append(get_grid(dyi, dxi))
        grids = jnp.stack(grids) # (16, H, W)

        # Pre-shifted PSFs (binned from jittered 4x)
        psfs_binned = []
        S = mean_psf_4x.shape[0] // O
        for dyi in range(O):
            for dxi in range(O):
                # Binning: slicing at offset
                psfs_binned.append(mean_psf_jit[dyi::O, dxi::O][:S, :S])
        psfs_binned = jnp.stack(psfs_binned) # (16, S, S)

        # 16-channel convolution
        k_h, k_w = S, S
        pad_h, pad_w = k_h // 2, k_w // 2
        
        final_image = lax.conv_general_dilated(
            grids[jnp.newaxis, :, :, :],
            psfs_binned[:, jnp.newaxis, ::-1, ::-1],
            window_strides=(1, 1),
            padding=[(pad_h, pad_h), (pad_w, pad_w)],
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
            feature_group_count=16
        ).sum(axis=1).squeeze()

        # 4. Correction Pass (Bright Stars)
        # For simplicity in JAX, we might just do the same 16-channel trick for each PCA component
        is_bright = mags < mag_limit
        bright_fluxes = jnp.where(is_bright, fluxes, 0.0)

        def add_eigen_contribution(image, i):
            eigen_psf_jit = convolve_psf(eigen_psfs_4x[i], jitter_kernel_4x)
            e_psfs_binned = []
            for dyi in range(O):
                for dxi in range(O):
                    e_psfs_binned.append(eigen_psf_jit[dyi::O, dxi::O][:S, :S])
            e_psfs_binned = jnp.stack(e_psfs_binned)
            
            w_f = bright_fluxes * star_weights[:, i]
            e_grids = []
            for dyi in range(O):
                for dxi in range(O):
                    mask = (x0 >= 0) & (x0 < mosaic_size) & (y0 >= 0) & (y0 < mosaic_size) & (dx_idx == dxi) & (dy_idx == dyi)
                    e_grid = jnp.zeros((mosaic_size, mosaic_size))
                    e_grids.append(e_grid.at[y0, x0].add(jnp.where(mask, w_f, 0.0)))
            e_grids = jnp.stack(e_grids)
            
            e_rendered = lax.conv_general_dilated(
                e_grids[jnp.newaxis, :, :, :],
                e_psfs_binned[:, jnp.newaxis, ::-1, ::-1],
                window_strides=(1, 1),
                padding=[(pad_h, pad_h), (pad_w, pad_w)],
                dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
                feature_group_count=16
            ).sum(axis=1).squeeze()
            return image + e_rendered, None

        final_image, _ = jax.lax.scan(add_eigen_contribution, final_image, jnp.arange(n_pca))
        
        return jnp.maximum(0, final_image), star_weights
    
    return render_core

def _get_fused_generator_renderer():
    render_core = _get_jax_renderer_core()
    
    def fused_op(key, fluxes, mags, weights_lib, mean_psf_4x, eigen_psfs_4x, jitter_params, mosaic_size, mag_limit):
        n_stars = fluxes.shape[0]
        k1, k2 = jax.random.split(key, 2)
        
        x = jax.random.uniform(k1, shape=(n_stars,), minval=0.0, maxval=float(mosaic_size))
        y = jax.random.uniform(k2, shape=(n_stars,), minval=0.0, maxval=float(mosaic_size))
        
        # Jitter kernel generation in JAX
        s_jit, q_jit, theta_jit = jitter_params
        O = 4
        S_high = eigen_psfs_4x.shape[1]
        k_half = S_high // 2
        gy, gx = jnp.meshgrid(jnp.arange(S_high) - k_half, jnp.arange(S_high) - k_half, indexing='ij')
        cos, sin = jnp.cos(theta_jit), jnp.sin(theta_jit)
        s_jit_high = s_jit * O
        gxp, gyp = gx * cos + gy * sin, -gx * sin + gy * cos
        jitter_kernel_4x = jnp.exp(-(gxp**2 / (2 * s_jit_high**2) + gyp**2 / (2 * (s_jit_high * q_jit)**2)))
        jitter_kernel_4x /= (jnp.sum(jitter_kernel_4x) + 1e-9)

        image, final_star_weights = render_core(x, y, fluxes, mags, weights_lib, mean_psf_4x, eigen_psfs_4x, jitter_kernel_4x, mosaic_size, mag_limit)
        catalog_mask = mags < mag_limit
        
        # Dummy psf_indices for compatibility
        psf_indices = jnp.zeros(n_stars, dtype=jnp.int32)
        
        return image, x, y, psf_indices, catalog_mask, final_star_weights

    return jax.jit(fused_op, static_argnums=(7, 8))

_FUSED_OP = None
_JAX_KEY = jax.random.PRNGKey(int(time.time())) if 'time' in globals() else jax.random.PRNGKey(42)

def render_generate_and_filter_gpu(fluxes, mags, weights_lib, mean_psf, eigen_psfs, mosaic_size, mag_limit=27.0):
    global _FUSED_OP, _JAX_KEY
    if _FUSED_OP is None:
        _FUSED_OP = _get_fused_generator_renderer()
    
    _JAX_KEY, subkey = jax.random.split(_JAX_KEY)
    
    # Generate random jitter params for this call
    # s_jit ~ 0.127, q_jit ~ 0.9, theta_jit ~ uniform
    s_jit = float(np.random.normal(0.127, 0.01))
    q_jit = float(np.random.uniform(0.8, 1.0))
    theta_jit = float(np.random.uniform(0, np.pi))
    jitter_params = jnp.array([s_jit, q_jit, theta_jit])

    # Ensure PSFs are oversampled (4x)
    O = 4
    if mean_psf.shape[0] != eigen_psfs.shape[1] * O:
        # Upsample if needed (should be handled by library loader ideally)
        from scipy.ndimage import zoom
        mean_psf_4x = zoom(mean_psf, O, order=3)
        eigen_psfs_4x = np.array([zoom(e, O, order=3) for e in eigen_psfs])
    else:
        mean_psf_4x = mean_psf
        eigen_psfs_4x = eigen_psfs

    current_size = len(fluxes)
    chunk_size = 10_000 # Reduced chunk size for complex JAX op
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
    mean_psf_jax = jnp.array(mean_psf_4x)
    eigen_psfs_jax = jnp.array(eigen_psfs_4x)
    
    image_jax, x_jax, y_jax, psf_jax, mask_jax, weights_jax_out = _FUSED_OP(
        subkey, fluxes_jax, mags_jax, weights_jax, mean_psf_jax, eigen_psfs_jax, jitter_params, mosaic_size, float(mag_limit)
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
