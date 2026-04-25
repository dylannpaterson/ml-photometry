import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import os
from castor.constants import DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL, SHAPE_SIZE, GLOBAL_STRETCH_SCALE

class CoordConv(nn.Module):
    """
    Adds normalized (x, y) coordinate channels to the input.

    Attributes
    ----------
    conv : nn.Conv2d
        The convolutional layer applied after concatenating coordinates.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        """
        Initialize CoordConv.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int, optional
            Size of the convolution kernel, by default 1.
        padding : int, optional
            Padding for the convolution, by default 0.
        """
        super(CoordConv, self).__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, C, H, W].

        Returns
        -------
        torch.Tensor
            Output tensor with coordinate channels concatenated and convolved.
        """
        batch_size, _, h, w = x.size()
        xx = torch.arange(w).view(1, 1, 1, w).expand(batch_size, 1, h, w).float() / (w - 1)
        yy = torch.arange(h).view(1, 1, h, 1).expand(batch_size, 1, h, w).float() / (h - 1)
        xx = xx.to(x.device) * 2 - 1
        yy = yy.to(x.device) * 2 - 1
        x = torch.cat([x, xx, yy], dim=1)
        return self.conv(x)

class FPNBlock(nn.Module):
    """
    Feature Pyramid Network (FPN) block for merging high and low res features.

    Attributes
    ----------
    lateral : nn.Conv2d
        Lateral convolution for high-resolution features.
    up : nn.Upsample
        Upsampling layer for low-resolution features.
    smooth : nn.Conv2d
        Smoothing convolution applied to the merged features.
    """
    def __init__(self, high_res_in, low_res_in, out_channels):
        """
        Initialize FPNBlock.

        Parameters
        ----------
        high_res_in : int
            Number of channels in high-resolution input.
        low_res_in : int
            Number of channels in low-resolution input.
        out_channels : int
            Number of channels in the output.
        """
        super(FPNBlock, self).__init__()
        self.lateral = nn.Conv2d(high_res_in, out_channels, kernel_size=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, high_res, low_res):
        """
        Forward pass.

        Parameters
        ----------
        high_res : torch.Tensor
            High-resolution feature map.
        low_res : torch.Tensor
            Low-resolution feature map from a deeper layer.

        Returns
        -------
        torch.Tensor
            The merged and smoothed feature map.
        """
        # low_res comes from deeper in the network, needs upsampling
        return self.smooth(self.lateral(high_res) + self.up(low_res))

class LearnedOpticalStem(nn.Module):
    """
    A 16-channel physics-agnostic front end. 
    Learns a dictionary of optical shapes (spikes, cores, blobs) directly from the data.
    """
    def __init__(self, kernel_size=21):
        super(LearnedOpticalStem, self).__init__()
        # 1 in channel, 16 out channels to capture diverse optical features
        self.conv = nn.Conv2d(1, 16, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        
        # FIX: Tamed Initialization
        # Instead of kaiming_normal_ (which creates loud random noise), 
        # initialize near-zero so the raw image channel dominates early training.
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.01)

    def forward(self, x):
        # Concatenate the 1 raw image channel with the 16 learned feature channels
        # Output shape: [Batch, 17, H, W]
        return torch.cat([x, self.conv(x)], dim=1)

def convert_bn_to_gn(module, num_groups=8):
    """
    Recursively replaces BatchNorm2d with GroupNorm.
    
    Using fewer groups (8 instead of 32) helps preserve absolute intensity
    information which is critical for photometric flux recovery.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            # Ensure num_channels is divisible by groups. 8 is a good balance.
            groups = num_groups if num_channels % num_groups == 0 else num_channels
            setattr(module, name, nn.GroupNorm(groups, num_channels))
        else:
            convert_bn_to_gn(child, num_groups)

def swap_relu_with_gelu(module):
    """
    Recursively replaces all ReLU layers with GELU for better gradient flow
    through negative noise troughs.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.GELU())
        else:
            swap_relu_with_gelu(child)

class DenseGridModel(nn.Module):
    """
    The main Castor neural network model using a dense grid output.

    Attributes
    ----------
    K : int
        Maximum number of stars per grid cell.
    cell_size : float
        The size of each grid cell in pixels.
    stretch_scale : float
        The scale used for arcsinh flux stretching.
    num_output_channels : int
        Total number of output channels in the prediction head.
    optical_stem : LearnedOpticalStem
        Learned optical feature extraction stem.
    initial, layer1, layer2, layer3, layer4 : nn.Module
        ResNet backbone components.
    top_layer, fpn3, fpn2, fpn1 : nn.Module
        FPN neck components.
    head : nn.Sequential
        Prediction head with CoordConv.
    """
    def __init__(self, K=MAX_CAPACITY_PER_CELL, shape_size=SHAPE_SIZE, cell_size=DEFAULT_CELL_SIZE, stretch_scale=GLOBAL_STRETCH_SCALE):
        """
        Initialize DenseGridModel.

        Parameters
        ----------
        K : int, optional
            Stars per cell, by default MAX_CAPACITY_PER_CELL.
        shape_size : int, optional
            Size of input image (not used directly in init), by default SHAPE_SIZE.
            cell_size : int, optional
            Size of each cell, by default DEFAULT_CELL_SIZE.
        stretch_scale : float, optional
            Flux stretch scale, by default GLOBAL_STRETCH_SCALE.
        """
        super(DenseGridModel, self).__init__()
        self.K = K
        # Output per slot: [p, dx, dy, asinh_flux, log_var_x, log_var_y, log_var_f] = 7 channels
        self.cell_size = float(cell_size)
        self.stretch_scale = float(stretch_scale)
        self.num_output_channels = self.K * 7 + 1

        # 1. Learned Optical Dictionary (Front Door)
        self.optical_stem = LearnedOpticalStem(kernel_size=21)

        # 2. Backbone: ResNet-34 with GroupNorm and GELU
        resnet = models.resnet34(weights=None)
        
        # 🚀 BN -> GN Conversion: Preserve global scale
        convert_bn_to_gn(resnet, num_groups=8)
        
        # 🚀 ReLU -> GELU Conversion: Maintain gradients for negative noise troughs
        swap_relu_with_gelu(resnet)

        self.initial = nn.Sequential(
            # 18 Channels: 1 Raw + 16 Optical Stem + 1 Prior
            nn.Conv2d(18, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1, # This is now GroupNorm
            nn.GELU(),
            # Replace MaxPool with AvgPool to mathematically conserve flux during downsampling
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = resnet.layer1 
        self.layer2 = resnet.layer2 
        self.layer3 = resnet.layer3 
        self.layer4 = resnet.layer4 

        # 3. FPN Neck: Merge deep context back to the 64x64 prediction grid
        self.top_layer = nn.Conv2d(512, 128, kernel_size=1) # 8x8
        self.fpn3 = FPNBlock(256, 128, 128) # 16x16
        self.fpn2 = FPNBlock(128, 128, 128) # 32x32
        self.fpn1 = FPNBlock(64, 128, 128)  # 64x64

        # 4. Prediction Head with CoordConv for spatial awareness
        self.head = nn.Sequential(
            CoordConv(128, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, self.num_output_channels, kernel_size=1)
        )

    def forward(self, x, prior=None):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Raw input image of shape [B, 1, H, W].
        prior : torch.Tensor, optional
            Confidence map of shape [B, 1, H, W], values in [0, 1].
            If None, a zero prior is used.

        Returns
        -------
        dict
            Dictionary containing predicted stars, background, and uncertainty.
        """
        # Bottom-up
        # 1. Pass through learned optical stem (Outputs 17 channels: 1 Raw, 16 Features)
        x_stem = self.optical_stem(x)
        
        # 2. Add the 18th channel: Confidence Prior
        if prior is None:
            prior = torch.zeros_like(x)
        
        x_combined = torch.cat([x_stem, prior], dim=1)
        
        # 3. Feed 18-channel input into ResNet
        c0 = self.initial(x_combined)
        c1 = self.layer1(c0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        # Top-down (FPN)
        p4 = self.top_layer(c4)
        p3 = self.fpn3(c3, p4)
        p2 = self.fpn2(c2, p3)
        p1 = self.fpn1(c1, p2) # Final 64x64 features

        # --- THE FP32 HEAD BYPASS ---
        # Disable autocast for the sensitive final convolutions to prevent FP16 gradient overflow
        with torch.autocast(device_type=p1.device.type, enabled=False):
            p1_fp32 = p1.float()
            out = self.head(p1_fp32)
            
            B, C, H, W = out.shape
            star_out = out[:, :-1, :, :]
            bg_out = out[:, -1:, :, :]
            
            star_out = star_out.view(B, self.K, 7, H, W)
            star_out = star_out.permute(0, 3, 4, 1, 2)
            
            # --- THE LOGIT BYPASS ---
            p_logits = star_out[..., 0:1] # Keep raw logits
            p = torch.sigmoid(p_logits)   # Standard probability for inference
            # ------------------------
            
            # 🚀 RELAXED SIGMOID: Expand range by 20% to keep gradients steep at boundaries
            # and prevent "pixel-locking" centroids.
            dx = (torch.sigmoid(star_out[..., 1:2]) * 1.2 - 0.1) * self.cell_size
            dy = (torch.sigmoid(star_out[..., 2:3]) * 1.2 - 0.1) * self.cell_size
            
            # FIX: Use Arcsinh for flux instead of Log.
            # Predicted value is 'raw_asinh_flux'
            raw_asinh_flux = star_out[..., 3:4]
            # Safety bound for extreme brights (asinh(1e12/10) ~ 26)
            raw_asinh_flux = torch.clamp(raw_asinh_flux, min=-2.0, max=30.0) 
            flux = torch.sinh(raw_asinh_flux) * float(self.stretch_scale)
            
            # Uncertainty Estimates (Log-variance)
            # Use Softplus + epsilon to enforce a hard floor on predicted uncertainty
            # and prevent Gaussian NLL from collapsing to negative infinity.
            log_vars_raw = star_out[..., 4:7]
            # 1e-4 floor as suggested by user (log(1e-4) ~= -9.21)
            vars = F.softplus(log_vars_raw) + 1e-4
            log_vars = torch.log(vars)
            
            # Limit the clamp to max=20.0 to ensure numerical stability in FP16 AMP.
            log_vars = torch.clamp(log_vars, max=20.0) 
            
            # Background residuals can be negative
            bg = bg_out.permute(0, 2, 3, 1)
            
            return {
                "stars": torch.cat([p, dx, dy, flux, log_vars], dim=-1),
                "p_logits": p_logits,
                "raw_asinh_flux": raw_asinh_flux,
                "log_vars": log_vars,
                "background": bg
            }

def compute_nll_loss(pred, target, log_var, beta=0.5, weights=None, current_epoch=None):
    """
    Calculates Beta-NLL with optional per-sample weighting and epoch-based warm-up.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values.
    target : torch.Tensor
        Target values.
    log_var : torch.Tensor
        Log-variance prediction for aleatoric uncertainty.
    beta : float, optional
        Power for variance scaling (Beta-NLL), by default 0.5.
    weights : torch.Tensor, optional
        Per-sample weights for the loss, by default None.
    current_epoch : int, optional
        Current training epoch for warm-up logic, by default None.

    Returns
    -------
    torch.Tensor
        The calculated NLL loss.
    """
    # 1. Warm-up: Use standard MSE for the first 10 epochs
    if current_epoch is not None and current_epoch < 10:
        loss = 0.5 * ((pred - target)**2)
    else:
        precision = torch.exp(-log_var)
        # 2. Shift NLL Floor: Offset by the minimum possible log_var (ln(1e-4) ~= -9.21)
        # to ensure the loss remains positive and well-behaved.
        nll_floor = -9.21034
        loss = 0.5 * (precision * (pred - target)**2 + (log_var - nll_floor))
    
    # 3. Scale by detached variance^beta (Beta-NLL)
    # We only do this after warm-up
    if current_epoch is None or current_epoch >= 10:
        var_detached = torch.exp(log_var.detach())
        beta_scale = var_detached ** beta
        weighted_loss = loss * beta_scale
    else:
        weighted_loss = loss
    
    if weights is not None:
        # Ensure weights match coordinate dimensions
        if weighted_loss.dim() > weights.dim():
            weights = weights.unsqueeze(-1)
        # Weighted mean
        return (weighted_loss * weights).sum() / (weights.sum() + 1e-9)
    
    return weighted_loss.mean()

def compute_curvature_loss(bg_pred):
    """
    Calculates the L2 penalty on the spatial curvature of the background.

    Uses an interior mask to avoid boundary artifacts from zero-padding.

    Parameters
    ----------
    bg_pred : torch.Tensor
        Predicted background map of shape [B, H, W, 1].

    Returns
    -------
    torch.Tensor
        The curvature loss value.
    """
    # 1. Prepare for Conv2D (B, 1, H, W)
    x = bg_pred.permute(0, 3, 1, 2)
    
    # 2. Define Discrete Laplacian Kernel
    laplacian = torch.tensor([[[[ 0.,  1.,  0.],
                                [ 1., -4.,  1.],
                                [ 0.,  1.,  0.]]]], device=x.device, dtype=x.dtype)
    
    # 3. Convolve to find curvature
    curvature = F.conv2d(x, laplacian, padding=1)
    
    # 4. Interior Masking: Ignore the 1-pixel boundary
    interior = curvature[:, :, 1:-1, 1:-1]
    
    return torch.mean(interior ** 2)

def compute_relative_entropy_loss(bg_pred):
    """
    Calculates the Relative Entropy of the background map against a mesh prior.
    
    Uses a stable residual-space asymmetric penalty: exp(d) - d - 1.

    Parameters
    ----------
    bg_pred : torch.Tensor
        Predicted background map of shape [B, H, W, 1].

    Returns
    -------
    torch.Tensor
        The calculated entropy/moat-wall loss.
    """
    B, H, W, _ = bg_pred.shape
    # 1. Background Mesh (The "Astronomer's Low-Pass")
    x = bg_pred.permute(0, 3, 1, 2)
    
    mesh_size = 8
    kh, kw = H // mesh_size, W // mesh_size
    
    with torch.no_grad():
        # Calculate local medians to be robust to stars and diffraction spikes
        patches = x.unfold(2, kh, kh).unfold(3, kw, kw) # (B, 1, mesh_size, mesh_size, kh, kw)
        local_medians = patches.contiguous().view(B, 1, mesh_size, mesh_size, -1).median(dim=-1)[0]
        
        # Upsample back to full resolution using bilinear interpolation
        prior_mesh = F.interpolate(local_medians, size=(H, W), mode='bilinear', align_corners=False)

    # 2. Asymmetric "Moat-Wall" Penalty
    # d > 0 when bg_pred < prior_mesh (the moat)
    diff = (prior_mesh - x)
    
    # 🚀 STABILITY FIX: Clamp diff on BOTH sides. 
    # Max=10.0 prevents exp() explosion for "moats".
    # Min=-10.0 prevents linear explosion for "walls" (extremely high bg predictions).
    diff_clamped = torch.clamp(diff, min=-10.0, max=10.0)
    
    # Formulation: exp(d) - d - 1
    # This is 0 when x == prior_mesh, explodes when x << prior_mesh.
    penalty_map = torch.exp(diff_clamped) - diff_clamped - 1.0

    return penalty_map.mean()

def compute_dreg_loss(optical_stem):
    """
    L2 penalty on the stem's convolution weights to keep them smooth.
    """
    if optical_stem is None:
        return torch.tensor(0.0)
    return torch.sum(optical_stem.conv.weight ** 2)

def compute_grid_loss(preds, targets, pca_std=None, lambda_prob=1.0, lambda_pos=1.0, lambda_flux=1.0, lambda_bg=1.0, lambda_curvature=1.0, focal_alpha=0.50, focal_gamma=2.0, stretch_scale=GLOBAL_STRETCH_SCALE, current_epoch=None, optical_stem_reference=None, **kwargs):
    """
    Refactored loss using Aleatoric Uncertainty Estimation (NLL).

    Parameters
    ----------
    preds : dict
        Predictions from the model.
    targets : torch.Tensor
        Ground truth grid.
    pca_std : torch.Tensor, optional
        Standard deviation for PCA (unused), by default None.
    lambda_prob, lambda_pos, lambda_flux, lambda_bg, lambda_curvature : float, optional
        Legacy weight parameters, by default 1.0.
    focal_alpha, focal_gamma : float, optional
        Focal loss parameters, by default 0.50 and 2.0.
    stretch_scale : float, optional
        Arcsinh stretch scale, by default GLOBAL_STRETCH_SCALE.
    current_epoch : int, optional
        Current training epoch, by default None.
    optical_stem_reference : nn.Module, optional
        The optical stem to regularize, by default None.
    **kwargs : dict
        Additional parameters like lambda_entropy.

    Returns
    -------
    tuple
        Tuple of (total_loss, prob_loss, pos_loss, flux_loss, bg_loss, curv_loss, ent_loss).
    """
    star_preds = preds["stars"]
    bg_preds = preds["background"]
    
    # 1. Unpack Target Grid
    B, H, W, C_target = targets.shape
    bg_targets = targets[..., -1:]
    star_targets_flat = targets[..., :-1]
    
    K = MAX_CAPACITY_PER_CELL
    star_targets = star_targets_flat.view(B, H, W, K, -1)
    
    # Probability target for Detection Head
    p_target = star_targets[..., 0]
    
    # 2. Probability Loss (p) with Focal Loss
    # We keep computing this for ALL stars so the network learns to suppress faint ones.
    p_pred_probs = torch.clamp(star_preds[..., 0], 1e-7, 1.0 - 1e-7)
    p_pred_logits = preds["p_logits"].squeeze(-1) 
    
    bce_loss = F.binary_cross_entropy_with_logits(p_pred_logits, p_target, reduction='none')
    p_t = p_pred_probs * p_target + (1 - p_pred_probs) * (1 - p_target)
    focal_weight = (1 - p_t) ** focal_gamma
    alpha_t = p_target * focal_alpha + (1 - p_target) * (1 - focal_alpha)
    
    raw_prob_loss = (alpha_t * focal_weight * bce_loss).mean()
    
    # --- THE FIX: STRICT REGRESSION MASK ---
    # Only regress positions/fluxes for stars that are physically detectable.
    # We use SNR >= 3.0 as the cutoff for localization training.
    # The target SNR is at index 4 of the [p, dx, dy, flux, snr] layout.
    snr_target_full = star_targets[..., 4]
    regression_mask = (p_target > 0.0) & (snr_target_full >= 3.0)
    
    # 3. Regression Losses (Masked & Weighted)
    if regression_mask.sum() > 0:
        # Unpack predictions using the STRICT mask
        pos_pred = star_preds[..., 1:3][regression_mask]
        pos_target = star_targets[..., 1:3][regression_mask]
        log_var_pos = star_preds[..., 4:6][regression_mask]
        
        # Align flux loss with Arcsinh stretching
        asinh_flux_pred = preds["raw_asinh_flux"][regression_mask]
        asinh_flux_target = star_targets[..., 3:4][regression_mask] # Stretched by Trainer
        log_var_flux = star_preds[..., 6:7][regression_mask]
        
        # Extract SNR for the valid stars
        snr_target = snr_target_full[regression_mask]
        
        # NEW: Linear SNR weighting (minimum clamped to 3.0 to match regression mask)
        # Brightest stars (SNR 1500) now carry 500x more weight than faint ones,
        # forcing precise recovery of high-flux sources.
        snr_weight = torch.clamp(snr_target, min=3.0)
        
        # Final weights: Probability label * SNR Weight
        reg_weights = p_target[regression_mask] * snr_weight

        # Calculate Weighted NLL Losses
        raw_pos_loss = compute_nll_loss(pos_pred, pos_target, log_var_pos, weights=reg_weights, current_epoch=current_epoch)
        raw_flux_loss = compute_nll_loss(asinh_flux_pred, asinh_flux_target, log_var_flux, weights=reg_weights, current_epoch=current_epoch)
    else:
        raw_pos_loss = torch.tensor(0.0, device=star_preds.device)
        raw_flux_loss = torch.tensor(0.0, device=star_preds.device)
        
    # 4. Background Losses
    raw_bg_loss = F.mse_loss(bg_preds, bg_targets, reduction='mean')
    raw_curvature_loss = compute_curvature_loss(bg_preds)
    raw_entropy_loss = compute_relative_entropy_loss(bg_preds)
    
    # 5. Diffraction Regularization
    raw_dreg_loss = compute_dreg_loss(optical_stem_reference)
    if optical_stem_reference is not None:
        raw_dreg_loss = raw_dreg_loss.to(star_preds.device)
    
    # --- STATIC WEIGHTING ---
    prob_loss = lambda_prob * raw_prob_loss
    pos_loss = lambda_pos * raw_pos_loss
    flux_loss = lambda_flux * raw_flux_loss
    bg_loss = lambda_bg * raw_bg_loss
    curv_loss = lambda_curvature * raw_curvature_loss
    
    # Pull lambda_entropy from config (defaults to 1.0 if not found)
    lambda_entropy = kwargs.get("lambda_entropy", 1.0)
    ent_loss = lambda_entropy * raw_entropy_loss
    
    lambda_diffraction_reg = kwargs.get("lambda_diffraction_reg", 10.0)
    dreg_loss = lambda_diffraction_reg * raw_dreg_loss
    
    total_loss = (prob_loss + pos_loss + flux_loss + bg_loss + curv_loss + ent_loss + dreg_loss)
                  
    return total_loss, prob_loss, pos_loss, flux_loss, bg_loss, curv_loss, ent_loss
