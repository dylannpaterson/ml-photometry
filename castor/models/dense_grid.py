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

class DiffractionAwareFilter(nn.Module):
    """
    Mexican Hat (Laplacian of Gaussian) filter for diffraction-aware preprocessing.

    Attributes
    ----------
    conv : nn.Conv2d
        The convolutional layer with the LoG kernel.
    init_weight : torch.Tensor
        The initial analytical LoG kernel preserved for regularization.
    """
    def __init__(self, kernel_size=21, sigma=3.0):
        """
        Initialize DiffractionAwareFilter.

        Parameters
        ----------
        kernel_size : int, optional
            Size of the filter kernel, by default 21.
        sigma : float, optional
            Gaussian sigma for the Mexican Hat, by default 3.0.
        """
        super(DiffractionAwareFilter, self).__init__()

        # 1 in channel (raw flux), 1 out channel (filter response)
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)

        # Generate Analytical Mexican Hat Prior
        kernel = self._generate_mexican_hat(kernel_size, sigma)
        print("🛰️ DiffractionAwareFilter: Initialized with Analytical Mexican Hat")

        # Zero-mean and normalize (Acts as a high-pass/edge-like detector)
        kernel = kernel - kernel.mean()
        kernel = kernel / (torch.max(torch.abs(kernel)) + 1e-9)

        initial_weight = kernel.view(1, 1, kernel_size, kernel_size).float()
        self.conv.weight.data = initial_weight.clone()
        self.register_buffer("init_weight", initial_weight)
        self.conv.weight.requires_grad = True

    def _generate_mexican_hat(self, kernel_size, sigma):
        """Generates the 2D Mexican Hat (Laplacian of Gaussian) kernel."""
        device = self.conv.weight.device
        grid = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device, dtype=torch.float32)
        y, x = torch.meshgrid(grid, grid, indexing='ij')
        r2 = x**2 + y**2

        # LoG Formula
        pi = torch.tensor(np.pi, device=device, dtype=torch.float32)
        kernel = -(1.0 / (pi * sigma**4)) * (1.0 - r2 / (2 * sigma**2)) * torch.exp(-r2 / (2 * sigma**2))
        return kernel

    def get_regularization_loss(self):
        """
        Calculates the L2 distance from the initial LoG kernel.

        This prevents the filter from drifting into a random conv layer 
        during training.

        Returns
        -------
        torch.Tensor
            The L2 regularization loss value.
        """
        return torch.sum((self.conv.weight - self.init_weight) ** 2)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Raw input image tensor [B, 1, H, W].

        Returns
        -------
        torch.Tensor
            Concatenated original image and filtered response [B, 2, H, W].
        """
        # Concatenate the original raw image with the filtered response
        # Output shape: [Batch, 2, H, W]
        return torch.cat([x, self.conv(x)], dim=1)

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
    diffraction_filter : DiffractionAwareFilter
        Initial physics-aware filtering layer.
    initial, layer1, layer2, layer3, layer4 : nn.Module
        ResNet backbone components.
    top_layer, fpn3, fpn2, fpn1 : nn.Module
        FPN neck components.
    head : nn.Sequential
        Prediction head with CoordConv.
    log_task_vars : nn.Parameter
        Learnable parameters for homoscedastic task uncertainty.
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

        # 1. Physics Prior Filter
        self.diffraction_filter = DiffractionAwareFilter(kernel_size=21)

        # 2. Backbone: Full ResNet-34
        resnet = models.resnet34(weights=None)
        self.initial = nn.Sequential(
            # 3 Channels: Raw Image, Physics Filter response, Confidence Prior
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool, # Stride 4, Output 64x64
        )
        self.layer1 = resnet.layer1 # 64x64, 64ch
        self.layer2 = resnet.layer2 # 32x32, 128ch
        self.layer3 = resnet.layer3 # 16x16, 256ch
        self.layer4 = resnet.layer4 # 8x8, 512ch

        # 3. FPN Neck: Merge deep context back to the 64x64 prediction grid
        self.top_layer = nn.Conv2d(512, 128, kernel_size=1) # 8x8
        self.fpn3 = FPNBlock(256, 128, 128) # 16x16
        self.fpn2 = FPNBlock(128, 128, 128) # 32x32
        self.fpn1 = FPNBlock(64, 128, 128)  # 64x64

        # 4. Prediction Head with CoordConv for spatial awareness
        self.head = nn.Sequential(
            CoordConv(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_output_channels, kernel_size=1)
        )
        
        # NEW: Homoscedastic Task Uncertainty Parameters
        # Indices: 0:Prob, 1:Pos, 2:Flux, 3:BG, 4:Curvature, 5:Entropy
        self.log_task_vars = nn.Parameter(torch.zeros(6))

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
        # 1. Pass through trainable physics prior (Outputs 2 channels: Raw, Filtered)
        x_physics = self.diffraction_filter(x)
        
        # 2. Add the 3rd channel: Confidence Prior
        if prior is None:
            prior = torch.zeros_like(x)
        
        x_combined = torch.cat([x_physics, prior], dim=1)
        
        # 3. Feed 3-channel input into ResNet
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
            
            dx = torch.sigmoid(star_out[..., 1:2]) * self.cell_size
            dy = torch.sigmoid(star_out[..., 2:3]) * self.cell_size
            
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
                "background": bg,
                "log_task_vars": self.log_task_vars # Export learnable weights
            }

def compute_nll_loss(pred, target, log_var, beta=0.5, weights=None):
    """
    Calculates Beta-NLL with optional per-sample weighting.

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

    Returns
    -------
    torch.Tensor
        The calculated NLL loss.
    """
    precision = torch.exp(-log_var)
    # 1. Standard Gaussian NLL
    # Note: Adding a small constant or clamping log_var here helps prevent the "cheat code" 
    # where the model minimizes the penalty term instead of the error.
    loss = 0.5 * (precision * (pred - target)**2 + log_var)
    
    # 2. Scale by detached variance^beta
    var_detached = torch.exp(log_var.detach())
    beta_scale = var_detached ** beta
    weighted_loss = loss * beta_scale
    
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
    
    # Clamp diff for numerical stability before exp (max 10.0 -> ~22000 penalty)
    diff_clamped = torch.clamp(diff, max=10.0)
    
    # Formulation: exp(d) - d - 1
    # This is 0 when x == prior_mesh, explodes when x << prior_mesh.
    penalty_map = torch.exp(diff_clamped) - diff_clamped - 1.0
    
    return penalty_map.mean()

def compute_grid_loss(preds, targets, pca_std=None, lambda_prob=1.0, lambda_pos=1.0, lambda_flux=1.0, lambda_bg=1.0, lambda_curvature=1.0, focal_alpha=0.50, focal_gamma=2.0, stretch_scale=GLOBAL_STRETCH_SCALE, log_task_vars=None):
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
    log_task_vars : torch.Tensor, optional
        Task uncertainty weights, by default None.

    Returns
    -------
    tuple
        Tuple of (total_loss, prob_loss, pos_loss, flux_loss, bg_loss, ent_loss).
    """
    star_preds = preds["stars"]
    bg_preds = preds["background"]
    
    # 1. Unpack Target Grid
    B, H, W, C_target = targets.shape
    bg_targets = targets[..., -1:]
    star_targets_flat = targets[..., :-1]
    
    K = MAX_CAPACITY_PER_CELL
    star_targets = star_targets_flat.view(B, H, W, K, -1)
    
    # Object mask uses target p > 0 (Soft Labels)
    obj_mask = star_targets[..., 0] > 0.0
    p_target = star_targets[..., 0]
    
    # 2. Probability Loss (p) with Focal Loss
    p_pred_probs = torch.clamp(star_preds[..., 0], 1e-7, 1.0 - 1e-7)
    p_pred_logits = preds["p_logits"].squeeze(-1) 
    
    bce_loss = F.binary_cross_entropy_with_logits(p_pred_logits, p_target, reduction='none')
    p_t = p_pred_probs * p_target + (1 - p_pred_probs) * (1 - p_target)
    focal_weight = (1 - p_t) ** focal_gamma
    alpha_t = p_target * focal_alpha + (1 - p_target) * (1 - focal_alpha)
    
    raw_prob_loss = (alpha_t * focal_weight * bce_loss).mean()
    
    # 3. Regression Losses (Masked & Weighted by p_target)
    if obj_mask.sum() > 0:
        # Unpack predictions
        pos_pred = star_preds[..., 1:3][obj_mask]
        pos_target = star_targets[..., 1:3][obj_mask]
        log_var_pos = star_preds[..., 4:6][obj_mask]
        
        # Align flux loss with Arcsinh stretching
        asinh_flux_pred = preds["raw_asinh_flux"][obj_mask]
        asinh_flux_target = star_targets[..., 3:4][obj_mask] # Stretched by Trainer
        log_var_flux = star_preds[..., 6:7][obj_mask]
        
        reg_weights = p_target[obj_mask]

        # Calculate Weighted NLL Losses
        raw_pos_loss = compute_nll_loss(pos_pred, pos_target, log_var_pos, weights=reg_weights)
        raw_flux_loss = compute_nll_loss(asinh_flux_pred, asinh_flux_target, log_var_flux, weights=reg_weights)
    else:
        raw_pos_loss = torch.tensor(0.0, device=star_preds.device)
        raw_flux_loss = torch.tensor(0.0, device=star_preds.device)
        
    # 4. Background Losses
    raw_bg_loss = F.mse_loss(bg_preds, bg_targets, reduction='mean')
    raw_curvature_loss = compute_curvature_loss(bg_preds)
    raw_entropy_loss = compute_relative_entropy_loss(bg_preds)
    
    # --- TASK UNCERTAINTY WEIGHTING ---
    if log_task_vars is not None:
        # HARD FIX: To prevent the homoscedastic "cheat code", we must ensure log_task_vars 
        # do not become too negative when their respective raw losses are negative.
        # We clamp log_task_vars to a floor of -4.0 (variance ~0.018).
        safe_log_task_vars = torch.clamp(log_task_vars, min=-4.0)
        
        # Weights: 0:Prob, 1:Pos, 2:Flux, 3:BG, 4:Curvature, 5:Entropy
        prob_loss = torch.exp(-safe_log_task_vars[0]) * raw_prob_loss + safe_log_task_vars[0]
        pos_loss = torch.exp(-safe_log_task_vars[1]) * raw_pos_loss + safe_log_task_vars[1]
        flux_loss = torch.exp(-safe_log_task_vars[2]) * raw_flux_loss + safe_log_task_vars[2]
        bg_loss = torch.exp(-safe_log_task_vars[3]) * raw_bg_loss + safe_log_task_vars[3]
        curv_loss = torch.exp(-safe_log_task_vars[4]) * raw_curvature_loss + safe_log_task_vars[4]
        ent_loss = torch.exp(-safe_log_task_vars[5]) * raw_entropy_loss + safe_log_task_vars[5]
        
        # Final weighted sum (base scales set to 1.0)
        total_loss = (prob_loss + pos_loss + flux_loss + bg_loss + curv_loss + ent_loss)
    else:
        # Fallback
        total_loss = (raw_prob_loss + raw_pos_loss + raw_flux_loss + raw_bg_loss + raw_curvature_loss + raw_entropy_loss)
        prob_loss, pos_loss, flux_loss, bg_loss, curv_loss, ent_loss = raw_prob_loss, raw_pos_loss, raw_flux_loss, raw_bg_loss, raw_curvature_loss, raw_entropy_loss
                  
    return total_loss, prob_loss, pos_loss, flux_loss, bg_loss, ent_loss
