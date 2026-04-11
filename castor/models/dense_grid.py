import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import os
from castor.constants import DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL, SHAPE_SIZE, GLOBAL_STRETCH_SCALE, N_PCA_COMPONENTS

class CoordConv(nn.Module):
    """Adds normalized (x, y) coordinate channels to the input."""
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super(CoordConv, self).__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        batch_size, _, h, w = x.size()
        xx = torch.arange(w).view(1, 1, 1, w).expand(batch_size, 1, h, w).float() / (w - 1)
        yy = torch.arange(h).view(1, 1, h, 1).expand(batch_size, 1, h, w).float() / (h - 1)
        xx = xx.to(x.device) * 2 - 1
        yy = yy.to(x.device) * 2 - 1
        x = torch.cat([x, xx, yy], dim=1)
        return self.conv(x)

class FPNBlock(nn.Module):
    def __init__(self, high_res_in, low_res_in, out_channels):
        super(FPNBlock, self).__init__()
        self.lateral = nn.Conv2d(high_res_in, out_channels, kernel_size=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, high_res, low_res):
        # low_res comes from deeper in the network, needs upsampling
        return self.smooth(self.lateral(high_res) + self.up(low_res))

class DiffractionAwareFilter(nn.Module):
    def __init__(self, kernel_size=21, sigma=3.0, psf_library_path="master_psf_library.pt"):
        super(DiffractionAwareFilter, self).__init__()

        # 1 in channel (raw flux), 1 out channel (filter response)
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)

        # Priority 1: Master PSF Library (The survey-average blurred PSF)
        # Priority 2: Analytical Mexican Hat (Fallback)
        
        kernel = None
        
        # Try Priority 1: Library Mean
        if os.path.exists(psf_library_path):
            try:
                # Set weights_only=False to allow loading NumPy objects in the master library
                master_data = torch.load(psf_library_path, map_location='cpu', weights_only=False)
                
                # Robust extraction of mean_psf
                if isinstance(master_data, dict):
                    m_psf = torch.from_numpy(master_data['mean_psf']).float()
                elif torch.is_tensor(master_data):
                    # Squeeze batch and channel dims -> e.g., [516, 516]
                    master_data = master_data.squeeze()
                    
                    if master_data.dim() == 2:
                        m_psf = master_data.float()
                    elif master_data.dim() == 1:
                        # Old flattened format
                        s = int(master_data.shape[0]**0.5)
                        m_psf = master_data.view(s, s).float()
                    else:
                        # Fallback for old [N_PCA+1, H*W] format
                        m_psf = master_data[-1].float()
                        if m_psf.dim() == 1:
                            s = int(m_psf.shape[0]**0.5)
                            m_psf = m_psf.view(s, s)
                elif isinstance(master_data, (list, tuple)):
                    # Assume tuple (eigen, weights, mean)
                    m_psf = torch.from_numpy(master_data[2]).float()
                else:
                    print(f"⚠️ Unknown master library format: {type(master_data)}")
                    m_psf = None
                
                if m_psf is not None:
                    # NEW: Bin down to 1x if oversampled
                    from castor.constants import SHAPE_SIZE
                    S_full = m_psf.shape[0]
                    if S_full > SHAPE_SIZE:
                        O = S_full // SHAPE_SIZE
                        m_psf = m_psf.reshape(SHAPE_SIZE, O, SHAPE_SIZE, O).mean(dim=(1, 3))
                        m_psf = m_psf / (m_psf.sum() + 1e-9)

                    kernel = self._fit_to_kernel_size(m_psf, kernel_size)
                    print(f"🛰️ DiffractionAwareFilter: Initialized with Master Library Mean ({psf_library_path})")
            except Exception as e:
                print(f"⚠️ Failed to load master library for prior: {e}")


        # Priority 3: Analytical Fallback
        if kernel is None:
            kernel = self._generate_analytical_prior(kernel_size, sigma)
            print("🛰️ DiffractionAwareFilter: Initialized with Analytical Mexican Hat")

        # Zero-mean and normalize (Acts as a high-pass/edge-like detector)
        kernel = kernel - kernel.mean()
        kernel = kernel / torch.max(torch.abs(kernel))

        initial_weight = kernel.view(1, 1, kernel_size, kernel_size).float()
        self.conv.weight.data = initial_weight.clone()
        self.register_buffer("init_weight", initial_weight)
        self.conv.weight.requires_grad = True

    def _fit_to_kernel_size(self, psf, kernel_size):
        """ Crops or pads a PSF to match the target kernel size. """
        curr_s = psf.shape[0]
        if curr_s > kernel_size:
            start = (curr_s - kernel_size) // 2
            return psf[start:start+kernel_size, start:start+kernel_size]
        elif curr_s < kernel_size:
            pad = (kernel_size - curr_s) // 2
            return F.pad(psf, [pad, pad, pad, pad])
        return psf

    def _generate_analytical_prior(self, kernel_size, sigma):
        # 1. Generate the 2D Mexican Hat (Laplacian of Gaussian) kernel
        device = self.conv.weight.device
        grid = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device, dtype=torch.float32)
        y, x = torch.meshgrid(grid, grid, indexing='ij')
        r2 = x**2 + y**2

        # LoG Formula
        pi = torch.tensor(np.pi, device=device, dtype=torch.float32)
        kernel = -(1.0 / (pi * sigma**4)) * (1.0 - r2 / (2 * sigma**2)) * torch.exp(-r2 / (2 * sigma**2))

        # 2. Add Spikes to the prior (3 lines at 0, 60, 120 degrees)
        angles = [0.0, np.pi/3.0, 2.0*np.pi/3.0]
        for angle in angles:
            angle_t = torch.tensor(angle, device=device, dtype=torch.float32)
            # Distance from each pixel to the infinite line at this angle
            dist_to_line = torch.abs(x * torch.sin(angle_t) - y * torch.cos(angle_t))
            # Add a thin exponential spike
            kernel += torch.exp(-dist_to_line / 0.5) * 0.05

        # Normalize the kernel so it doesn't blow up the activations
        kernel = kernel - kernel.mean()
        kernel = kernel / torch.max(torch.abs(kernel))
        return kernel
    def get_regularization_loss(self):
        """
        Calculates the L2 distance from the initial LoG kernel.
        This prevents the filter from drifting into a random conv layer.
        """
        return torch.sum((self.conv.weight - self.init_weight) ** 2)

    def forward(self, x):
        # Concatenate the original raw image with the filtered response
        # Output shape: [Batch, 2, H, W]
        return torch.cat([x, self.conv(x)], dim=1)

class DenseGridModel(nn.Module):
    def __init__(self, K=MAX_CAPACITY_PER_CELL, shape_size=SHAPE_SIZE, cell_size=DEFAULT_CELL_SIZE, stretch_scale=GLOBAL_STRETCH_SCALE):
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
            # Takes 2 channels (Raw Flux + Wavelet Response)
            nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),
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
        # Indices: 0:Prob, 1:Pos, 2:Flux, 3:BG, 4:Curvature
        self.log_task_vars = nn.Parameter(torch.zeros(5))

    def forward(self, x):
        # Bottom-up
        # 1. Pass through trainable physics prior (Outputs 2 channels)
        x_physics = self.diffraction_filter(x)
        
        # 2. Feed 2-channel input into ResNet
        c0 = self.initial(x_physics)
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
            log_vars = star_out[..., 4:7]
            # Limit the clamp to -10.0 (sigma ~ 0.006) to ensure numerical stability in FP16 AMP.
            log_vars = torch.clamp(log_vars, min=-10.0, max=20.0) 
            
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
    """
    precision = torch.exp(-log_var)
    # 1. Standard Gaussian NLL
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
    bg_pred shape: (Batch, H, W, 1)
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

def compute_grid_loss(preds, targets, pca_std=None, lambda_prob=1.0, lambda_pos=1.0, lambda_flux=1.0, lambda_bg=1.0, lambda_curvature=1.0, focal_alpha=0.50, focal_gamma=2.0, stretch_scale=GLOBAL_STRETCH_SCALE, log_task_vars=None):
    """
    Refactored loss using Aleatoric Uncertainty Estimation (NLL).
    Supports Homoscedastic Task Uncertainty Weighting and Curvature Regularization.
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
        flux_target = star_targets[..., 3:4][obj_mask]
        asinh_flux_target = torch.asinh(flux_target / float(stretch_scale))
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
    
    # --- TASK UNCERTAINTY WEIGHTING ---
    if log_task_vars is not None:
        # Weights: 0:Prob, 1:Pos, 2:Flux, 3:BG, 4:Curvature
        prob_loss = torch.exp(-log_task_vars[0]) * raw_prob_loss + log_task_vars[0]
        pos_loss = torch.exp(-log_task_vars[1]) * raw_pos_loss + log_task_vars[1]
        flux_loss = torch.exp(-log_task_vars[2]) * raw_flux_loss + log_task_vars[2]
        bg_loss = torch.exp(-log_task_vars[3]) * raw_bg_loss + log_task_vars[3]
        curv_loss = torch.exp(-log_task_vars[4]) * raw_curvature_loss + log_task_vars[4]
        
        # Final weighted sum (base scales set to 1.0)
        total_loss = (prob_loss + pos_loss + flux_loss + bg_loss + curv_loss)
    else:
        # Fallback
        total_loss = (raw_prob_loss + raw_pos_loss + raw_flux_loss + raw_bg_loss + raw_curvature_loss)
        prob_loss, pos_loss, flux_loss, bg_loss = raw_prob_loss, raw_pos_loss, raw_flux_loss, raw_bg_loss
                  
    return total_loss, prob_loss, pos_loss, flux_loss, bg_loss
