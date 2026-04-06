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
                master_data = torch.load(psf_library_path, map_location='cpu', weights_only=True)
                
                # Robust extraction of mean_psf
                if isinstance(master_data, dict):
                    m_psf = torch.from_numpy(master_data['mean_psf']).float()
                elif torch.is_tensor(master_data):
                    # Check for [Batch, N_PCA+1, H*W] or [N_PCA+1, H*W]
                    if master_data.dim() == 3:
                        m_psf = master_data[0, -1].float()
                    else:
                        m_psf = master_data[-1].float()
                elif isinstance(master_data, (list, tuple)):
                    # Assume tuple (eigen, weights, mean)
                    m_psf = torch.from_numpy(master_data[2]).float()
                else:
                    print(f"⚠️ Unknown master library format: {type(master_data)}")
                    m_psf = None
                
                if m_psf is not None:
                    # Reshape to 2D if needed (assume square)
                    if m_psf.dim() == 1:
                        s = int(m_psf.shape[0]**0.5)
                        m_psf = m_psf.view(s, s)
                    
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
        grid = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        y, x = torch.meshgrid(grid, grid, indexing='ij')
        r2 = x**2 + y**2

        # LoG Formula
        kernel = -(1.0 / (np.pi * sigma**4)) * (1.0 - r2 / (2 * sigma**2)) * torch.exp(-r2 / (2 * sigma**2))

        # 2. Add Spikes to the prior (3 lines at 0, 60, 120 degrees)
        angles = [0, np.pi/3, 2*np.pi/3]
        for angle in angles:
            # Distance from each pixel to the infinite line at this angle
            dist_to_line = torch.abs(x * np.sin(angle) - y * np.cos(torch.tensor(angle)))
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
    def __init__(self, K=MAX_CAPACITY_PER_CELL, shape_size=SHAPE_SIZE, cell_size=DEFAULT_CELL_SIZE):
        super(DenseGridModel, self).__init__()
        self.K = K
        # CHANGED: Dropping PCA shape weights in favor of Aleatoric Uncertainty Estimation
        # Output per slot: [p, dx, dy, m, log_var_x, log_var_y, log_var_m] = 7 channels
        self.cell_size = float(cell_size)
        self.num_output_channels = self.K * 7 + 1

        # 1. Physics Prior Filter
        self.diffraction_filter = DiffractionAwareFilter(kernel_size=21)

        # 2. Backbone: Full ResNet-34
        resnet = models.resnet34(weights=None)
        self.initial = nn.Sequential(
            # CHANGED: Now takes 2 channels (Raw Flux + Wavelet Response)
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

        out = self.head(p1)
        
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
        
        # NEW: Predict in log-space, but output raw physical flux
        raw_log_flux = star_out[..., 3:4]
        # CRITICAL: Clamp before exp() to prevent exploding gradients during early epochs
        raw_log_flux = torch.clamp(raw_log_flux, min=-10.0, max=22.0) 
        # FIX: Force float32 evaluation to prevent FP16 overflow on bright stars
        flux = torch.exp(raw_log_flux.float())
        
        # NEW: Uncertainty Estimates (Log-variance)
        # log_var_x (4), log_var_y (5), log_var_m (6)
        log_vars = star_out[..., 4:7]
        
        # Background residuals can be negative
        bg = bg_out.permute(0, 2, 3, 1)
        
        return {
            "stars": torch.cat([p, dx, dy, flux, log_vars], dim=-1),
            "p_logits": p_logits,
            "raw_log_flux": raw_log_flux,
            "log_vars": log_vars,
            "background": bg
        }

def compute_nll_loss(pred, target, log_var):
    """
    Calculates the Gaussian Negative Log-Likelihood.
    log_var = ln(sigma^2). We use exp(-log_var) which equals 1/sigma^2
    """
    precision = torch.exp(-log_var)
    # 0.5 * (precision * (pred - target)**2 + log_var)
    loss = 0.5 * (precision * (pred - target)**2 + log_var)
    return loss.mean()

def compute_grid_loss(preds, targets, pca_std=None, lambda_prob=5.0, lambda_pos=50.0, lambda_flux=5.0, lambda_bg=0.1, focal_alpha=0.75, focal_gamma=2.0, stretch_scale=GLOBAL_STRETCH_SCALE):
    """
    Refactored loss using Aleatoric Uncertainty Estimation (NLL).
    Drops shape reconstruction loss in favor of calibrated uncertainty.
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
    
    # 2. Probability Loss (p) with Focal Loss
    p_pred_probs = torch.clamp(star_preds[..., 0], 1e-7, 1.0 - 1e-7)
    p_pred_logits = preds["p_logits"].squeeze(-1) 
    p_target = star_targets[..., 0]
    
    bce_loss = F.binary_cross_entropy_with_logits(p_pred_logits, p_target, reduction='none')
    p_t = p_pred_probs * p_target + (1 - p_pred_probs) * (1 - p_target)
    focal_weight = (1 - p_t) ** focal_gamma
    alpha_t = p_target * focal_alpha + (1 - p_target) * (1 - focal_alpha)
    
    prob_loss = (alpha_t * focal_weight * bce_loss).mean()
    
    # 3. Regression Losses (Masked)
    if obj_mask.sum() > 0:
        # Unpack predictions
        pos_pred = star_preds[..., 1:3][obj_mask]
        pos_target = star_targets[..., 1:3][obj_mask]
        log_var_pos = star_preds[..., 4:6][obj_mask]
        
        log_flux_pred = preds["raw_log_flux"][obj_mask]
        flux_target = star_targets[..., 3:4][obj_mask]
        log_flux_target = torch.log(flux_target + 1e-6)
        log_var_flux = star_preds[..., 6:7][obj_mask]

        # Calculate NLL Losses
        pos_loss = compute_nll_loss(pos_pred, pos_target, log_var_pos)
        flux_loss = compute_nll_loss(log_flux_pred, log_flux_target, log_var_flux)
    else:
        pos_loss = torch.tensor(0.0, device=star_preds.device)
        flux_loss = torch.tensor(0.0, device=star_preds.device)
        
    # 4. Background Loss (Global MSE)
    bg_loss = F.mse_loss(bg_preds, bg_targets, reduction='mean')
        
    total_loss = (lambda_prob * prob_loss + 
                  lambda_pos * pos_loss + 
                  lambda_flux * flux_loss +
                  lambda_bg * bg_loss)
                  
                  
    return total_loss, prob_loss, pos_loss, flux_loss, bg_loss
