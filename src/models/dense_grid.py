import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

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
    def __init__(self, kernel_size=21, sigma=3.0):
        super(DiffractionAwareFilter, self).__init__()
        
        # 1 in channel (raw flux), 1 out channel (filter response)
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        
        # 1. Generate the 2D Mexican Hat (Laplacian of Gaussian) kernel
        grid = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        y, x = torch.meshgrid(grid, grid, indexing='ij')
        r2 = x**2 + y**2
        
        # LoG Formula
        kernel = -(1.0 / (np.pi * sigma**4)) * (1.0 - r2 / (2 * sigma**2)) * torch.exp(-r2 / (2 * sigma**2))
        
        # Normalize the kernel so it doesn't blow up the activations
        kernel = kernel - kernel.mean()
        kernel = kernel / torch.max(torch.abs(kernel))
        
        # 2. Assign the mathematical prior to the Conv2d weights
        # Reshape to match PyTorch weight format: [out_channels, in_channels, H, W]
        self.conv.weight.data = kernel.view(1, 1, kernel_size, kernel_size).float()
        
        # 3. CRITICAL: Allow the network to backpropagate and warp this shape 
        # to match the true Roman PSF diffraction spikes!
        self.conv.weight.requires_grad = True

    def forward(self, x):
        # Concatenate the original raw image with the filtered response
        # Output shape: [Batch, 2, H, W]
        return torch.cat([x, self.conv(x)], dim=1)

class DenseGridModel(nn.Module):
    def __init__(self, K=3, shape_size=9, cell_size=4):
        super(DenseGridModel, self).__init__()
        self.K = K
        self.S2 = shape_size * shape_size
        self.cell_size = float(cell_size)
        self.num_output_channels = self.K * (5 + self.S2) + 1

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
        
        star_out = star_out.view(B, self.K, 5 + self.S2, H, W)
        star_out = star_out.permute(0, 3, 4, 1, 2)
        
        p = torch.sigmoid(star_out[..., 0:1])
        dx = torch.sigmoid(star_out[..., 1:2]) * self.cell_size
        dy = torch.sigmoid(star_out[..., 2:3]) * self.cell_size
        
        # NEW: Predict in log-space, but output raw physical flux
        raw_log_flux = star_out[..., 3:4]
        # CRITICAL: Clamp before exp() to prevent exploding gradients during early epochs
        raw_log_flux = torch.clamp(raw_log_flux, min=-10.0, max=22.0) 
        flux = torch.exp(raw_log_flux)
        
        c = torch.sigmoid(star_out[..., 4:5])
        shape_logits = star_out[..., 5:]
        shape = F.softmax(shape_logits, dim=-1)
        
        # Background residuals can be negative
        bg = bg_out.permute(0, 2, 3, 1)
        
        return {
            "stars": torch.cat([p, dx, dy, flux, c, shape], dim=-1),
            "raw_log_flux": raw_log_flux, # Logit Bypass
            "background": bg
        }

def compute_grid_loss(preds, targets, lambda_prob=5.0, lambda_pos=50.0, lambda_flux=5.0, lambda_comp=1.0, lambda_shape=1.0, lambda_bg=0.1, focal_alpha=0.75, focal_gamma=2.0, stretch_scale=10.0):
    """
    Standard Generative Loss without TV regularization (optimized for speed).
    Maintains positional weighting and faint-star boost.
    Supports flattened target tensors and independent flux/completeness weights.
    """
    star_preds = preds["stars"]
    bg_preds = preds["background"]
    
    # 1. Unpack Flattened Target
    # Shape: [B, H, W, (K * (5 + S2)) + 1]
    B, H, W, C_target = targets.shape
    bg_targets = targets[..., -1:]
    star_targets_flat = targets[..., :-1]
    
    # Infer K: C_pred = K * (5 + S2)
    K = star_preds.shape[-2]
    S2_plus_5 = star_preds.shape[-1]
    star_targets = star_targets_flat.view(B, H, W, K, S2_plus_5)
    
    obj_mask = star_targets[..., 0] == 1.0
    
    # 2. Probability Loss (p) with Faint Star Boosting
    p_pred = torch.clamp(star_preds[..., 0], 1e-7, 1.0 - 1e-7)
    p_target = star_targets[..., 0]
    
    bce_loss = F.binary_cross_entropy(p_pred, p_target, reduction='none')
    p_t = p_pred * p_target + (1 - p_pred) * (1 - p_target)
    focal_weight = (1 - p_t) ** focal_gamma
    alpha_t = focal_alpha * p_target + (1 - focal_alpha) * (1 - p_target)
    
    with torch.no_grad():
        raw_flux_target = star_targets[..., 3]
        
        # Recreate the stretch locally just to calculate the curriculum boost weight
        stretched_target = torch.arcsinh(raw_flux_target / stretch_scale)
        
        # Boost based on the stretched value, keeping your original logic intact
        boost_weight = torch.where(obj_mask, 1.0 + (12.0 - stretched_target) / 6.0, torch.tensor(1.0, device=p_pred.device))
        boost_weight = torch.clamp(boost_weight, 1.0, 5.0)
    
    prob_loss = (alpha_t * focal_weight * bce_loss * boost_weight).mean()
    
    # 3. Regression Losses (Masked)
    if obj_mask.sum() > 0:
        pos_pred = star_preds[..., 1:3][obj_mask]
        pos_target = star_targets[..., 1:3][obj_mask]
        pos_loss = F.mse_loss(pos_pred, pos_target, reduction='mean')
        
        # LOGIT BYPASS: Use pre-activation log-flux directly for stability
        log_flux_pred = preds["raw_log_flux"][obj_mask]
        
        flux_target = star_targets[..., 3:4][obj_mask]
        log_flux_target = torch.log(flux_target + 1e-6)
        flux_loss = F.mse_loss(log_flux_pred, log_flux_target, reduction='mean')
        
        comp_pred = star_preds[..., 4:5][obj_mask]
        comp_target = star_targets[..., 4:5][obj_mask]
        comp_loss = F.mse_loss(comp_pred, comp_target, reduction='mean')
        
        shape_pred = star_preds[..., 5:][obj_mask]
        shape_target = star_targets[..., 5:][obj_mask]
        shape_loss = F.mse_loss(shape_pred, shape_target, reduction='mean')
    else:
        pos_loss = torch.tensor(0.0, device=star_preds.device)
        flux_loss = torch.tensor(0.0, device=star_preds.device)
        comp_loss = torch.tensor(0.0, device=star_preds.device)
        shape_loss = torch.tensor(0.0, device=star_preds.device)
        
    # 4. Background Loss (Global MSE)
    bg_loss = F.mse_loss(bg_preds, bg_targets, reduction='mean')
        
    total_loss = (lambda_prob * prob_loss + 
                  lambda_pos * pos_loss + 
                  lambda_flux * flux_loss +
                  lambda_comp * comp_loss +
                  lambda_shape * shape_loss + 
                  lambda_bg * bg_loss)
                  
    return total_loss, prob_loss, pos_loss, flux_loss, shape_loss, bg_loss
