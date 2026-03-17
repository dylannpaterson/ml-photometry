import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class DenseGridModel(nn.Module):
    def __init__(self, K=3, shape_size=9):
        super(DenseGridModel, self).__init__()
        self.K = K
        self.S2 = shape_size * shape_size
        self.num_output_channels = self.K * (5 + self.S2) + 1

        resnet = models.resnet34(weights=None)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
            resnet.layer1, 
        )
        
        self.head = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_output_channels, kernel_size=1)
        )
        
        # Initialize background bias to ~10.0
        # The last channel is the background channel
        with torch.no_grad():
            self.head[-1].bias[-1].fill_(10.0)

    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features)
        
        B, C, H, W = out.shape
        star_out = out[:, :-1, :, :]
        bg_out = out[:, -1:, :, :]
        
        star_out = star_out.view(B, self.K, 5 + self.S2, H, W)
        star_out = star_out.permute(0, 3, 4, 1, 2)
        
        p = torch.sigmoid(star_out[..., 0:1])
        dx = torch.sigmoid(star_out[..., 1:2]) * 2.0
        dy = torch.sigmoid(star_out[..., 2:3]) * 2.0
        m = star_out[..., 3:4]
        c = torch.sigmoid(star_out[..., 4:5])
        
        shape_logits = star_out[..., 5:]
        shape = F.softmax(shape_logits, dim=-1)
        
        bg = F.relu(bg_out.permute(0, 2, 3, 1))
        
        return {
            "stars": torch.cat([p, dx, dy, m, c, shape], dim=-1),
            "background": bg
        }

def compute_grid_loss(preds, targets, lambda_prob=1.0, lambda_reg=1.0, lambda_shape=1.0, lambda_bg=0.01, alpha=0.25, gamma=2.0):
    """
    Implements Masked Loss with Shape Estimation and Background Fit.
    lambda_bg is set to 0.01 to balance against star parameters.
    """
    star_preds = preds["stars"]
    bg_preds = preds["background"]
    
    bg_targets = targets[..., 0, -1:]
    star_targets = targets[..., :-1]
    
    obj_mask = star_targets[..., 0] == 1.0
    
    # 1. Focal Loss for Probability (p)
    p_pred = torch.clamp(star_preds[..., 0], 1e-7, 1.0 - 1e-7)
    p_target = star_targets[..., 0]
    
    bce_loss = F.binary_cross_entropy(p_pred, p_target, reduction='none')
    p_t = p_pred * p_target + (1 - p_pred) * (1 - p_target)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = alpha * p_target + (1 - alpha) * (1 - p_target)
    prob_loss = (alpha_t * focal_weight * bce_loss).mean()
    
    # 2. Regression Loss (Masked MSE)
    if obj_mask.sum() > 0:
        reg_pred = star_preds[..., 1:5][obj_mask]
        reg_target = star_targets[..., 1:5][obj_mask]
        reg_loss = F.mse_loss(reg_pred, reg_target, reduction='mean')
        
        shape_pred = star_preds[..., 5:][obj_mask]
        shape_target = star_targets[..., 5:][obj_mask]
        shape_loss = F.mse_loss(shape_pred, shape_target, reduction='mean')
    else:
        reg_loss = torch.tensor(0.0, device=star_preds.device)
        shape_loss = torch.tensor(0.0, device=star_preds.device)
        
    # 3. Background Loss (Global MSE)
    bg_loss = F.mse_loss(bg_preds, bg_targets, reduction='mean')
        
    total_loss = (lambda_prob * prob_loss + 
                  lambda_reg * reg_loss + 
                  lambda_shape * shape_loss + 
                  lambda_bg * bg_loss)
                  
    return total_loss, prob_loss, reg_loss, shape_loss, bg_loss
