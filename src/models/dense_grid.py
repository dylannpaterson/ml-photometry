import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DenseGridModel(nn.Module):
    def __init__(self, K=3, shape_size=7):
        super(DenseGridModel, self).__init__()
        self.K = K
        self.S2 = shape_size * shape_size
        self.num_output_channels = self.K * (5 + self.S2)  # p, dx, dy, m, c + 49 pixels

        # Backbone: Using a ResNet-34
        resnet = models.resnet34(weights=None)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
            resnet.layer1, # Output stride 2: [B, 64, 128, 128]
        )
        
        # Grid Prediction Head
        # Added a 3x3 layer to give it more "thinking space" for the complex PSF shape
        self.head = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_output_channels, kernel_size=1)
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features)
        
        B, C, H, W = out.shape
        out = out.view(B, self.K, 5 + self.S2, H, W)
        out = out.permute(0, 3, 4, 1, 2) # [B, 128, 128, K, 54]
        
        # 1. Base Activations
        p = torch.sigmoid(out[..., 0:1])
        dx = torch.sigmoid(out[..., 1:2]) * 2.0
        dy = torch.sigmoid(out[..., 2:3]) * 2.0
        m = out[..., 3:4]
        c = torch.sigmoid(out[..., 4:5])
        
        # 2. Shape Activation (Softmax over 49 pixels)
        # We apply softmax along the channel dimension of the 49 pixels
        shape_logits = out[..., 5:]
        shape = F.softmax(shape_logits, dim=-1)
        
        return torch.cat([p, dx, dy, m, c, shape], dim=-1)

def compute_grid_loss(pred, target, lambda_prob=1.0, lambda_reg=1.0, lambda_shape=1.0, alpha=0.25, gamma=2.0):
    """
    Implements Masked Loss with Shape Estimation.
    pred, target: [B, 128, 128, 3, 54]
    """
    obj_mask = target[..., 0] == 1.0
    
    # 1. Focal Loss for Probability (p)
    p_pred = torch.clamp(pred[..., 0], 1e-7, 1.0 - 1e-7)
    p_target = target[..., 0]
    
    bce_loss = F.binary_cross_entropy(p_pred, p_target, reduction='none')
    p_t = p_pred * p_target + (1 - p_pred) * (1 - p_target)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = alpha * p_target + (1 - alpha) * (1 - p_target)
    prob_loss = (alpha_t * focal_weight * bce_loss).mean()
    
    # 2. Regression Loss (Masked MSE)
    # dx, dy, m, c
    if obj_mask.sum() > 0:
        reg_pred = pred[..., 1:5][obj_mask]
        reg_target = target[..., 1:5][obj_mask]
        reg_loss = F.mse_loss(reg_pred, reg_target, reduction='mean')
        
        # 3. Shape Loss (Masked MSE)
        shape_pred = pred[..., 5:][obj_mask]
        shape_target = target[..., 5:][obj_mask]
        shape_loss = F.mse_loss(shape_pred, shape_target, reduction='mean')
    else:
        reg_loss = torch.tensor(0.0, device=pred.device)
        shape_loss = torch.tensor(0.0, device=pred.device)
        
    total_loss = lambda_prob * prob_loss + lambda_reg * reg_loss + lambda_shape * shape_loss
    return total_loss, prob_loss, reg_loss, shape_loss
