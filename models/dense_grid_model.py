import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DenseGridModel(nn.Module):
    def __init__(self, K=5):
        super(DenseGridModel, self).__init__()
        self.K = K
        self.num_output_channels = self.K * 5  # p, dx, dy, m, c

        # Backbone: Using a ResNet-34
        # Input 256x256 -> Stride 2 -> 128x128.
        resnet = models.resnet34(weights=None)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
            # resnet.maxpool, # Removing maxpool to stay at stride 2
            resnet.layer1, # Output stride 2: [B, 64, 128, 128] for 256x256 input
        )
        
        # Grid Prediction Head
        self.head = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.num_output_channels, kernel_size=1)
        )

    def forward(self, x):
        # 1. Feature Extraction (Backbone)
        # Input: [B, 1, 256, 256] -> Output: [B, 64, 128, 128]
        features = self.backbone(x)
        
        # 2. Grid Prediction
        # Output: [B, K*5, 128, 128]
        out = self.head(features)
        
        # 3. Reshape to [B, 128, 128, K, 5]
        B, C, H, W = out.shape
        out = out.view(B, self.K, 5, H, W)
        out = out.permute(0, 3, 4, 1, 2) # [B, 128, 128, K, 5]
        
        # 4. Apply Activations
        p = torch.sigmoid(out[..., 0:1])
        dx = torch.sigmoid(out[..., 1:2]) * 2.0 # Cell size is 2.0
        dy = torch.sigmoid(out[..., 2:3]) * 2.0 # Cell size is 2.0
        m = out[..., 3:4]
        c = torch.sigmoid(out[..., 4:5])
        
        return torch.cat([p, dx, dy, m, c], dim=-1)

def compute_loss(pred, target, lambda_prob=1.0, lambda_reg=1.0, alpha=0.25, gamma=2.0):
    """
    Implements Step 4.B: The Masked Loss with Focal Loss for Probability.
    pred, target: [B, 128, 128, 5, 5]
    """
    # Masks
    obj_mask = target[..., 0] == 1.0
    
    # 1. Focal Loss for Probability (p)
    p_pred = pred[..., 0]
    p_target = target[..., 0]
    
    # Clip for stability
    p_pred = torch.clamp(p_pred, 1e-7, 1.0 - 1e-7)
    
    bce_loss = F.binary_cross_entropy(p_pred, p_target, reduction='none')
    p_t = p_pred * p_target + (1 - p_pred) * (1 - p_target)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = alpha * p_target + (1 - alpha) * (1 - p_target)
    
    prob_loss = (alpha_t * focal_weight * bce_loss).mean()
    
    # 2. Regression Loss (Masked MSE)
    if obj_mask.sum() > 0:
        reg_pred = pred[..., 1:][obj_mask]
        reg_target = target[..., 1:][obj_mask]
        reg_loss = F.mse_loss(reg_pred, reg_target, reduction='mean')
    else:
        reg_loss = torch.tensor(0.0, device=pred.device)
        
    total_loss = lambda_prob * prob_loss + lambda_reg * reg_loss
    return total_loss, prob_loss, reg_loss

if __name__ == "__main__":
    model = DenseGridModel()
    test_input = torch.randn(2, 1, 256, 256)
    output = model(test_input)
    print(f"Model Output Shape: {output.shape}")
    
    target = torch.zeros_like(output)
    # Put a fake star in one slot
    target[0, 64, 64, 0, 0] = 1.0 
    target[0, 64, 64, 0, 1:5] = torch.tensor([1.0, 1.0, 100.0, 0.8])
    
    total, prob, reg = compute_loss(output, target)
    print(f"Loss Test - Total: {total.item():.4f}, Prob: {prob.item():.4f}, Reg: {reg.item():.4f}")
