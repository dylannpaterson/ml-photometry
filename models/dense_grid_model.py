import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DenseGridModel(nn.Module):
    def __init__(self, K=5):
        super(DenseGridModel, self).__init__()
        self.K = K
        self.num_output_channels = self.K * 5  # p, dx, dy, m, c

        # Backbone: Using a ResNet-34 as recommended in the design doc
        # We need to modify the first layer to accept 1-channel grayscale images
        resnet = models.resnet34(weights=None)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1, # Output stride 4: [B, 64, 96, 96] for 384x384 input
        )
        
        # Grid Prediction Head
        # The design says we downsample by 4, then crop the central 64x64.
        # layer1 in ResNet-34 provides a downsampling factor of 4.
        self.head = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.num_output_channels, kernel_size=1)
        )

    def forward(self, x):
        # 1. Feature Extraction (Backbone)
        # Input: [B, 1, 384, 384] -> Output: [B, 64, 96, 96]
        features = self.backbone(x)
        
        # 2. Central Crop to 64x64
        # (96 - 64) / 2 = 16 pixels of padding to remove on each side
        features_cropped = features[:, :, 16:80, 16:80]
        
        # 3. Grid Prediction
        # Output: [B, K*5, 64, 64]
        out = self.head(features_cropped)
        
        # 4. Reshape to [B, 64, 64, K, 5]
        # Torch uses [B, C, H, W], we want to reorder to [B, H, W, K, 5]
        B, C, H, W = out.shape
        out = out.view(B, self.K, 5, H, W)
        out = out.permute(0, 3, 4, 1, 2) # [B, 64, 64, K, 5]
        
        # 5. Apply Activations (As per Design Doc Step 3, Stage 2)
        # Channels: p, dx, dy, m, c
        p = torch.sigmoid(out[..., 0:1])
        dx = torch.sigmoid(out[..., 1:2]) * 4.0
        dy = torch.sigmoid(out[..., 2:3]) * 4.0
        m = out[..., 3:4] # Linear for magnitude
        c = torch.sigmoid(out[..., 4:5])
        
        return torch.cat([p, dx, dy, m, c], dim=-1)

def compute_loss(pred, target, lambda_prob=1.0, lambda_reg=1.0, alpha=0.25, gamma=2.0):
    """
    Implements Step 4.B: The Masked Loss with Focal Loss for Probability.
    Focal Loss helps the model focus on hard-to-detect faint stars by 
    down-weighting easy background cells.
    
    pred, target: [B, 64, 64, 5, 5]
    """
    # Masks
    obj_mask = target[..., 0] == 1.0  # Slots that actually contain a star
    
    # 1. Focal Loss for Probability (p)
    p_pred = pred[..., 0]
    p_target = target[..., 0]
    
    # Standard BCE per element (no reduction yet)
    bce_loss = F.binary_cross_entropy(p_pred, p_target, reduction='none')
    
    # p_t is p if target=1, and (1-p) if target=0
    p_t = p_pred * p_target + (1 - p_pred) * (1 - p_target)
    
    # Focal weight: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** gamma
    
    # Alpha balancing (optional, helps with class imbalance)
    alpha_t = alpha * p_target + (1 - alpha) * (1 - p_target)
    
    prob_loss = (alpha_t * focal_weight * bce_loss).mean()
    
    # 2. Regression Loss (Masked MSE) - Only for slots with a star
    if obj_mask.sum() > 0:
        # dx, dy, m, c
        reg_pred = pred[..., 1:][obj_mask]
        reg_target = target[..., 1:][obj_mask]
        reg_loss = F.mse_loss(reg_pred, reg_target, reduction='mean')
    else:
        reg_loss = torch.tensor(0.0, device=pred.device)
        
    total_loss = lambda_prob * prob_loss + lambda_reg * reg_loss
    return total_loss, prob_loss, reg_loss

if __name__ == "__main__":
    model = DenseGridModel()
    test_input = torch.randn(2, 1, 384, 384)
    output = model(test_input)
    print(f"Model Output Shape: {output.shape}")
    
    target = torch.zeros_like(output)
    # Put a fake star in one slot
    target[0, 32, 32, 0, 0] = 1.0 
    target[0, 32, 32, 0, 1:5] = torch.tensor([2.0, 2.0, 100.0, 0.8])
    
    total, prob, reg = compute_loss(output, target)
    print(f"Loss Test - Total: {total.item():.4f}, Prob: {prob.item():.4f}, Reg: {reg.item():.4f}")