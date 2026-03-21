import os
import torch
from torch.utils.data import Dataset

class PregeneratedDataset(Dataset):
    """Dataset that loads sparse samples and re-densifies them on the fly."""
    def __init__(self, data_dir, K=3, shape_size=9):
        self.data_dir = data_dir
        self.K = K
        self.S2 = shape_size * shape_size
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
            
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pt")])
        if not self.files:
             raise RuntimeError(f"No .pt files found in {data_dir}. Run pre-generation first.")

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        sparse_data = torch.load(file_path, weights_only=True)
        
        image = sparse_data["image"]
        base_grid = sparse_data["base_grid"] # [128, 128, K, 5]
        bg_map = sparse_data["background_map"] # [128, 128]
        shapes = sparse_data["shapes"]       # [N_stars, S2]
        indices = sparse_data["indices"]     # [N_stars, 3] (y, x, slot)
        
        H, W, K, _ = base_grid.shape
        C_stars = K * (5 + self.S2)
        
        # Target: [128, 128, (K * 86) + 1] -> Flat channels to match DenseGridModel output
        target = torch.zeros((H, W, C_stars + 1), dtype=torch.float32)
        
        # 1. Build K-slot star targets
        star_target = torch.zeros((H, W, K, 5 + self.S2), dtype=torch.float32)
        
        # Fill base metadata (p, dx, dy, m, c)
        star_target[..., :5] = base_grid
        
        # Fill shapes
        if len(indices) > 0:
            # indices is [N, 3] (y, x, k)
            star_target[indices[:, 0], indices[:, 1], indices[:, 2], 5:] = shapes
            
        # 2. Flatten K dimension and Append Background
        target[..., :C_stars] = star_target.view(H, W, C_stars)
        target[..., -1] = bg_map
            
        return image, target

    def __len__(self):
        return len(self.files)
