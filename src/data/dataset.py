import os
import torch
from torch.utils.data import Dataset

class PregeneratedDataset(Dataset):
    """Dataset that loads sparse samples and re-densifies them on the fly."""
    def __init__(self, data_dir, K=3, shape_size=7):
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
        shapes = sparse_data["shapes"]       # [N_stars, 49]
        indices = sparse_data["indices"]     # [N_stars, 3] (y, x, slot)
        
        H, W, K, _ = base_grid.shape
        
        # 1. Create the full dense target tensor [128, 128, K, 54]
        # 54 = 5 (p, dx, dy, m, c) + 49 (shape)
        target = torch.zeros((H, W, K, 5 + self.S2), dtype=torch.float32)
        
        # 2. Fill the first 5 channels
        target[..., :5] = base_grid
        
        # 3. Re-densify the 49 shape channels
        if len(indices) > 0:
            # indices[:, 0] is y, indices[:, 1] is x, indices[:, 2] is slot
            target[indices[:, 0], indices[:, 1], indices[:, 2], 5:] = shapes
            
        return image, target

    def __len__(self):
        return len(self.files)
