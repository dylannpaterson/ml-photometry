import os
import torch
from torch.utils.data import Dataset

class PregeneratedDataset(Dataset):
    """Dataset that loads samples pre-generated on disk."""
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
            
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pt")])
        if not self.files:
             raise RuntimeError(f"No .pt files found in {data_dir}. Run pre-generation first.")

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        # Returns (image_tensor, target_tensor)
        return torch.load(file_path, weights_only=True)

    def __len__(self):
        return len(self.files)
