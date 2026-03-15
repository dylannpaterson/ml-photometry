import os
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from scripts.generate_simple_synthetic_data import GaussianStarDataset

def generate_and_save_sample(args):
    """Worker function to generate a single sample and save it."""
    idx, output_dir, dataset_params = args
    dataset = GaussianStarDataset(**dataset_params)
    image, target = dataset[0] 
    
    torch.save((image, target), os.path.join(output_dir, f"sample_{idx:05d}.pt"))

def pregenerate_dataset(num_samples, output_dir, dataset_params, num_workers=4):
    """Parallel pre-generation of the dataset."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Pregenerating {num_samples} samples into {output_dir}...")
    
    tasks = [(i, output_dir, dataset_params) for i in range(num_samples)]
    
    with Pool(num_workers) as p:
        list(tqdm(p.imap(generate_and_save_sample, tasks), total=num_samples))

class PregeneratedDataset(torch.utils.data.Dataset):
    """PyTorch Dataset that loads samples from pre-generated .pt files."""
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return torch.load(os.path.join(self.data_dir, self.samples[idx]))

if __name__ == "__main__":
    # Match scripts/train.py
    train_dir = "data/train"
    val_dir = "data/val"
    num_train = 5000
    num_val = 500
    
    common_params = {
        "min_stars": 500,
        "max_stars": 1500,
        "image_size": 256,
        "max_capacity_per_cell": 5
    }
    
    num_cpus = os.cpu_count() or 4
    workers = max(1, int(num_cpus * 0.75))
    
    pregenerate_dataset(num_train, train_dir, common_params, num_workers=workers)
    pregenerate_dataset(num_val, val_dir, common_params, num_workers=workers)
    
    print("Dataset pre-generation complete!")
