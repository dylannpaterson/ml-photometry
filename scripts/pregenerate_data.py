import os
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from scripts.generate_simple_synthetic_data import GaussianStarDataset

class PregeneratedDataset(torch.utils.data.Dataset):
    """Dataset that loads samples pre-generated on disk."""
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pt")])
        if not self.files:
             raise RuntimeError(f"No .pt files found in {data_dir}. Run pre-generation first.")

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        return torch.load(file_path, weights_only=True)

    def __len__(self):
        return len(self.files)

def generate_and_save_sample(args):
    """Worker function to generate a single sample and save it."""
    idx, output_dir, dataset_params = args
    dataset = GaussianStarDataset(**dataset_params)
    image, target = dataset[0] 
    
    torch.save((image, target), os.path.join(output_dir, f"sample_{idx:05d}.pt"))

from scripts.config_utils import load_config
import shutil

def pregenerate_dataset(num_samples, output_dir, dataset_params, num_workers=4, force_regenerate=False):
    """Parallel pre-generation of the dataset."""
    if force_regenerate and os.path.exists(output_dir):
        print(f"Force regenerate: Deleting existing directory {output_dir}")
        shutil.rmtree(output_dir)
        
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print(f"Directory {output_dir} already exists and is not empty. Skipping pre-generation.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Pregenerating {num_samples} samples into {output_dir}...")
    
    tasks = [(i, output_dir, dataset_params) for i in range(num_samples)]
    
    with Pool(num_workers) as p:
        list(tqdm(p.imap(generate_and_save_sample, tasks), total=num_samples))

if __name__ == "__main__":
    config = load_config()
    
    run_cfg = config["run_config"]
    data_cfg = config["data_params"]
    
    train_dir = "data/train"
    val_dir = "data/val"
    num_train = data_cfg["num_train_samples"]
    num_val = data_cfg["num_val_samples"]
    
    common_params = {
        "min_stars": data_cfg["min_stars"],
        "max_stars": data_cfg["max_stars"],
        "image_size": data_cfg["image_size"],
        "max_capacity_per_cell": data_cfg["max_capacity_per_cell"]
    }
    
    num_cpus = os.cpu_count() or 4
    workers = max(1, int(num_cpus * 0.75))
    
    force = run_cfg["force_regenerate_data"]
    
    pregenerate_dataset(num_train, train_dir, common_params, num_workers=workers, force_regenerate=force)
    pregenerate_dataset(num_val, val_dir, common_params, num_workers=workers, force_regenerate=force)
    
    print("Dataset pre-generation complete!")
