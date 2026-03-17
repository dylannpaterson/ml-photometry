import argparse
import os
import torch
from tqdm import tqdm
from multiprocessing import Pool
from src.data.stage0_gaussian import GaussianPretrainingProvider
from src.cloud.config_utils import load_config
import shutil

def generate_and_save_sample(args):
    """Worker function to generate a single sample and save it."""
    idx, output_dir, dataset_params = args
    provider = GaussianPretrainingProvider(**dataset_params)
    sparse_data = provider.generate_chunk()
    # Save the sparse dictionary directly
    torch.save(sparse_data, os.path.join(output_dir, f"sample_{idx:05d}.pt"))

def pregenerate_dataset(num_samples, output_dir, dataset_params, num_workers=4, force_regenerate=False):
    """Parallel pre-generation of the dataset."""
    if force_regenerate and os.path.exists(output_dir):
        print(f"Force regenerate: Deleting existing directory {output_dir}")
        shutil.rmtree(output_dir)
        
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print(f"Directory {output_dir} already exists. Skipping pre-generation.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Pregenerating {num_samples} samples into {output_dir}...")
    
    tasks = [(i, output_dir, dataset_params) for i in range(num_samples)]
    
    with Pool(num_workers) as p:
        list(tqdm(p.imap(generate_and_save_sample, tasks), total=num_samples))

def main():
    parser = argparse.ArgumentParser(description="Stage-specific data pre-generation")
    parser.add_argument("stage", type=int, help="Stage to generate data for (0=Gaussian)")
    parser.add_argument("--config", default="config/config.yaml")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    if args.stage != 0:
        print(f"❌ Error: Automated generation for stage {args.stage} is not yet implemented.")
        return

    stage_cfg = config["curriculum"]["stage0"]
    data_cfg = config["data_params"]
    
    train_dir = os.path.join(stage_cfg["data_dir"], "train")
    val_dir = os.path.join(stage_cfg["data_dir"], "val")
    
    common_params = {
        "min_stars": data_cfg["min_stars"],
        "max_stars": data_cfg["max_stars"],
        "image_size": data_cfg["image_size"],
        "max_capacity_per_cell": data_cfg["max_capacity_per_cell"],
        "shape_size": data_cfg.get("shape_size", 7)
    }
    
    num_cpus = os.cpu_count() or 4
    workers = max(1, int(num_cpus * 0.75))
    force = config["run_config"]["force_regenerate_data"]
    
    pregenerate_dataset(data_cfg["num_train_samples"], train_dir, common_params, num_workers=workers, force_regenerate=force)
    pregenerate_dataset(data_cfg["num_val_samples"], val_dir, common_params, num_workers=workers, force_regenerate=force)
    
    print(f"✅ Stage {args.stage} data generation complete!")

if __name__ == "__main__":
    main()
