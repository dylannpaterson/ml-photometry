import torch
from torch.utils.data import DataLoader
from castor.data.stage0_gaussian import GaussianMosaicDataset
from castor.cloud.config_utils import load_config
import os
import matplotlib.pyplot as plt

def test_loading(config_path="config/config.yaml"):
    config = load_config(config_path)
    stage_cfg = config["curriculum"]["stage0"]
    data_dir = os.path.join(stage_cfg["data_dir"], "mosaics")
    
    print(f"Testing Stage 0 Macro-Sparse Loading from {data_dir}...")
    
    dataset = GaussianMosaicDataset(
        data_dir,
        num_samples=100,
        image_size=256
    )
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Get one batch
    batch = next(iter(loader))
    images = batch["image"]
    targets = batch["target"]
    
    print(f"Batch loaded successfully!")
    print(f"  Images shape: {images.shape}")
    print(f"  Targets shape: {targets.shape}")
    
    # Visualize one crop from the batch
    img = images[0, 0].numpy()
    target = targets[0].numpy()
    
    # Star mask from target
    # target is [64, 64, 259] for K=3, S=9
    star_mask = (np.sum(target[..., :-1].reshape(64, 64, 3, -1)[..., 0], axis=-1) > 0).astype(float)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='inferno', origin='lower')
    plt.title("Network Input (Stretched + Noise)")
    
    plt.subplot(1, 2, 2)
    plt.imshow(star_mask, cmap='gray', origin='lower')
    plt.title("JIT Star Mask Grid")
    
    plt.savefig("test_stage0_loading.png")
    print("✅ Loading test complete. Visualization saved to test_stage0_loading.png")

if __name__ == "__main__":
    import numpy as np
    test_loading()
