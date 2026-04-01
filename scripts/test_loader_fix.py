import torch
from castor.data.stage0_gaussian import GaussianPretrainingProvider
import numpy as np

def test_loader():
    print("Testing GaussianPretrainingProvider...")
    dataset = GaussianPretrainingProvider(num_samples=5)
    
    for i in range(len(dataset)):
        print(f"Generating sample {i}...")
        sample = dataset[i]
        image = sample["image"]
        target = sample["target"]
        print(f"  Sample {i} generated. Image shape: {image.shape}, Target shape: {target.shape}")
    
    print("✅ GaussianPretrainingProvider test passed.")

if __name__ == "__main__":
    test_loader()
