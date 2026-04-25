import h5py
import numpy as np
import os

def check_chunk_zero(h5_path):
    if not os.path.exists(h5_path):
        print(f"File {h5_path} not found.")
        return

    with h5py.File(h5_path, 'r') as f:
        print(f"\nChecking {h5_path} (Chunk 0)...")
        if "targets" in f:
            targets = f["targets"][0] # Get first chunk
            bg_channel = targets[..., -1]
            print(f"  Background Channel Chunk 0 Min:  {np.min(bg_channel)}")
            print(f"  Background Channel Chunk 0 Max:  {np.max(bg_channel)}")
            print(f"  Background Channel Chunk 0 Mean: {np.mean(bg_channel)}")
            
            if np.any(np.abs(bg_channel) > 1e10):
                print(f"  !!! EXTREME VALUES DETECTED in Background Channel !!!")
                indices = np.where(np.abs(bg_channel) > 1e10)
                for i in range(min(5, len(indices[0]))):
                    idx = (indices[0][i], indices[1][i])
                    print(f"    Value at {idx}: {bg_channel[idx]}")
        
        if "images" in f:
            img = f["images"][0]
            print(f"  Image Chunk 0 Min:  {np.min(img)}")
            print(f"  Image Chunk 0 Max:  {np.max(img)}")

if __name__ == "__main__":
    check_chunk_zero("data/bulge_stage0_full/stage0_val.h5")
    check_chunk_zero("data/smoke_test/stage0_data.h5")
