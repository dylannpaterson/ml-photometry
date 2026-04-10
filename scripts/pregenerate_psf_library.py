import os
import torch
import numpy as np
import argparse
from castor.data.stage0_gaussian import generate_field_realistic_psf_library, _compute_eigen_psfs
from castor.constants import SHAPE_SIZE, N_PCA_COMPONENTS

def main():
    parser = argparse.ArgumentParser(description="Pre-generate a Master PSF Library for fast mosaic rendering.")
    parser.add_argument("--num_psfs", type=int, default=100, help="Number of unique PSF states to generate.")
    parser.add_argument("--output", type=str, default="master_psf_library.pt", help="Output filename.")
    args = parser.parse_args()

    print(f"🛠️  Pre-generating Master PSF Library with {args.num_psfs} states...")
    
    # 1. Generate the physical PSFs
    # This involves rotations and convolutions (the slow part)
    kb_array = generate_field_realistic_psf_library(num_psfs=args.num_psfs, grid_size=SHAPE_SIZE)
    
    # 2. Extract the PCA components
    eigen_psfs, weights_lib, mean_psf = _compute_eigen_psfs(kb_array, n_components=N_PCA_COMPONENTS)
    
    # 3. Save as a dictionary for easy loading
    master_data = {
        'kb_array': kb_array, # Physical PSFs
        'eigen_psfs': eigen_psfs,
        'weights_lib': weights_lib,
        'mean_psf': mean_psf,
        'num_psfs': args.num_psfs,
        'n_pca': N_PCA_COMPONENTS,
        'shape_size': SHAPE_SIZE
    }
    
    torch.save(master_data, args.output)
    print(f"✅ Master PSF Library saved to {args.output}")

if __name__ == "__main__":
    main()
