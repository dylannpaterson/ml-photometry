#!/bin/bash
#SBATCH --job-name=stage0_gen
#SBATCH --time=04:00:00            
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32          
#SBATCH --mem=128G                  
#SBATCH --partition=batch        
#SBATCH --output=logs/stage0_gen_%j.out
#SBATCH --error=logs/stage0_gen_%j.err

# Exit on error
set -e

echo "🚀 Starting Parallel Stage 0 HDF5 Data Generation..."

# 1. Environment Setup
module purge
module load mamba/25.11.0
eval "$(mamba shell hook --shell bash)"
mamba activate /home/malpas.1/project_amber/castor_model/castor_env

# Ensure the root project dir is in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

# Set STPSF_PATH (mostly for consistency, Stage 0 training uses HDF5)
if [ -z "$STPSF_PATH" ]; then
    export STPSF_PATH="/home/malpas.1/project_amber/stpsf-data"
    echo "🛰️  Setting STPSF_PATH to $STPSF_PATH"
fi

# 2. Execute Parallel Generation
# Directly call the stage0_gaussian script which now handles parallel HDF5 generation
python -u castor/data/stage0_gaussian.py

echo "✅ Stage 0 Generation Complete."