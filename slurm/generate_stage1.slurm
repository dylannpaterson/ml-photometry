#!/bin/bash
#SBATCH --job-name=stage1_gen
#SBATCH --time=06:00:00            
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32          
#SBATCH --mem=128G                  
#SBATCH --partition=batch        
#SBATCH --output=logs/stage1_gen_%j.out
#SBATCH --error=logs/stage1_gen_%j.err

echo "🚀 Starting Parallel Stage 1 GalSim Data Generation..."

# 1. Environment Setup
module purge
module load mamba/25.11.0
eval "$(mamba shell hook --shell bash)"
mamba activate /home/malpas.1/project_amber/castor_model/castor_env

# Ensure the root project dir is in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

# 2. Execute Parallel Generation
# We use all 32 requested CPUs
srun python -u castor/data/stage1_galsim.py \
    --config config/config.yaml \
    --num_workers 32

echo "✅ Stage 1 Generation Complete."