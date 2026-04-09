#!/bin/bash
#SBATCH --job-name=verify_gpu_render
#SBATCH --output=logs/verify_gpu_%j.out
#SBATCH --error=logs/verify_gpu_%j.err
#SBATCH --partition=batch-gpu
#SBATCH --gpus-per-node=1 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00

echo "🚀 Starting GPU Renderer Verification on Cluster..."
date

# 1. Environment Setup
module purge
module load mamba/25.11.0
module load cuda/12.9.0-vvrwkrx
eval "$(mamba shell hook --shell bash)"
mamba activate /home/malpas.1/project_amber/castor_model/castor_env

# Ensure the root project dir is in PYTHONPATH so imports work
export PYTHONPATH=$PYTHONPATH:.

# Ensure logs directory exists
mkdir -p logs

# Run verification
python3 scripts/verify_gpu_renderer_accuracy.py --mosaic_size 1024 --num_stars 500

echo "✅ Verification Job Complete."
date
