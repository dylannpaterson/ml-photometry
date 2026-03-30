#!/bin/bash
#SBATCH --job-name=castor_train
#SBATCH --time=48:00:00            
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          
#SBATCH --mem=64G                  
#SBATCH --partition=batch-gpu      
#SBATCH --gpus-per-node=1          
#SBATCH --constraint="a100|v100" 
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# --- SET YOUR STAGE HERE ---
TARGET_STAGE="stage0"
# ---------------------------

echo "🚀 Starting Castor Training for $TARGET_STAGE..."

# 1. Environment Setup
module purge
module load mamba/25.11.0
module load cuda/12.9.0-vvrwkrx
eval "$(mamba shell hook --shell bash)"
mamba activate /home/malpas.1/project_amber/castor_model/castor_env

# Ensure the root project dir is in PYTHONPATH so imports work
export PYTHONPATH=$PYTHONPATH:.

# 2. Extract configuration and Index
# Extracts the integer (0) from "stage0"
STAGE_IDX=$(echo $TARGET_STAGE | sed 's/stage//')
STAGE_NAME=$(python3 -c "import yaml; c=yaml.safe_load(open('config/config.yaml')); print(c['curriculum']['$TARGET_STAGE']['name'])")

if [ -z "$STAGE_NAME" ]; then
    echo "❌ Error: $TARGET_STAGE not found in config/config.yaml curriculum."
    exit 1
fi

echo "📂 Stage Name: $STAGE_NAME"

# Create the standard checkpoints directory if it doesn't exist
# run_stage.py manages sub-prefixes automatically (e.g., stage0_final.pth)
mkdir -p "./checkpoints"

# 3. Hardware Check
nvidia-smi --query-gpu=name,memory.total --format=csv

# 4. Execute Training
# Fixed: Points to scripts/run_stage.py
# Fixed: Passes STAGE_IDX (int) and "train" as positional arguments
srun python -u scripts/run_stage.py \
    $STAGE_IDX \
    train \
    --config config/config.yaml

echo "✅ Training session finished."