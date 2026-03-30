#!/bin/bash
#SBATCH --job-name=castor_gen
#SBATCH --time=12:00:00            
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32         
#SBATCH --mem=128G                 
#SBATCH --partition=batch          
#SBATCH --output=logs/gen_%j.out
#SBATCH --error=logs/gen_%j.err

# --- SET YOUR STAGE HERE ---
TARGET_STAGE="stage0"
# ---------------------------

echo "🚀 Starting Castor Data Generation for $TARGET_STAGE..."

# 1. Environment Setup
module purge
module load mamba/25.11.0
eval "$(mamba shell hook --shell bash)"
mamba activate /home/malpas.1/project_amber/castor_model/castor_env

# Ensure the root project dir is in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

# 2. Extract configuration values dynamically
FINAL_DIR=$(python3 -c "import yaml; c=yaml.safe_load(open('config/config.yaml')); print(c['curriculum']['$TARGET_STAGE']['data_dir'])")
STAGE_IDX=$(echo $TARGET_STAGE | sed 's/stage//')
NUM_TRAIN=$(python3 -c "import yaml; c=yaml.safe_load(open('config/config.yaml')); print(c['data_params']['num_train_samples'])")
NUM_VAL=$(python3 -c "import yaml; c=yaml.safe_load(open('config/config.yaml')); print(c['data_params']['num_val_samples'])")

if [ -z "$FINAL_DIR" ]; then
    echo "❌ Error: Could not find $TARGET_STAGE in config/config.yaml curriculum."
    exit 1
fi

echo "📂 Target Directory: $FINAL_DIR"

# 3. Data Generation
# Fixed: Uses dynamic STAGE_IDX and standard config path
echo "1️⃣ Generating raw mosaics..."
srun python -u scripts/generate_mosaics.py \
    --config config/config.yaml \
    --stage $STAGE_IDX \
    --num 1000

# Fixed: Removed RAW_DIR mismatch; now points to directory containing /mosaics
# Fixed: Added required sample count arguments from config
echo "2️⃣ Converting to HDF5..."
srun python -u scripts/convert_to_hdf5.py \
    --data_dir "$FINAL_DIR" \
    --train_samples $NUM_TRAIN \
    --val_samples $NUM_VAL

echo "✅ Generation Complete."