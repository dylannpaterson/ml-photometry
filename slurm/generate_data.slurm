#!/bin/bash
#SBATCH --job-name=castor_gen
#SBATCH --time=12:00:00            
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16         
#SBATCH --mem=64G                 
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
NUM_MOSAICS=$(python3 -c "import yaml; c=yaml.safe_load(open('config/config.yaml')); print(c['curriculum']['$TARGET_STAGE']['mosaic_params']['num_mosaics'])")
NUM_VAL_MOSAICS=$(python3 -c "import yaml; c=yaml.safe_load(open('config/config.yaml')); print(c['curriculum']['$TARGET_STAGE']['mosaic_params']['val_mosaics'])")

if [ -z "$FINAL_DIR" ]; then
    echo "❌ Error: Could not find $TARGET_STAGE in config/config.yaml curriculum."
    exit 1
fi

TRAIN_MOS_DIR="$FINAL_DIR/mosaics_train"
VAL_MOS_DIR="$FINAL_DIR/mosaics_val"

echo "📂 Target Directory: $FINAL_DIR"
echo "🧩 Generating $NUM_MOSAICS train and $NUM_VAL_MOSAICS val mosaics..."

# 3. Data Generation
echo "1️⃣ Generating raw TRAINING mosaics..."
mkdir -p "$TRAIN_MOS_DIR"
srun python -u scripts/generate_mosaics.py \
    --config config/config.yaml \
    --stage $STAGE_IDX \
    --num $NUM_MOSAICS \
    --output_dir "$TRAIN_MOS_DIR"

echo "2️⃣ Generating raw VALIDATION mosaics..."
mkdir -p "$VAL_MOS_DIR"
srun python -u scripts/generate_mosaics.py \
    --config config/config.yaml \
    --stage $STAGE_IDX \
    --num $NUM_VAL_MOSAICS \
    --output_dir "$VAL_MOS_DIR"

echo "3️⃣ Converting to HDF5 (Separate Splits)..."
srun python -u scripts/convert_to_hdf5.py \
    --train_dir "$TRAIN_MOS_DIR" \
    --val_dir "$VAL_MOS_DIR" \
    --output_dir "$FINAL_DIR" \
    --train_samples $NUM_TRAIN \
    --val_samples $NUM_VAL

# Cleanup raw directories
echo "🧹 Cleaning up raw mosaic directories..."
rm -rf "$TRAIN_MOS_DIR"
rm -rf "$VAL_MOS_DIR"

echo "✅ Generation Complete."