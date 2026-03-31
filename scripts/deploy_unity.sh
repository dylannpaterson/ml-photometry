#!/bin/bash
set -e

REMOTE_HOST="unity-login"
REMOTE_DIR="~/project_amber/castor_model"

echo "🚀 Syncing Castor repository to Unity..."

# Ensure the nested directories exist before syncing
ssh $REMOTE_HOST "mkdir -p $REMOTE_DIR/logs"

# Sync the codebase over the tunnel
# Notice the leading slash on /data/ !
rsync -avz --progress \
    --exclude='.git/' \
    --exclude='castor_env/' \
    --exclude='__pycache__/' \
    --exclude='checkpoints/' \
    --exclude='/data/' \
    --exclude='backups/' \
    --exclude='*.fits' \
    --exclude='*.png' \
    --exclude='*.asdf' \
    ./ $REMOTE_HOST:$REMOTE_DIR/

echo "✅ Sync complete."
echo ""
echo "To set up the environment, SSH in and run: cd ~/project_amber/castor_model && ./cluster_setup.sh"
echo "To generate data, run: sbatch generate_data.slurm"
echo "To train the model, run: sbatch unity_train.slurm"