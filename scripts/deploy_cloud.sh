#!/bin/bash

# Cloud Deployment Utility for Roman ML Pipeline
# Optimized for Public GitHub Repositories

# --- CONFIGURATION ---
INSTANCE_NAME="bulge-survey-ml-worker"
ZONE="us-central1-a" # Change this if you move zones
REPO_URL="https://github.com/dylannpaterson/ml-photometry.git" # Replace with your actual URL
REMOTE_DIR="ml-photometry"

echo "--- 🚀 Starting Cloud Workflow ---"

# 1. Ensure the VM is running
echo "🔋 Checking VM status..."
STATUS=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format='get(status)' 2>/dev/null)

if [ "$STATUS" != "RUNNING" ]; then
    echo "🔃 Starting instance $INSTANCE_NAME..."
    gcloud compute instances start $INSTANCE_NAME --zone=$ZONE
    if [ $? -ne 0 ]; then
        echo "❌ Error: Could not start VM. Check for 'ZONE_RESOURCE_POOL_EXHAUSTED' errors."
        exit 1
    fi
fi

# 2. Remote Command: Sync Code and Launch
echo "🛰️  Connecting to VM and updating code..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command "
    set -e
    TARGET_DIR=\$HOME/$REMOTE_DIR
    
    if [ ! -d \"\$TARGET_DIR/.git\" ]; then
        echo '📦 Cloning repository...'
        # Remove anything that exists but isn't a git repo
        rm -rf \"\$TARGET_DIR\" 
        git clone $REPO_URL \"\$TARGET_DIR\"
    fi

    cd \"\$TARGET_DIR\"
    echo '📥 Pulling latest changes...'
    git fetch --all
    git reset --hard origin/main # Force sync to latest main

    echo '⚙️  Running environment setup...'
    chmod +x scripts/cloud_setup.sh
    ./scripts/cloud_setup.sh

    # 3. Pregenerate Data (if missing)
    TARGET_DIR=\$HOME/$REMOTE_DIR
    if [ ! -d \"\$TARGET_DIR/data/train\" ] || [ -z \"\$(ls -A \$TARGET_DIR/data/train 2>/dev/null)\" ]; then
        echo '📦 Pregenerating synthetic dataset (one-time setup)...'
        export PYTHONPATH=\$PYTHONPATH:.
        python3 scripts/pregenerate_data.py
    else
        echo '✅ Pregenerated data found. Skipping generation.'
    fi

    echo '🔥 Starting GPU Training...'
    export PYTHONPATH=\$PYTHONPATH:.
    
    # Kill any existing training runs. 
    # We use a regex that starts with 'python3' to ensure we don't match this shell script.
    echo 'Checking for existing training processes...'
    pkill -f '^python3 .*-m scripts.train' || true
    
    # Launch in background, fully detached with redirection
    echo 'Launching nohup process...'
    nohup python3 -u -m scripts.train > training_cloud.log 2>&1 < /dev/null &
    
    # Give it a moment to start and check if it stayed alive
    sleep 3
    if pgrep -f '^python3 .*-m scripts.train' > /dev/null; then
        echo '🚀 Training process verified running in background.'
        echo \"✅ Training successfully launched. PID: \$(pgrep -f '^python3 .*-m scripts.train')\"
    else
        echo '❌ Error: Training process failed to start or died immediately.'
        echo '--- Last 10 lines of training_cloud.log ---'
        tail -n 10 training_cloud.log
        exit 1
    fi
"

echo "--- 🛰️  Workflow Complete! ---"
echo "Monitor logs with:"
echo "gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command 'tail -f ~/ml-photometry/training_cloud.log'"
