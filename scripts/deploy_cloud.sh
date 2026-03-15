#!/bin/bash

# Cloud Deployment & Lifecycle Utility for Roman ML Pipeline
# Optimized for Public GitHub Repositories

# --- CONFIGURATION ---
INSTANCE_NAME="bulge-survey-ml-worker"
ZONE="us-central1-a"
REMOTE_DIR="ml-photometry"

# --- HELPER: DOWNLOAD RESULTS ---
if [ "$1" == "get-results" ]; then
    echo "🔋 Ensuring VM is started for download..."
    gcloud compute instances start $INSTANCE_NAME --zone=$ZONE --quiet
    
    echo "🛰️  Downloading results..."
    mkdir -p checkpoints
    gcloud compute scp --recurse $INSTANCE_NAME:~/ml-photometry/checkpoints/* ./checkpoints/ --zone=$ZONE
    gcloud compute scp $INSTANCE_NAME:~/ml-photometry/training_cloud.log ./training_cloud_downloaded.log --zone=$ZONE
    
    echo "✅ Results downloaded to local 'checkpoints/' and 'training_cloud_downloaded.log'."
    echo "🔃 Stopping VM to save costs..."
    gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE --quiet
    exit 0
fi

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
        rm -rf \"\$TARGET_DIR\" 
        git clone $REPO_URL \"\$TARGET_DIR\"
    fi

    cd \"\$TARGET_DIR\"
    echo '📥 Pulling latest changes...'
    git fetch --all
    git reset --hard origin/main

    echo '⚙️  Running environment setup...'
    chmod +x scripts/cloud_setup.sh
    ./scripts/cloud_setup.sh

    # 3. Pregenerate Data (if missing)
    if [ ! -d \"data/train\" ] || [ -z \"\$(ls -A data/train 2>/dev/null)\" ]; then
        echo '📦 Pregenerating synthetic dataset (one-time setup)...'
        export PYTHONPATH=\$PYTHONPATH:.
        python3 scripts/pregenerate_data.py
    else
        echo '✅ Pregenerated data found. Skipping generation.'
    fi

    echo '🔥 Launching training with AUTO-SHUTDOWN...'
    export PYTHONPATH=\$PYTHONPATH:.
    
    # Kill any existing training runs
    pkill -f '^python3 .*-m scripts.train' || true
    
    # Launch subshell in background: (Run Training -> Poweroff)
    # Redirecting ALL outputs to log and detaching with nohup
    nohup bash -c '(python3 -u -m scripts.train && echo \"SUCCESS: Training finished.\" || echo \"FAILURE: Training crashed.\") >> training_cloud.log 2>&1; sudo poweroff' > /dev/null 2>&1 < /dev/null &
    
    echo '✅ Training successfully launched. VM will power off automatically when done.'
    sleep 3
    pgrep -f '^python3 .*-m scripts.train' > /dev/null && echo '🚀 Process verified running.' || echo '⚠️ Warning: Process not found after launch.'
"

echo "--- 🛰️  Workflow Complete! ---"
echo "To check logs: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command 'tail -f ~/ml-photometry/training_cloud.log'"
echo "To collect results later: ./scripts/deploy_cloud.sh get-results"
