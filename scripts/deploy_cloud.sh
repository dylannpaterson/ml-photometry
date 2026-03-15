#!/bin/bash

# Cloud Deployment Utility for Roman ML Pipeline
# Optimized for Public GitHub Repositories

# --- CONFIGURATION ---
INSTANCE_NAME="bulge-survey-ml-worker"
ZONE="us-central1-b" # Change this if you move zones
REPO_URL="https://github.com/dylannpaterson/ml-photometry.git" # Replace with your actual URL
REMOTE_DIR="~/ml-photometry"

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
    if [ ! -d \"$REMOTE_DIR\" ]; then
        echo '📦 Cloning repository...'
        git clone $REPO_URL $REMOTE_DIR
    fi

    cd $REMOTE_DIR
    echo '📥 Pulling latest changes...'
    git fetch --all
    git reset --hard origin/main # Force sync to latest main

    echo '⚙️  Running environment setup...'
    chmod +x scripts/cloud_setup.sh
    ./scripts/cloud_setup.sh

    echo '🔥 Starting GPU Training...'
    export PYTHONPATH=\$PYTHONPATH:.
    # Kill any existing training runs to avoid conflicts
    pkill -f 'python3 -u -m scripts.train' || true
    nohup python3 -u -m scripts.train > training_cloud.log 2>&1 &
    
    echo '✅ Training successfully launched in the background.'
"

echo "--- 🛰️  Workflow Complete! ---"
echo "Monitor logs with:"
echo "gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command 'tail -f ~/ml-photometry/training_cloud.log'"
