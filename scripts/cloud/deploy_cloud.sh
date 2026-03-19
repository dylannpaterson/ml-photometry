#!/bin/bash

# Cloud Curriculum Deployer for Roman ML Pipeline
# Usage: ./scripts/deploy_cloud.sh [start | get-zone] [stage_index]

# --- CONFIGURATION ---
INSTANCE_NAME="bulge-survey-ml-worker"
ZONES=("asia-east1-b" "us-east4-c")
REPO_URL="https://github.com/dylannpaterson/castor.git"
BRANCH="model-v2-fpn"

# Stage index (defaults to 0: Gaussian Pre-training)
STAGE=${2:-0}

# Function to find the current zone of the VM
get_current_zone() {
    for zone in "${ZONES[@]}"; do
        if gcloud compute instances describe "$INSTANCE_NAME" --zone="$zone" &>/dev/null; then
            echo "$zone"
            return 0
        fi
    done
    return 1
}

# Function to safely start the instance with retries for resource exhaustion
safe_start_instance() {
    local ZONE=$1
    echo "🔃 Attempting to start instance $INSTANCE_NAME in $ZONE..."
    while true; do
        # Capture stderr to check for resource exhaustion
        ERROR_MSG=$(gcloud compute instances start "$INSTANCE_NAME" --zone="$ZONE" --quiet 2>&1)
        if [ $? -eq 0 ]; then
            echo "✅ Instance started successfully."
            return 0
        fi
        
        echo "$ERROR_MSG"
        if [[ "$ERROR_MSG" == *"ZONE_RESOURCE_POOL_EXHAUSTED"* ]]; then
            echo "🕒 Zone resource pool exhausted. Retrying in 60 seconds... (Ctrl+C to cancel)"
            sleep 60
        else
            echo "❌ Failed to start instance due to an unexpected error."
            return 1
        fi
    done
}

if [ "$1" == "get-zone" ]; then
    get_current_zone
    exit 0
fi

if [ "$1" == "start" ]; then
    echo "--- 🚀 Launching Curriculum Stage $STAGE ---"

    ZONE=$(get_current_zone)
    if [ -z "$ZONE" ]; then
        echo "❌ Error: Instance $INSTANCE_NAME not found."
        exit 1
    fi

    # Ensure VM is running
    STATUS=$(gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" --format='get(status)')
    if [ "$STATUS" != "RUNNING" ]; then
        safe_start_instance "$ZONE" || exit 1
    fi

    # 1. Connect and Reset Repo to $BRANCH
    echo "🛰️  Syncing repository to $BRANCH..."
    gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" << EOF
        if [ ! -d "~/ml-photometry" ]; then
            echo "📦 Directory ~/ml-photometry not found. Cloning repository..."
            git clone $REPO_URL ~/ml-photometry
        fi
        cd ~/ml-photometry
        git fetch --all
        git checkout $BRANCH
        git reset --hard origin/$BRANCH
EOF

    # 2. Sync local config (Overwrites the version from the repo)
    echo "🛰️  Syncing local config/config.yaml..."
    gcloud compute scp config/config.yaml "$INSTANCE_NAME":~/ml-photometry/config/config.yaml --zone="$ZONE"

    # 3. Launch Stage
    echo "🛰️  Connecting and starting Stage $STAGE..."

    gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" << EOF
        cd ~/ml-photometry
        
        # Wipe old logs for a clean start
        rm -f training.log pregen.log setup.log
        
        chmod +x scripts/cloud/cloud_setup.sh
        ./scripts/cloud/cloud_setup.sh >> setup.log
        
        export PYTHONPATH=\$PYTHONPATH:.
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

        # 2. Launch Training for this specific stage
        echo "Launching Training for Stage $STAGE..."
        pkill -9 -f 'scripts.run_stage' || true
        nohup python3 -u -m scripts.run_stage $STAGE train >> training.log 2>&1 < /dev/null &

        sleep 2
        echo "✅ Stage $STAGE dispatched. Logs at training.log"
        exit
EOF

    echo "--- 🛰️  Workflow Dispatched! ---"
    exit 0
fi

echo "Usage: ./scripts/deploy_cloud.sh [start | get-zone] [stage_index]"
