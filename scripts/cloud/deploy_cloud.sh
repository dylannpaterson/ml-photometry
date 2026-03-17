#!/bin/bash

# Cloud Curriculum Deployer for Roman ML Pipeline
# Usage: ./scripts/deploy_cloud.sh [get-results | start | get-zone] [stage_index]

# --- CONFIGURATION ---
INSTANCE_NAME="bulge-survey-ml-worker"
ZONES=("us-east4-c" "asia-east1-c" "asia-east1-a" "us-central1-c" "us-central1-a")
REPO_URL="https://github.com/dylannpaterson/ml-photometry.git"
BRANCH="add-psf"

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

# --- HELPER: DOWNLOAD RESULTS ---
if [ "$1" == "get-results" ]; then
    ZONE=$(get_current_zone)
    if [ -z "$ZONE" ]; then
        echo "❌ Error: Could not find instance $INSTANCE_NAME."
        exit 1
    fi
    
    echo "🔋 Preparing VM in $ZONE for download..."
    safe_start_instance "$ZONE" || exit 1
    
    echo "⏳ Waiting for SSH..."
    for i in {1..10}; do
        if gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command "echo SSH_READY" --quiet &>/dev/null; then break; fi
        sleep 5
    done

    echo "📦 Downloading results..."
    gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command "cd ~/ml-photometry && tar -czf results_bundle.tar.gz checkpoints/ training_cloud.log"
    gcloud compute scp "$INSTANCE_NAME":~/ml-photometry/results_bundle.tar.gz ./results_bundle.tar.gz --zone="$ZONE"
    tar -xzf results_bundle.tar.gz && rm results_bundle.tar.gz
    
    echo "✅ Results updated. Stopping VM..."
    gcloud compute instances stop "$INSTANCE_NAME" --zone="$ZONE" --quiet
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
        
        chmod +x scripts/cloud/cloud_setup.sh
        ./scripts/cloud/cloud_setup.sh
        
        export PYTHONPATH=\$PYTHONPATH:.
        
        # 1. Pregenerate data for this specific stage (if needed)
        echo "Checking data for Stage $STAGE (Logging to pregen_cloud.log)..."
        python3 scripts/pregenerate_data.py $STAGE >> pregen_cloud.log 2>&1
        
        # 2. Launch Training for this specific stage
        echo "Launching Training for Stage $STAGE..."
        pkill -9 -f 'scripts.run_stage' || true
        nohup bash -c "python3 -u -m scripts.run_stage $STAGE train >> training.log 2>&1" < /dev/null &
        disown
        
        echo "✅ Stage $STAGE dispatched. Logs at training_cloud.log"
        exit
EOF
    echo "--- 🛰️  Workflow Dispatched! ---"
    exit 0
fi

echo "Usage: ./scripts/deploy_cloud.sh [get-results | start | get-zone] [stage_index]"
