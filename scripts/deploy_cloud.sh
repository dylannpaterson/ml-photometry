#!/bin/bash

# Cloud Deployment & Lifecycle Utility for Roman ML Pipeline

# --- CONFIGURATION ---
INSTANCE_NAME="bulge-survey-ml-worker"
ZONES=("us-central1-c" "us-central1-a") # MATCHING YOUR EXACT ORDER
REPO_URL="https://github.com/dylannpaterson/ml-photometry.git"
REMOTE_DIR="ml-photometry"

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

# --- HELPER: DOWNLOAD RESULTS ---
if [ "$1" == "get-results" ]; then
    ZONE=$(get_current_zone)
    if [ -z "$ZONE" ]; then
        echo "❌ Error: Could not find instance $INSTANCE_NAME in any configured zone."
        exit 1
    fi
    
    echo "🔋 Ensuring VM is started in $ZONE for download..."
    gcloud compute instances start "$INSTANCE_NAME" --zone="$ZONE" --quiet
    
    echo "⏳ Waiting for SSH service to become available..."
    # Loop until SSH is ready (up to 60 seconds)
    for i in {1..12}; do
        if gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command "echo SSH_READY" --quiet &>/dev/null; then
            echo "✅ SSH is ready."
            break
        fi
        if [ $i -eq 12 ]; then
            echo "❌ Error: SSH timed out after 60 seconds."
            exit 1
        fi
        echo "..."
        sleep 5
    done

    echo "📦 Preparing results on VM..."
    # Archive on the VM first to make SCP more robust (one single file)
    gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" << 'EOF'
        cd ~/ml-photometry
        tar -czf results_bundle.tar.gz checkpoints/ training_cloud.log
EOF

    echo "🛰️  Downloading results bundle..."
    mkdir -p checkpoints
    gcloud compute scp "$INSTANCE_NAME":~/ml-photometry/results_bundle.tar.gz ./results_bundle.tar.gz --zone="$ZONE"
    
    echo "📦 Extracting results locally..."
    tar -xzf results_bundle.tar.gz
    cp training_cloud.log training_cloud_downloaded.log
    rm results_bundle.tar.gz
    
    echo "✅ Results updated in local 'checkpoints/' and 'training_cloud_downloaded.log'."
    echo "🔃 Stopping VM to save costs..."
    gcloud compute instances stop "$INSTANCE_NAME" --zone="$ZONE" --quiet
    exit 0
fi

echo "--- 🚀 Starting Cloud Workflow ---"

# 1. Ensure the VM is running
ZONE=$(get_current_zone)
if [ -z "$ZONE" ]; then
    echo "❌ Error: Instance $INSTANCE_NAME not found in configured zones."
    exit 1
fi

STATUS=$(gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" --format='get(status)')
if [ "$STATUS" != "RUNNING" ]; then
    echo "🔃 Starting instance $INSTANCE_NAME in $ZONE..."
    gcloud compute instances start "$INSTANCE_NAME" --zone="$ZONE" --quiet
fi

# 2. Sync config.yaml before starting
echo "🛰️  Syncing config.yaml to VM..."
gcloud compute scp config.yaml "$INSTANCE_NAME":~/ml-photometry/config.yaml --zone="$ZONE"

# 3. Remote Command: Sync Code and Launch
echo "🛰️  Connecting to VM in $ZONE and launching training..."

# The << 'EOF' tells SSH to just read the following lines as if you were typing them
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" << 'EOF'
    # 1. Go to directory and update
    cd ~/ml-photometry
    git fetch --all
    git reset --hard origin/main
    
    # 2. Setup environment (source it so it persists)
    chmod +x scripts/cloud_setup.sh
    source scripts/cloud_setup.sh
    
    # 3. Pregenerate Data (Always run, it will check if it needs to regenerate)
    export PYTHONPATH=$PYTHONPATH:.
    python3 scripts/pregenerate_data.py
    
    # 4. Clean up old runs and launch!
    pkill -f 'scripts.train' || true
    export PYTHONPATH=$PYTHONPATH:.
    
    # Launch in background with auto-shutdown
    nohup bash -c "(python3 -u -m scripts.train || echo 'TRAINING_CRASHED') >> training_cloud.log 2>&1; sudo poweroff" < /dev/null &
    disown -h %1
    
    echo "✅ Training launched with auto-shutdown. Closing SSH connection..."
    exit
EOF

echo "--- 🛰️  Workflow Complete! ---"
echo "Check progress with: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command \"tail -f ~/ml-photometry/training_cloud.log\""

