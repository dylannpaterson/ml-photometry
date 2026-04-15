#!/bin/bash

# Cloud Watchdog for Roman ML Pipeline
# Periodically attempts to relaunch training if it's not active or finished.

INSTANCE_NAME="bulge-survey-ml-worker"
ZONES=("asia-east1-a" "us-central1-c" "us-central1-a")
CHECK_INTERVAL=600 # 10 minutes

# Function to get current zone
get_current_zone() {
    for zone in "${ZONES[@]}"; do
        if gcloud compute instances describe "$INSTANCE_NAME" --zone="$zone" &>/dev/null; then
            echo "$zone"
            return 0
        fi
    done
    return 1
}

# Function to check if training is completed in logs
check_completion() {
    ZONE=$(get_current_zone)
    if [ -z "$ZONE" ]; then return 1; fi
    
    # Try to check if training_cloud.log says it's done
    # We might need the VM to be running to check logs, or use the local copy
    # Let's assume we check the VM's log if it's running
    STATUS=$(gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" --format='get(status)')
    if [ "$STATUS" == "RUNNING" ]; then
        if gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command "grep -q 'Final Model saved' ~/ml-photometry/training_cloud.log" 2>/dev/null; then
            return 0 # Completed!
        fi
    fi
    return 1 # Not completed or can't tell
}

echo "--- 🐕 Starting Cloud Watchdog ---"
echo "Will check every $(($CHECK_INTERVAL / 60)) minutes."

while true; do
    ZONE=$(get_current_zone)
    if [ -z "$ZONE" ]; then
        echo "❌ Error: Instance $INSTANCE_NAME not found in configured zones."
        exit 1
    fi

    STATUS=$(gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" --format='get(status)')
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] VM Status: $STATUS"

    if [ "$STATUS" == "TERMINATED" ]; then
        echo "🔄 VM is down. Attempting relaunch..."
        bash scripts/deploy_cloud.sh
        
        # Give it a moment to actually start or fail
        sleep 30
        NEW_STATUS=$(gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" --format='get(status)')
        if [ "$NEW_STATUS" == "TERMINATED" ]; then
            echo "⚠️  Relaunch failed (likely stockout). Retrying in $(($CHECK_INTERVAL / 60)) mins..."
        else
            echo "✅ Relaunch successful. Monitoring..."
        fi
    elif [ "$STATUS" == "RUNNING" ]; then
        # If it's running, check if it's actually training or just idling/finishing
        if check_completion; then
            echo "🎉 Training appears to be FINISHED! Exiting watchdog."
            exit 0
        fi
        echo "🛰️  VM is running. Assuming training is in progress."
    fi

    echo "⏳ Waiting $CHECK_INTERVAL seconds..."
    sleep $CHECK_INTERVAL
done
