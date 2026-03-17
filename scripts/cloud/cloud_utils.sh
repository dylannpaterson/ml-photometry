#!/bin/bash

# Cloud Utilities for Roman ML Pipeline
# Usage: cloud [status | logs | tail | gpu | kill | ssh | reboot | results]

INSTANCE_NAME="bulge-survey-ml-worker"
ZONES=("us-east4-c" "asia-east1-c" "asia-east1-a" "us-central1-c" "us-central1-a")

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

ZONE=$(get_current_zone)
if [ -z "$ZONE" ]; then
    echo "❌ Error: Could not find instance $INSTANCE_NAME."
    exit 1
fi

case "$1" in
    status)
        gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" --format='table(name,zone,status,guestAttributes.architecture:label=ARCH)'
        ;;
    logs)
        gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command "tail -n 50 ~/ml-photometry/training.log" --quiet
        ;;
    tail)
        gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command "tail -f ~/ml-photometry/training.log" --quiet
        ;;
    gpu)
        gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command "nvidia-smi" --quiet
        ;;
    kill)
        echo "🛑 Killing running training processes..."
        gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command "pkill -9 -f 'scripts.run_stage'" --quiet
        echo "✅ Processes killed."
        ;;
    ssh)
        gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE"
        ;;
    run)
        if [ -z "$2" ]; then
            echo "❌ Error: No command provided. Usage: cloud run <command>"
            exit 1
        fi
        gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command "${@:2}" --quiet
        ;;
    reboot)
        echo "🔄 Rebooting instance $INSTANCE_NAME..."
        gcloud compute instances reset "$INSTANCE_NAME" --zone="$ZONE"
        echo "✅ Reset signal sent. Wait a minute for it to come back online."
        ;;
    results)
        # Check if instance is already running
        INITIAL_STATUS=$(gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" --format='get(status)')
        echo "Current VM status: $INITIAL_STATUS"
        
        if [ "$INITIAL_STATUS" != "RUNNING" ]; then
            echo "🔋 Preparing VM in $ZONE for download..."
            safe_start_instance "$ZONE" || exit 1
        fi
        
        echo "⏳ Waiting for SSH..."
        for i in {1..10}; do
            if gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command "echo SSH_READY" --quiet &>/dev/null; then break; fi
            sleep 5
        done

        echo "📦 Downloading results..."
        # Create a timestamped bundle to avoid overwriting local ones if desired, 
        # but sticking to the current logic of results_bundle.tar.gz for now.
        gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command "cd ~/ml-photometry && tar -czf results_bundle.tar.gz checkpoints/ training.log"
        gcloud compute scp "$INSTANCE_NAME":~/ml-photometry/results_bundle.tar.gz ./results_bundle.tar.gz --zone="$ZONE"
        tar -xzf results_bundle.tar.gz && rm results_bundle.tar.gz
        
        if [ "$INITIAL_STATUS" != "RUNNING" ]; then
            echo "✅ Results updated. Stopping VM (as it was started by this script)..."
            gcloud compute instances stop "$INSTANCE_NAME" --zone="$ZONE" --quiet
        else
            echo "✅ Results updated. Leaving VM running."
        fi
        ;;
    *)
        echo "Usage: $0 [status | logs | tail | gpu | kill | ssh | reboot | results]"
        exit 1
        ;;
esac
