#!/bin/bash

# Default values
INDIR=${1:-"data/microlensing_stack_strict"}
OUTDIR=${2:-"animations"}

mkdir -p "$OUTDIR"

# 1. Detect number of events from the first ground truth file
FIRST_GT=$(ls "$INDIR"/epoch_*_gt.json 2>/dev/null | head -n 1)

if [ -z "$FIRST_GT" ]; then
    echo "❌ Error: No ground truth files found in $INDIR"
    exit 1
fi

# Use python to safely parse JSON and get number of events
NUM_EVENTS=$(python3 -c "import json; print(len(json.load(open('$FIRST_GT'))['events']))")

echo "🎬 Found $NUM_EVENTS events in $INDIR"
echo "📂 Animations will be saved to $OUTDIR"

# 2. Loop through all events
for (( i=0; i<$NUM_EVENTS; i++ ))
do
    OUT_FILE="$OUTDIR/event_${i}.mp4"
    echo "🎥 Animating Event $i -> $OUT_FILE"
    
    python3 scripts/animate_microlensing.py \
        --indir "$INDIR" \
        --out "$OUT_FILE" \
        --event_idx $i \
        --fps 24
done

echo "✅ All animations complete."
