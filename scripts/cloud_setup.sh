#!/bin/bash

# Cloud Setup Script for Roman ML Photometry Pipeline
# Designed for GCP Vertex AI Workbench (PyTorch/GPU)

echo "--- Starting Cloud Environment Setup ---"

# 1. Update and Install System Dependencies (if any)
# Vertex AI usually has most of these, but good to be sure
sudo apt-get update && sudo apt-get install -y htop nvtop screen

# 2. Install Python Dependencies
echo "Installing Python dependencies..."
# scipy is required for Hungarian matching in evaluate.py
pip install --upgrade pip
pip install scipy matplotlib numpy torch torchvision

# 3. Verify GPU Availability (Crucial for Cloud runs)
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null
then
    nvidia-smi
    python3 -c "import torch; print(f'PyTorch GPU available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
else
    echo "WARNING: No NVIDIA GPU detected. Ensure your GCP instance has a GPU attached."
fi

# 4. Set PYTHONPATH for the current session
# This ensures scripts can find the 'models' and 'scripts' modules
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "PYTHONPATH set to: $PYTHONPATH"

# 5. Create necessary directories
mkdir -p checkpoints

echo "--- Setup Complete! ---"
echo "To run training on GPU, use: python3 -m scripts.train"
echo "To monitor GPU usage, run 'nvtop' in a separate terminal."
