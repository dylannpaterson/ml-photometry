#!/bin/bash

echo "🛠️ Setting up Castor Environment on Unity with Mamba..."

module purge
module load mamba/25.11.0
module load cuda

# Hook Mamba into this bash session so we can use the 'activate' command
eval "$(mamba shell hook --shell bash)"

# Create the environment with Python 3.10 if it doesn't exist
if [ ! -d "castor_env" ]; then
    echo "Creating Mamba environment..."
    mamba create -y -p ./castor_env python=3.10
fi

# Activate and install dependencies
mamba activate ./castor_env
echo "Upgrading pip and installing requirements..."
pip install --upgrade pip
pip install --extra-index-url https://download.pytorch.org/whl/cu124 -e .
pip install webbpsf stpsf

echo "✅ Environment 'castor_env' is ready to use!"