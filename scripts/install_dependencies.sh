#!/bin/bash

set -e

echo "Setting up DeepSeek R1 Verilog Fine-tuning Environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment
ENV_NAME="deepseek_verilog"
echo "Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.10 -y

# Activate environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt

# Install additional Verilog tools
echo "Installing Verilog evaluation tools..."
# Install iverilog for syntax checking
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y iverilog verilator
elif command -v yum &> /dev/null; then
    sudo yum install -y iverilog verilator
elif command -v brew &> /dev/null; then
    brew install icarus-verilog verilator
else
    echo "Warning: Could not install Verilog tools automatically. Please install iverilog and verilator manually."
fi

# Install the package in development mode
echo "Installing package in development mode..."
pip install -e .

echo "Environment setup complete!"
echo "To activate the environment, run: conda activate $ENV_NAME"