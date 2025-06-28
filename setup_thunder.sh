#!/bin/bash

# Thunder Compute Setup Script for DeepSeek R1 Verilog Fine-tuning

echo "Setting up DeepSeek R1 Verilog Fine-tuning Environment on Thunder Compute..."

# Create and activate conda environment
conda create -n deepseek_verilog python=3.10 -y
conda activate deepseek_verilog

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install transformers accelerate datasets evaluate
pip install deepspeed wandb
pip install trl peft bitsandbytes
pip install huggingface_hub tokenizers

# Install Verilog evaluation tools
pip install verilog-eval

# Additional ML utilities
pip install numpy pandas scikit-learn matplotlib seaborn jupyter ipywidgets

# Create project directories
mkdir -p data/raw data/processed models checkpoints logs scripts configs

# Download datasets
echo "Downloading Verilog datasets..."
cd data/raw

# VerilogEval
git clone https://github.com/NVlabs/verilog-eval.git
cd verilog-eval && pip install -e . && cd ..

# HDL-Bits
wget https://github.com/alinush/hdlbits-verilog/archive/main.zip -O hdlbits.zip
unzip hdlbits.zip

# OpenROAD examples
git clone --depth 1 https://github.com/The-OpenROAD-Project/OpenROAD.git
find OpenROAD -name "*.v" -o -name "*.sv" > verilog_files.txt

# Additional Verilog repositories
git clone --depth 1 https://github.com/steveicarus/ivtest.git
git clone --depth 1 https://github.com/YosysHQ/yosys.git

cd ../../

echo "Thunder Compute environment setup complete!"
echo "Run 'conda activate deepseek_verilog' to activate the environment"