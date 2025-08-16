#!/bin/bash
# Section 5: Cloud GPU Environment Setup for Training
# This script sets up the remote EC2 instance environment for Whisper fine-tuning

set -e  # Exit on any error

echo "🚀 Section 5: Cloud GPU Environment for Training"
echo "================================================"
echo ""

echo "📋 Section 5.1: GPU Selection Recommendations"
echo "--------------------------------------------"
echo "Cost-effective GPU options recommended for this task:"
echo "  • NVIDIA T4     - Good entry-level option"
echo "  • NVIDIA A5000  - Balanced performance/cost"
echo "  • NVIDIA RTX A6000 - High performance option"
echo ""
echo "These GPUs offer optimal balance of performance and cost for Whisper fine-tuning."
echo ""

echo "🔧 Section 5.2: Driver, CUDA, and Environment Setup"
echo "---------------------------------------------------"
echo ""

# Check if running on EC2 instance
if ! curl -s --max-time 3 http://169.254.169.254/latest/meta-data/instance-id &>/dev/null; then
    echo "⚠️  This script is designed to run on an AWS EC2 instance."
    echo "   Please run this on your cloud GPU instance."
    echo ""
    echo "📝 Manual setup instructions for EC2 instance:"
    echo "   1. Launch EC2 instance with GPU (T4/A5000/RTX A6000)"
    echo "   2. SSH into the instance"
    echo "   3. Run this script on the remote instance"
    echo ""
    exit 1
fi

echo "✅ Running on EC2 instance"
echo ""

# Update system packages
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers and CUDA
echo "🎮 Installing NVIDIA Drivers and CUDA..."
echo "----------------------------------------"

# Check if NVIDIA drivers are already installed
if ! command -v nvidia-smi &> /dev/null; then
    echo "Installing NVIDIA drivers..."
    
    # Install NVIDIA driver
    sudo apt install -y ubuntu-drivers-common
    sudo ubuntu-drivers autoinstall
    
    # Install CUDA Toolkit 12.1
    echo "Installing CUDA Toolkit 12.1..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-1
    
    # Add CUDA to PATH
    echo 'export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
    source ~/.bashrc
    
    echo "⚠️  System reboot required for NVIDIA drivers to take effect."
    echo "   Please reboot the instance and run this script again."
    exit 0
else
    echo "✅ NVIDIA drivers already installed"
    nvidia-smi
fi

# Install Miniconda if not present
echo ""
echo "🐍 Installing Miniconda..."
echo "-------------------------"

if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    rm miniconda.sh
else
    echo "✅ Conda already installed"
fi

# Create conda environment as specified in Section 5.2
echo ""
echo "🌍 Creating Conda Environment (Section 5.2)..."
echo "----------------------------------------------"

# Remove existing environment if it exists
conda env remove -n asr_env -y 2>/dev/null || true

# Create new environment with Python 3.10 as specified
conda create -n asr_env -y python=3.10

echo "✅ Conda environment 'asr_env' created successfully"
echo ""

echo "🎯 Next Steps:"
echo "1. Activate environment: conda activate asr_env"
echo "2. Install PyTorch with CUDA: ./scripts/install_pytorch_cuda.sh"
echo "3. Install dependencies: ./scripts/install_dependencies.sh"
echo "4. Download data: ./scripts/download_data_from_s3.sh"
