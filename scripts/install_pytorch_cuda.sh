#!/bin/bash
# Section 5.2: Install PyTorch with CUDA Support
# Installs PyTorch with CUDA 12.1 support as specified in the plan

set -e  # Exit on any error

echo "🔥 Section 5.2: Installing PyTorch with CUDA Support"
echo "===================================================="
echo ""

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "asr_env" ]]; then
    echo "❌ Please activate the asr_env conda environment first:"
    echo "   conda activate asr_env"
    exit 1
fi

echo "✅ Conda environment 'asr_env' is activated"
echo ""

# Check CUDA availability
echo "🔍 Checking CUDA installation..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA drivers detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
    echo ""
else
    echo "⚠️  NVIDIA drivers not detected. This script will install CPU-only PyTorch."
    echo "   For GPU support, ensure NVIDIA drivers and CUDA are installed first."
    echo ""
fi

# Install PyTorch with CUDA 12.1 support as specified in Section 5.2
echo "📦 Installing PyTorch with CUDA 12.1 support..."
echo "-----------------------------------------------"
echo "Command from plan: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
echo ""

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "✅ PyTorch installation completed"
echo ""

# Verify installation
echo "🧪 Verifying PyTorch installation..."
echo "-----------------------------------"

python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('CUDA not available - running on CPU')
"

echo ""
echo "🎯 PyTorch with CUDA 12.1 support installed successfully!"
echo "   Ready for the next step: Install project dependencies"
