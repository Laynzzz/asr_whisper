#!/bin/bash
# Section 5.2: Install Dependencies
# Installs project dependencies from requirements.txt as specified in the plan

set -e  # Exit on any error

echo "📦 Section 5.2: Installing Project Dependencies"
echo "=============================================="
echo ""

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "asr_env" ]]; then
    echo "❌ Please activate the asr_env conda environment first:"
    echo "   conda activate asr_env"
    exit 1
fi

echo "✅ Conda environment 'asr_env' is activated"
echo ""

# Check if requirements.txt exists
if [[ ! -f "requirements.txt" ]]; then
    echo "❌ requirements.txt not found in current directory"
    echo "   Please ensure you are in the project root directory"
    exit 1
fi

echo "✅ Found requirements.txt"
echo ""

# Display requirements that will be installed
echo "📋 Dependencies to install:"
echo "--------------------------"
cat requirements.txt
echo ""

# Install dependencies as specified in Section 5.2
echo "🔧 Installing dependencies from requirements.txt..."
echo "Command from plan: pip install -r requirements.txt"
echo ""

pip install -r requirements.txt

echo ""
echo "✅ All dependencies installed successfully!"
echo ""

# Verify key installations
echo "🧪 Verifying key installations..."
echo "--------------------------------"

python -c "
import sys
print(f'Python version: {sys.version}')
print()

# Check key packages
packages = [
    'transformers', 'datasets', 'torch', 'torchaudio', 
    'evaluate', 'jiwer', 'peft', 'accelerate', 
    'tensorboard', 'pydub', 'librosa'
]

for package in packages:
    try:
        module = __import__(package)
        version = getattr(module, '__version__', 'unknown')
        print(f'✅ {package}: {version}')
    except ImportError:
        print(f'❌ {package}: NOT INSTALLED')
"

echo ""
echo "🎯 Project dependencies installed successfully!"
echo "   Ready for the next step: Download processed data from S3"
