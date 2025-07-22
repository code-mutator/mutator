#!/bin/bash

# Fix Architecture Issues for Mutator
echo "🔧 Fixing architecture issues for Mutator..."

# Step 1: Check current architecture
echo "📊 Current architecture:"
arch
uname -m
python3 -c "import platform; print(f'Python arch: {platform.machine()}')" 2>/dev/null || echo "Python check failed"

# Step 2: Navigate to project directory
cd /Users/amrmagdy/Desktop/work/mutator

# Step 3: Remove problematic virtual environments
echo "🧹 Cleaning up old virtual environments..."
rm -rf venv_arm64 venv_native_arm64 2>/dev/null

# Step 4: Check if we have Homebrew Python (preferred for ARM64 Macs)
if command -v /opt/homebrew/bin/python3 &> /dev/null; then
    echo "✅ Using Homebrew Python (native ARM64)"
    PYTHON_CMD="/opt/homebrew/bin/python3"
elif command -v /usr/local/bin/python3 &> /dev/null; then
    echo "⚠️  Using system Python - will force ARM64"
    PYTHON_CMD="arch -arm64 /usr/local/bin/python3"
else
    echo "⚠️  Using default Python - will force ARM64"
    PYTHON_CMD="arch -arm64 python3"
fi

# Step 5: Create new virtual environment with correct architecture
echo "🏗️  Creating new virtual environment..."
$PYTHON_CMD -m venv venv_fixed

# Step 6: Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv_fixed/bin/activate

# Step 7: Verify we're using the right architecture
echo "🔍 Verifying architecture in virtual environment..."
python -c "import platform; print(f'Python architecture: {platform.machine()}')"

# Step 8: Upgrade pip to latest version
echo "⬆️  Upgrading pip..."
python -m pip install --upgrade pip

# Step 9: Set architecture flags for compilation
export ARCHFLAGS="-arch arm64"
export _PYTHON_HOST_PLATFORM="macosx-10.9-arm64"
export MACOSX_DEPLOYMENT_TARGET="10.9"

# Step 10: Install packages that commonly have architecture issues first
echo "📦 Installing architecture-sensitive packages..."
pip install --no-cache-dir --force-reinstall \
    "numpy>=1.24.0,<2.0" \
    "regex" \
    "tiktoken" \
    "chromadb>=0.4.15"

# Step 11: Install remaining requirements
echo "📦 Installing remaining requirements..."
pip install -r requirements.txt

# Step 12: Install the mutator package in development mode
echo "📦 Installing mutator in development mode..."
pip install -e .

# Step 13: Test the installation
echo "🧪 Testing mutator installation..."
mutator --help

if [ $? -eq 0 ]; then
    echo "✅ SUCCESS! Mutator is now working."
    echo "🚀 You can now run: mutator chat"
else
    echo "❌ Installation failed. Let's try alternative approach..."
    
    # Alternative: Install with specific constraints
    echo "🔄 Trying alternative installation..."
    pip uninstall -y numpy chromadb tiktoken regex litellm
    
    # Install with specific version constraints that are known to work
    pip install --no-cache-dir \
        "numpy==1.26.4" \
        "tiktoken==0.7.0" \
        "regex==2024.9.11" \
        "litellm==1.60.2" \
        "chromadb==0.4.24"
    
    # Try again
    pip install -e .
    mutator --help
    
    if [ $? -eq 0 ]; then
        echo "✅ SUCCESS with alternative versions!"
        echo "🚀 You can now run: mutator chat"
    else
        echo "❌ Still failing. Manual intervention needed."
        echo "📋 Debug info:"
        python -c "import platform; print(f'Platform: {platform.platform()}')"
        python -c "import sys; print(f'Python executable: {sys.executable}')"
        pip list | grep -E "(numpy|regex|tiktoken|litellm|chromadb)"
    fi
fi

echo "�� Script completed." 