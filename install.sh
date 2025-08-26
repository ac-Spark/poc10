#!/bin/bash

# Hunyuan3D-2 Installation Script
# This script installs all dependencies and compiles required modules

set -e  # Exit on any error

echo "=== Hunyuan3D-2 Installation Script ==="
echo "Starting installation process..."

# Note: Please make sure you're in your preferred virtual environment
# (Poetry, venv, conda, or Docker) before running this script

echo "1. Installing PyTorch with CUDA 12.4 support..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu126

echo "2. Installing pre-compiled custom_rasterizer..."
pip install custom_rasterizer-0.1-cp310-cp310-linux_x86_64.whl

echo "3. Installing Blender Python API..."
pip install bpy==4.0.0 --extra-index-url https://download.blender.org/pypi/

echo "4. Installing other Python dependencies..."
pip install -r requirements.txt

echo "5. Compiling DifferentiableRenderer module..."
cd hy3dpaint/DifferentiableRenderer
bash compile_mesh_painter.sh
cd ../..

echo "6. Downloading RealESRGAN model (if not exists)..."
if [ ! -f "hy3dpaint/ckpt/RealESRGAN_x4plus.pth" ]; then
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt
else
    echo "RealESRGAN model already exists, skipping download."
fi

# Set CUDA library path for runtime
echo "7. Setting up CUDA library paths..."
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Note: If you encounter CUDA library issues, make sure CUDA runtime is installed:"
echo "  sudo apt-get install cuda-runtime-12-6"
echo "  or add to your shell profile: export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:\$LD_LIBRARY_PATH"

echo "=== Installation completed successfully! ==="
echo "You can now run the demo with: python demo.py"