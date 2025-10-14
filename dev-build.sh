#!/bin/bash

# Development installation script for nvshmem-triton
set -e

echo "Setting up nvshmem-triton for development..."

# Check if clang-21 is installed
if ! command -v clang-21 &> /dev/null; then
    echo "clang-21 not found, installing..."
    apt-get update
    apt-get install -y lsb-release wget software-properties-common gnupg
    wget https://apt.llvm.org/llvm.sh
    chmod +x llvm.sh
    ./llvm.sh 21 all
else
    echo "clang-21 is already installed, skipping installation."
fi

# Install build dependencies
echo "Installing build dependencies..."
pip install -U pip setuptools wheel
pip install cmake>=3.18 scikit-build-core>=0.3.3
pip install nvidia-nvshmem-cu12

# Install the package in editable mode with dev dependencies
echo "Installing package in editable mode..."
pip install --no-build-isolation -e ".[dev]"

echo ""
echo "Development setup completed!"
echo ""
echo "You can now:"
echo "  - Edit Python files in nvshmem_triton/ and changes will be reflected immediately"
echo "  - Run tests with: pytest"
echo "  - Format code with: black ."
echo "  - Sort imports with: isort ."
echo "  - Type check with: mypy nvshmem_triton/"
echo ""
echo "If you modify C++/CUDA files (csrc/), you'll need to rebuild:"
echo "  pip install -e . --no-build-isolation --force-reinstall --no-deps"
