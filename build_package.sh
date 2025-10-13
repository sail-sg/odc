#!/bin/bash

# Build script for odc package
set -e

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

echo "Building odc package..."

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/

# Create build directory
mkdir -p build

# Build the package
echo "Building package..."
python -m pip install build
python -m build --no-isolation --verbose

echo "Build completed successfully!"
echo "Built packages are in the dist/ directory:"
ls -la dist/

echo ""
echo "To install the package, run:"
echo "pip install dist/odc-*.whl"
