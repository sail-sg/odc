#!/bin/bash

# Development installation script for nvshmem-triton
set -e

echo "Setting up nvshmem-triton for development..."

apt-get update
apt-get install git

# Check if clang-21 is installed
if ! command -v clang-21 &> /dev/null; then
    echo "clang-21 not found, installing..."
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
pip install cmake>=3.18
pip install nvidia-nvshmem-cu12

# Build nvshmem wrapper
mkdir -p build
cd build
cmake ..
make
