#!/bin/bash

# Development installation script for nvshmem-triton
set -e

echo "Setting up nvshmem-triton for development..."

bash build_nvshmem_wrapper.sh

# Install the package in editable mode with dev dependencies
echo "Installing package in editable mode..."
pip install --no-build-isolation -e ".[dev]"

echo ""
echo "Development setup completed!"
echo ""
echo "If you modify C++/CUDA files (csrc/), you'll need to rebuild:"
echo "  bash build_nvshmem_wrapper.sh"
echo "  pip install -e . --no-build-isolation --force-reinstall --no-deps"
