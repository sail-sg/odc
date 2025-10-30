#!/bin/bash

# Development installation script for nvshmem-triton
set -e

echo "Setting up nvshmem-triton for development..."

# Install the package in editable mode with dev dependencies
echo "Installing package in editable mode..."
pip install --no-build-isolation -e ".[dev]"

echo ""
echo "Development setup completed!"
echo ""
echo "If you modify C++/CUDA files (csrc/), you'll need to rebuild:"
echo "  pip install -e . --no-build-isolation --force-reinstall --no-deps"
