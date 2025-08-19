#!/bin/bash

# Build script for odc package
set -e

echo "Installing build dependencies..."
pip install -U pip setuptools wheel

echo "Building package..."
python -m build --no-isolation --verbose

echo "Build completed successfully!"
echo "Built packages are in the dist/ directory:"
ls -la dist/

echo ""
echo "To install the package, run:"
echo "pip install dist/odc-*.whl"
