#!/bin/bash

# Build script for odc package
set -e

bash build_nvshmem_wrapper.sh

echo "Building package..."
python -m build --no-isolation --verbose

echo "Build completed successfully!"
echo "Built packages are in the dist/ directory:"
ls -la dist/

echo ""
echo "To install the package, run:"
echo "pip install dist/odc-*.whl"
