#!/bin/bash

# Test runner script for nvshmem-triton
# This script sets up the NVSHMEM environment and runs pytest

set -e

echo "Setting up NVSHMEM environment for testing..."

# Function to find NVSHMEM_HOME using the same logic as get_nvshmem_home_path()
function set_nvshmem_home() {
    # Use the same logic as get_nvshmem_home_path() from nvshmem_triton.api
    if [ -n "$NVSHMEM_HOME" ]; then
        echo "Found NVSHMEM_HOME from environment variable: $NVSHMEM_HOME"
    else
        # Try to find from Python nvidia-nvshmem-cu12 package
        export NVSHMEM_HOME=$(python3 -c "import nvidia.nvshmem, pathlib; print(pathlib.Path(nvidia.nvshmem.__path__[0]))" 2>/dev/null)

        if [ -n "$NVSHMEM_HOME" ]; then
            echo "Found NVSHMEM_HOME from Python nvidia-nvshmem-cu12: $NVSHMEM_HOME"
        else
            echo "warning: NVSHMEM_HOME could not be determined."
            echo "Please set NVSHMEM_HOME environment variable manually."
            exit 1
        fi
    fi
}

# Set up NVSHMEM environment
set_nvshmem_home

# Set up LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVSHMEM_HOME}/lib

echo "NVSHMEM_HOME: ${NVSHMEM_HOME}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

# Run pytest with any additional arguments passed to this script
echo "Running pytest..."
pytest "$@"

