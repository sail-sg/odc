#! /bin/bash

# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# Set default values if not already set
START_GPU=${START_GPU:-0}
NUM_GPUS=${NUM_GPUS:-4}
export ODC_SINGLE_BUFFER=0
export ODC_NUM_BUFFERS=${NUM_GPUS}
# export ODC_MAX_BUFFER_SIZE=128000000

# Check if nvidia-cuda-mps-control process exists
# if ! pgrep -f "nvidia-cuda-mps-control" > /dev/null; then
#     echo "Error: nvidia-cuda-mps-control process not found!"
#     echo "Please start the NVIDIA CUDA Multi-Process Service (MPS) before running this script."
#     echo "You can start it with: nvidia-cuda-mps-control -d"
#     exit 1
# fi
# 
# echo "nvidia-cuda-mps-control process found."

# Initialize ALL_FREE variable
ALL_FREE=true

for ((i=START_GPU; i<START_GPU+NUM_GPUS; i++)); do
    # Check if GPU memory usage is zero
    GPU_USAGE=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$GPU_USAGE" -lt 1000 ]; then
        echo "GPU $i is free."
    else
        echo "GPU $i is in use. Usage is $GPU_USAGE."
        ALL_FREE=false
    fi
done

# Keep checking until all GPUs are free
while [ "$ALL_FREE" = false ]; do
    echo "Waiting for all GPUs to be free..."
    sleep 10  # Wait 10 seconds before checking again
    
    ALL_FREE=true
    for ((i=START_GPU; i<START_GPU+NUM_GPUS; i++)); do
        GPU_USAGE=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader,nounits)
        if [ "$GPU_USAGE" -lt 1000 ]; then
            echo "GPU $i is free."
        else
            echo "GPU $i is in use. Usage is $GPU_USAGE."
            ALL_FREE=false
        fi
    done
done

echo "All GPUs are now free!"

export CUDA_DEVICE_MAX_CONNECTIONS=12
# export CUDA_LAUNCH_BLOCKING=1
# export NVSHMEM_SYMMETRIC_SIZE=128000000
export NVSHMEM_SYMMETRIC_SIZE=$((128000000 * 8 * 20 * 2))

export PYTHONPATH="$(dirname $(pwd))"
echo $PYTHONPATH

# Check if RUN_REDUCE_SCATTER is defined
if [ -z "${RUN_REDUCE_SCATTER+x}" ]; then
    echo "Error: RUN_REDUCE_SCATTER environment variable is not defined!"
    echo "Please set RUN_REDUCE_SCATTER to either 0 (for all_gather) or 1 (for reduce_scatter)"
    echo "Example: RUN_REDUCE_SCATTER=1 ./run.sh"
    exit 1
fi

netdevs=$(ls /sys/class/net | grep 'bond0\|eth0')
for netdev in $netdevs; do
    echo "Netdev: $netdev"
done

if [ $RUN_REDUCE_SCATTER -eq 1 ]; then
    NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=${netdevs} bash launch.sh --nproc_per_node=${NUM_GPUS} reduce_scatter.py
else
    NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=${netdevs} bash launch.sh --nproc_per_node ${NUM_GPUS} all_gather.py
fi

# NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 bash launch.sh --nproc_per_node 4 latency.py
# NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 bash launch.sh --nproc_per_node 2 crosslock.py

