# /bin/bash
# bash run_example.sh {file_to_run.py} {num_gpus}
# where file_to_run = example to run. Default = 'example.py'
# num_gpus = num local gpus to use (must be at least 2). Default = 2

# samples to run include:
# example.py
# export ODC=1

export ODC=${ODC:-1}
SCRIPT_DIR=$(dirname $BASH_SOURCE)
# echo "SCRIPT_DIR: ${SCRIPT_DIR}"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra _cuda_visible <<< "${CUDA_VISIBLE_DEVICES}"
    export GPUS_PER_NODE=${#_cuda_visible[@]}
    echo "Setting GPUS_PER_NODE=${GPUS_PER_NODE} from CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

netdevs=$(ls /sys/class/net | grep 'bond0\|eth0')
netdev=$(echo $netdevs | head -n1 | awk '{print $1}')
echo "Netdev: $netdev"

echo "Launching ${1:-example.py} with ${2:-2} gpus"
NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=${netdev} bash launch.sh --nproc_per_node=${2:-2} ${1:-${SCRIPT_DIR}/example.py} --mixed-precision
