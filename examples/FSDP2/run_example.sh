# /bin/bash
# bash run_example.sh {file_to_run.py} {num_gpus}
# where file_to_run = example to run. Default = 'example.py'
# num_gpus = num local gpus to use (must be at least 2). Default = 2

# samples to run include:
# example.py
export ODC=1

SCRIPT_DIR=$(dirname $BASH_SOURCE)
# echo "SCRIPT_DIR: ${SCRIPT_DIR}"

netdevs=$(ls /sys/class/net | grep 'bond0\|eth0')
for netdev in $netdevs; do
    echo "Netdev: $netdev"
done

echo "Launching ${1:-example.py} with ${2:-2} gpus"
NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=${netdevs} bash launch.sh --nproc_per_node=${2:-2} ${1:-${SCRIPT_DIR}/example.py}
# torchrun --nnodes=1 --nproc_per_node=${2:-2} ${1:-example.py}
