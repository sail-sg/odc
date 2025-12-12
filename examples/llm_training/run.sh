# export WANDB_MODE=disabled
export ODC=${ODC:-1}
if [ "${HSDP:-0}" -eq '1' ]; then
    HSDP_FLAG="_HSDP"
    export NVSHMEM_SYMMETRIC_SIZE=${NVSHMEM_SYMMETRIC_SIZE:-10000000000}
else
    HSDP_FLAG=""
    export NVSHMEM_SYMMETRIC_SIZE=${NVSHMEM_SYMMETRIC_SIZE:-10000000000}
fi

if [ "${FSDP2:-0}" -eq '1' ]; then
    FSDP_NAME="FSDP2"
else
    FSDP_NAME="FSDP1"
fi

if [ "${ODC}" -eq '1' ]; then
    COMM_NAME="ODC"
else
    COMM_NAME="NCCL"
fi

export RUN_NAME="${FSDP_NAME}_${COMM_NAME}${HSDP_FLAG}"
group_name="profile_group"

if [ ! -d "data/longalign64" ]; then
    echo "data/longalign64 does not exist, running preprocessing..."
    python examples/llm_training/preprocess_dataset.py --num_samples 1000 --output data/longalign64
else
    echo "data/longalign64 already exists, skipping preprocessing"
fi

netdevs=$(ls /sys/class/net | grep 'bond0\|eth0')
netdev=$(echo $netdevs | head -n1 | awk '{print $1}')
echo "Netdev: $netdev"
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=${netdev}
export NCCL_SOCKET_IFNAME=${netdev}

if [ "${FSDP2:-0}" -eq 1 ]; then
    echo "Running FSDP2: ODC: ${ODC}"
    script="examples/llm_training/torch_fsdp2.py"
else
    echo "Running FSDP1: ODC: ${ODC}"
    script="examples/llm_training/torch_fsdp.py"
fi

bash launch.sh ${script} \
           --minibatch_size 4 \
           --micro_batch_size 1 \
           --run_name ${RUN_NAME} \
           --project_name ${group_name}
        #    --limit_dataset_token_len 40000 \
