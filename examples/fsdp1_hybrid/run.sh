# export WANDB_MODE=disabled
export ODC=${ODC:-1}
export NVSHMEM_SYMMETRIC_SIZE=${NVSHMEM_SYMMETRIC_SIZE:-20000000000}

if [ "${FSDP2:-0}" -eq 1 ]; then
    FSDP_NAME="FSDP2"
else
    FSDP_NAME="FSDP1"
fi

if [ "${ODC}" -eq 1 ]; then
    COMM_NAME="ODC"
else
    COMM_NAME="NCCL"
fi

run_name="${FSDP_NAME}_${COMM_NAME}"
group_name="fsdp_test"

if [ ! -d "data/longalign64" ]; then
    echo "data/longalign64 does not exist, running preprocessing..."
    python examples/fsdp1_hybrid/preprocess_dataset.py --num_samples 1000 --output data/longalign64
else
    echo "data/longalign64 already exists, skipping preprocessing"
fi

netdevs=$(ls /sys/class/net | grep 'bond0\|eth0')
for netdev in $netdevs; do
    echo "Netdev: $netdev"
done
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=${netdevs}

if [ "${FSDP2:-0}" -eq 1 ]; then
    echo "Running FSDP2"
    script="examples/fsdp1_hybrid/torch_fsdp2.py"
else
    echo "Running FSDP1"
    script="examples/fsdp1_hybrid/torch_fsdp.py"
fi

bash launch.sh ${script} \
           --limit_dataset_token_len 20000 \
           --packing_method DynamicSameMicro \
           --minibatch_size 2 \
           --micro_batch_size 1 \
           --run_name ${run_name} \
           --project_name ${group_name}

