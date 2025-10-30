# export WANDB_MODE=disabled
export ODC=1
export NVSHMEM_SYMMETRIC_SIZE=${NVSHMEM_SYMMETRIC_SIZE:-20000000000}

run_name='fsdp1_odc'
group_name='fsdp1_test'

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

bash launch.sh examples/fsdp1_hybrid/torch_fsdp.py \
           --run_name ${run_name} \
           --project_name ${group_name}

