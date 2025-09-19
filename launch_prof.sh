#! /bin/bash

# Array of buffer size strings
# BUFFER_SIZES=("256kb" "512kb" "1mb" "2mb" "4mb" "8mb" "16mb" "32mb" "64mb")
PROFILE_ADD_SYNC=0
DUMP_PROFILE_DATA=1
NUM_NODES=1
BUFFER_SIZES=("46799360" "233061376" "233373696" "544997376")
export RUN_REDUCE_SCATTER=0
if [ $RUN_REDUCE_SCATTER -eq 1 ]; then
    DATA_DIR="rs-profile"
    ODC_FUNC="reduce_scatter_accumulation"
    NCCL_FUNC="reduce_scatter_accumulation_nccl"
else
    DATA_DIR="ag-profile"
    ODC_FUNC="all_gather_into_tensor"
    NCCL_FUNC="all_gather_into_tensor_nccl"
fi

# Array of GPU counts
NUM_GPUS_ARRAY=(2 4 8)

check_all_files_exist() {
    local size=$1
    local num_gpus=$2
    
    # Check if all ranks exist for both functions
    for rank in $(seq 0 $((num_gpus - 1))); do
        for func_name in "$ODC_FUNC" "$NCCL_FUNC"; do
            local file_path="${DATA_DIR}/${size}/${func_name}-${size}-${NUM_NODES}-${num_gpus}-${rank}.json"
            if [ ! -f "$file_path" ]; then
                return 1  # File doesn't exist
            fi
        done
    done
    return 0  # All files exist
}

# Iterate over each GPU count
for num_gpus in "${NUM_GPUS_ARRAY[@]}"; do
    echo "Running with NUM_GPUS=$num_gpus"
    
    # Iterate over each buffer size
    for size in "${BUFFER_SIZES[@]}"; do
        if check_all_files_exist "$size" "$num_gpus"; then
            echo "Skipping DATA_SIZE=$size with NUM_GPUS=$num_gpus because all files already exist"
            continue
        fi
        echo "Running with DATA_SIZE=$size and NUM_GPUS=$num_gpus"
        DATA_SIZE=$size NUM_GPUS=$num_gpus bash run.sh
        echo "Completed DATA_SIZE=$size with NUM_GPUS=$num_gpus"
        echo "----------------------------------------"
    done
    
    echo "Completed all buffer sizes for NUM_GPUS=$num_gpus"
    echo "========================================"
done
