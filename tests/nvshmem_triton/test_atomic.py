import nvshmem.core
import torch
import triton
import triton.language as tl
from cuda.core.experimental import Device

from odc.primitives import NVSHMEM_EXTERN_LIBS, int_atomic_compare_swap, int_atomic_swap


@triton.jit
def kernel_test_atomic_compare_swap(target_ptr, results_ptr):
    """
    Simple kernel that tests atomic_compare_swap functionality.

    This kernel performs a basic atomic compare-and-swap operation and stores
    the return value to verify the operation is working correctly.

    Args:
        target_ptr: Pointer to the target value (int32*)
        results_ptr: Pointer to store return values (int32*)
    """
    cta_id = tl.program_id(axis=0)

    # Cast pointers to int32
    target_ptr = tl.cast(target_ptr, tl.pointer_type(tl.int32))
    results_ptr = tl.cast(results_ptr, tl.pointer_type(tl.int32))

    # Read the current value at target
    current_value = tl.load(target_ptr)

    # Perform atomic compare-and-swap:
    # If target == current_value, then set target = current_value + 1
    # The function returns the original value at target
    old_value = int_atomic_compare_swap(target_ptr, current_value, current_value + 1, 0)

    # Store the return value to verify the operation
    tl.store(results_ptr + cta_id, old_value)


def test_atomic_compare_swap():
    dev = Device(0)  # Use first GPU
    dev.set_current()

    # Initialize NVSHMEM with single PE using UID method
    uid = nvshmem.core.get_unique_id()
    nvshmem.core.init(device=dev, uid=uid, rank=0, nranks=1, initializer_method="uid")

    my_pe = nvshmem.core.my_pe()

    # Create shared tensors
    target = nvshmem.core.tensor((1,), dtype=torch.int32)
    results = nvshmem.core.tensor((10,), dtype=torch.int32)  # Store results from 10 threads

    # Initialize values
    init_value = 5
    target[0] = init_value  # Start with value 5
    results[:] = -1  # Initialize results with invalid values

    print(f"[PE {my_pe}] Initial target value: {target[0].item()}")

    # Launch the simple test kernel with 10 threads
    num_threads = 10
    grid_size = (num_threads,)

    print(f"[PE {my_pe}] Launching kernel with {num_threads} threads...")

    extern_libs = NVSHMEM_EXTERN_LIBS
    print(f"extern_libs: {extern_libs}")
    kernel_test_atomic_compare_swap[grid_size](target, results, extern_libs=extern_libs)

    # Synchronize
    torch.cuda.synchronize()

    # Print results
    print(f"[PE {my_pe}] Final target value: {target[0].item()}")
    print(f"[PE {my_pe}] Return values from atomic_compare_swap:")

    results_cpu = results.cpu().numpy()
    for i in range(num_threads):
        print(f"  Thread {i}: returned {results_cpu[i]}")

    # Analyze the results
    print(f"\n[PE {my_pe}] Analysis:")
    print(f"  Initial value: {init_value}")
    print(f"  Final value: {target[0].item()}")
    print(f"  Expected final value: {5 + num_threads} ({init_value} + number of threads)")

    # Check if exactly one thread succeeded (got the original value 5)
    successful_threads = sum(1 for val in results_cpu if val == 5)
    print(f"  Number of threads that got original value (5): {successful_threads}")
    print("  Expected: 1 (only one thread should succeed)")
    assert successful_threads == 1, "Only one thread should succeed"

    # Cleanup
    nvshmem.core.free_tensor(target)
    nvshmem.core.free_tensor(results)
    nvshmem.core.finalize()


@triton.jit
def kernel_test_atomic_swap(target_ptr, results_ptr):
    """
    Simple kernel that tests atomic_swap functionality.

    This kernel performs a basic atomic swap operation and stores
    the return value to verify the operation is working correctly.

    Args:
        target_ptr: Pointer to the target value (int32*)
        results_ptr: Pointer to store return values (int32*)
    """
    cta_id = tl.program_id(axis=0)

    # Cast pointers to int32
    target_ptr = tl.cast(target_ptr, tl.pointer_type(tl.int32))
    results_ptr = tl.cast(results_ptr, tl.pointer_type(tl.int32))

    # Perform atomic swap:
    # Set target = cta_id + 100 and return the old value
    # Each thread swaps with a different value based on its ID
    new_value = 100
    old_value = int_atomic_swap(target_ptr, new_value, 0)

    # Store the return value to verify the operation
    tl.store(results_ptr + cta_id, old_value)


def test_atomic_swap():
    dev = Device(0)  # Use first GPU
    dev.set_current()

    # Initialize NVSHMEM with single PE using UID method
    uid = nvshmem.core.get_unique_id()
    nvshmem.core.init(device=dev, uid=uid, rank=0, nranks=1, initializer_method="uid")

    my_pe = nvshmem.core.my_pe()

    # Create shared tensors
    target = nvshmem.core.tensor((1,), dtype=torch.int32)
    results = nvshmem.core.tensor((10,), dtype=torch.int32)  # Store results from 10 threads

    # Initialize values
    init_value = 5
    target[0] = init_value  # Start with value 5
    results[:] = -1  # Initialize results with invalid values

    print(f"[PE {my_pe}] Initial target value: {target[0].item()}")

    # Launch the simple test kernel with 10 threads
    num_threads = 10
    grid_size = (num_threads,)

    print(f"[PE {my_pe}] Launching kernel with {num_threads} threads...")

    extern_libs = NVSHMEM_EXTERN_LIBS
    print(f"extern_libs: {extern_libs}")
    kernel_test_atomic_swap[grid_size](target, results, extern_libs=extern_libs)

    # Synchronize
    torch.cuda.synchronize()

    # Print results
    print(f"[PE {my_pe}] Final target value: {target[0].item()}")
    print(f"[PE {my_pe}] Return values from atomic_swap:")

    results_cpu = results.cpu().numpy()
    for i in range(num_threads):
        print(f"  Thread {i}: returned {results_cpu[i]}")

    # Analyze the results
    print(f"\n[PE {my_pe}] Analysis:")
    print(f"  Initial value: {init_value}")
    print(f"  Final value: {target[0].item()}")
    print(f"  Expected final value: {100} (last thread's new value)")

    # Check that all threads got valid return values
    # The first thread should get the original value (5)
    # Subsequent threads should get the values set by previous threads
    print(f"  Return values: {results_cpu}")

    # Verify that subsequent threads got values from previous threads
    # (This is a bit tricky to verify exactly due to race conditions,
    # but we can check that all values are reasonable)
    counts = {}
    for i in range(num_threads):
        counts[results_cpu[i]] = counts.get(results_cpu[i], 0) + 1
    assert len(counts) == 2
    assert counts[5] == 1
    assert counts[100] == num_threads - 1

    # Cleanup
    nvshmem.core.free_tensor(target)
    nvshmem.core.free_tensor(results)
    nvshmem.core.finalize()
