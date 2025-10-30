import nvshmem.core
import torch
import triton
import triton.language as tl
from cuda.core.experimental import Device

from odc.primitives import NVSHMEM_EXTERN_LIBS, getmem_nbi_block, quiet


@triton.jit
def getmem_test_kernel(dest_ptr, source_ptr, nbytes, pe):
    cta_id = tl.program_id(axis=0)

    if cta_id == 0:
        getmem_nbi_block(dest_ptr, source_ptr, nbytes, pe)
        quiet()


def test_getmem():
    dev = Device(0)  # Use first GPU
    dev.set_current()

    # Initialize NVSHMEM with single PE using UID method
    uid = nvshmem.core.get_unique_id()
    nvshmem.core.init(device=dev, uid=uid, rank=0, nranks=1, initializer_method="uid")

    my_pe = nvshmem.core.my_pe()
    n_pes = nvshmem.core.n_pes()

    print(f"[PE {my_pe}] Running with {n_pes} PE(s)")

    # Create shared tensors
    data_size = 16  # Number of int32 elements
    nbytes = data_size * 4  # 4 bytes per int32

    source_data = nvshmem.core.tensor((data_size,), dtype=torch.int32)
    dest_data = nvshmem.core.tensor((data_size,), dtype=torch.int32)

    # Initialize source data with a pattern
    for i in range(data_size):
        source_data[i] = i * 10 + 100  # Pattern: 100, 110, 120, 130, ...

    # Initialize destination with different values
    dest_data[:] = -1  # Fill with -1 to verify the copy worked

    print(f"[PE {my_pe}] Source data: {source_data.cpu().numpy()}")
    print(f"[PE {my_pe}] Initial dest data: {dest_data.cpu().numpy()}")

    # Launch the getmem test kernel
    grid_size = (1,)  # Use single thread to avoid race conditions

    print(f"[PE {my_pe}] Launching getmem kernel...")
    print(f"[PE {my_pe}] Copying {nbytes} bytes from source on PE {my_pe} to dest locally")

    extern_libs = NVSHMEM_EXTERN_LIBS
    getmem_test_kernel[grid_size](
        dest_data, source_data, nbytes, my_pe, num_warps=1, extern_libs=extern_libs
    )

    # Synchronize to ensure the non-blocking operation completes
    torch.cuda.synchronize()

    # Print results
    print(f"[PE {my_pe}] After getmem operation:")
    print(f"[PE {my_pe}] Source data: {source_data.cpu().numpy()}")
    print(f"[PE {my_pe}] Dest data: {dest_data.cpu().numpy()}")
    assert torch.all(dest_data == source_data)

    # Verify the copy was successful
    source_cpu = source_data.cpu().numpy()
    dest_cpu = dest_data.cpu().numpy()

    print(f"\n[PE {my_pe}] Verification:")
    success = True
    for i in range(data_size):
        if source_cpu[i] != dest_cpu[i]:
            print(f"  MISMATCH at index {i}: source={source_cpu[i]}, dest={dest_cpu[i]}")
            success = False

    if success:
        print("  ✓ SUCCESS: All data copied correctly!")
    else:
        print("  ✗ FAILURE: Data mismatch detected!")

    # Test with different data sizes
    print(f"\n[PE {my_pe}] Testing with different data patterns...")

    # Test 2: Copy only first half of the data
    half_size = data_size // 2
    half_nbytes = half_size * 4

    # Reset destination
    dest_data[:] = -999

    # Modify source data
    for i in range(data_size):
        source_data[i] = i + 1000  # New pattern: 1000, 1001, 1002, ...

    print(f"[PE {my_pe}] Test 2: Copying first {half_nbytes} bytes ({half_size} elements)")

    getmem_test_kernel[grid_size](
        dest_data, source_data, half_nbytes, my_pe, extern_libs=extern_libs
    )

    torch.cuda.synchronize()
    # nvshmem.core.barrier_all()

    print(f"[PE {my_pe}] After partial copy:")
    print(f"[PE {my_pe}] Source data: {source_data.cpu().numpy()}")
    print(f"[PE {my_pe}] Dest data: {dest_data.cpu().numpy()}")
    assert torch.all(dest_data[:half_size] == source_data[:half_size])
    assert torch.all(dest_data[half_size:] == -999)

    # Verify partial copy
    source_cpu = source_data.cpu().numpy()
    dest_cpu = dest_data.cpu().numpy()

    print(f"[PE {my_pe}] Partial copy verification:")
    success = True
    for i in range(half_size):
        if source_cpu[i] != dest_cpu[i]:
            print(f"  MISMATCH at index {i}: source={source_cpu[i]}, dest={dest_cpu[i]}")
            success = False

    # Check that remaining elements are unchanged
    for i in range(half_size, data_size):
        if dest_cpu[i] != -999:
            print(f"  UNEXPECTED CHANGE at index {i}: dest={dest_cpu[i]}, expected=-999")
            success = False

    if success:
        print("  ✓ SUCCESS: Partial copy worked correctly!")
    else:
        print("  ✗ FAILURE: Partial copy failed!")

    # Test 3: Test with different data patterns and verify edge cases
    print(f"\n[PE {my_pe}] Testing edge cases...")

    # Test with single element
    single_element_size = 1
    single_nbytes = single_element_size * 4

    # Reset destination
    dest_data[:] = -777

    # Set source to a specific pattern
    source_data[0] = 9999
    for i in range(1, data_size):
        source_data[i] = i + 2000

    print(f"[PE {my_pe}] Test 3: Copying single element ({single_nbytes} bytes)")

    getmem_test_kernel[grid_size](
        dest_data, source_data, single_nbytes, my_pe, extern_libs=extern_libs
    )

    torch.cuda.synchronize()

    print(f"[PE {my_pe}] After single element copy:")
    print(f"[PE {my_pe}] Source data: {source_data.cpu().numpy()}")
    print(f"[PE {my_pe}] Dest data: {dest_data.cpu().numpy()}")

    # Verify single element copy
    source_cpu = source_data.cpu().numpy()
    dest_cpu = dest_data.cpu().numpy()

    print(f"[PE {my_pe}] Single element copy verification:")
    success = True
    if source_cpu[0] != dest_cpu[0]:
        print(f"  MISMATCH at index 0: source={source_cpu[0]}, dest={dest_cpu[0]}")
        success = False

    # Check that remaining elements are unchanged
    for i in range(1, data_size):
        if dest_cpu[i] != -777:
            print(f"  UNEXPECTED CHANGE at index {i}: dest={dest_cpu[i]}, expected=-777")
            success = False

    if success:
        print("  ✓ SUCCESS: Single element copy worked correctly!")
    else:
        print("  ✗ FAILURE: Single element copy failed!")

    # Cleanup
    nvshmem.core.free_tensor(source_data)
    nvshmem.core.free_tensor(dest_data)
    nvshmem.core.finalize()


if __name__ == "__main__":
    test_getmem()
