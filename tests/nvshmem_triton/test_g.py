import nvshmem.core
import torch
import triton
import triton.language as tl
from cuda.core.experimental import Device

from odc.primitives import BC_PATH, __syncthreads, int_g, quiet, tid


@triton.jit
def int_g_test_kernel(src_ptr, dst_ptr, pe):
    # Get thread ID
    cta_id = tl.program_id(axis=0)
    tidx = tid(0)

    if cta_id == 0 and tidx == 0:
        int_g(src_ptr, pe)
        value = tl.load(src_ptr)
        # Store the result back to the destination pointer for verification
        tl.store(dst_ptr, value)
    __syncthreads()
    quiet()


def test_int_g():
    dev = Device(0)  # Use first GPU
    dev.set_current()

    # Initialize NVSHMEM with single PE using UID method
    uid = nvshmem.core.get_unique_id()
    nvshmem.core.init(device=dev, uid=uid, rank=0, nranks=1, initializer_method="uid")

    my_pe = nvshmem.core.my_pe()
    n_pes = nvshmem.core.n_pes()

    print(f"[PE {my_pe}] Running with {n_pes} PE(s)")

    # Create a tensor and initialize it with a test value
    data = nvshmem.core.tensor((1,), dtype=torch.int32)
    data.fill_(42)  # Initialize with value 42
    dest = nvshmem.core.tensor((1,), dtype=torch.int32)
    dest.fill_(-1)

    # Run the kernel that uses int_g to get the value
    int_g_test_kernel[(1,)](
        data, dest, my_pe, num_warps=4, extern_libs={"nvshmem_wrapper": BC_PATH}
    )
    torch.cuda.synchronize()

    # Verify the value was correctly retrieved
    assert data.item() == 42
    assert dest.item() == 42
    nvshmem.core.free_tensor(data)
    nvshmem.core.free_tensor(dest)
    nvshmem.core.finalize()
