import nvshmem.core
import torch
import triton
import triton.language as tl
from cuda.core.experimental import Device

# Import the putmem_nbi_block function from our custom API
from odc.primitives import NVSHMEM_EXTERN_LIBS, __syncthreads, int_p, quiet, tid


@triton.jit
def int_p_test_kernel(data_ptr, value, pe):
    # Get thread ID
    cta_id = tl.program_id(axis=0)
    tidx = tid(0)

    if cta_id == 0 and tidx == 0:
        int_p(data_ptr, value, pe)
    __syncthreads()
    quiet()


def test_int_p():
    dev = Device(0)  # Use first GPU
    dev.set_current()

    # Initialize NVSHMEM with single PE using UID method
    uid = nvshmem.core.get_unique_id()
    nvshmem.core.init(device=dev, uid=uid, rank=0, nranks=1, initializer_method="uid")

    my_pe = nvshmem.core.my_pe()
    n_pes = nvshmem.core.n_pes()

    print(f"[PE {my_pe}] Running with {n_pes} PE(s)")

    data = nvshmem.core.tensor((1,), dtype=torch.int32)
    int_p_test_kernel[(1,)](data, 42, my_pe, num_warps=4, extern_libs=NVSHMEM_EXTERN_LIBS)
    torch.cuda.synchronize()
    assert data.item() == 42
    nvshmem.core.free_tensor(data)
    nvshmem.core.finalize()
