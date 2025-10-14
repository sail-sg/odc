import torch
import triton
import triton.language as tl

from odc.primitives import __syncthreads, tid


@triton.jit
def kernel_test_utils(data_ptr):
    cta_id = tl.program_id(axis=0)
    assert cta_id == 0
    thread_id = tid(0)
    assert tid(1) == 0
    assert tid(2) == 0
    thread_indices = tl.arange(0, 32)
    tl.store(data_ptr + thread_indices, thread_id)
    __syncthreads()


def test_utils():
    torch.cuda.init()
    data = torch.full((32,), -1, dtype=torch.int32, device="cuda")
    kernel_test_utils[(1,)](data, num_warps=1)
    torch.cuda.synchronize()
    expected = torch.arange(32, dtype=torch.int32, device="cuda")
    assert torch.all(data == expected), f"data: {data}, expected: {expected}"
