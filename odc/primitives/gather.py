import math
from collections import defaultdict
from typing import List

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch import Tensor

from odc.primitives import __syncthreads, getmem_nbi_block, quiet, tid, NVSHMEM_EXTERN_LIBS
from odc.primitives.utils import (
    BufferSplitter,
    SymmBufferRegistry,
    get_comm_stream,
    get_local_world_pg,
    get_local_world_size,
    get_same_local_rank_pg,
    init_nvshmem,
)


@triton.jit
def nvshmem_device_producer_all_gather_2d_get_block_kernel_chunked_synced(
    remote_tensor_ptr,
    target_tensor_ptr,
    elem_per_rank,
    size_per_elem,
    rank,
    world_size,
    chunk_size,
    signal_ptr,
):
    pid = tl.program_id(axis=0)
    np = tl.num_programs(axis=0)
    num_nodes = world_size // np

    tidx = tid(axis=0)
    expected = 0
    for i in range(num_nodes):
        peer_node = (i + rank // np) % num_nodes
        peer = (pid + peer_node * np) % world_size
        # chunk_size = elem_per_rank // num_chunks
        num_chunks = tl.cdiv(elem_per_rank, chunk_size)
        for chunk in range(num_chunks):
            this_chunk_size = chunk_size
            if chunk == num_chunks - 1:
                this_chunk_size = elem_per_rank - chunk * chunk_size
            getmem_nbi_block(
                target_tensor_ptr + peer * elem_per_rank + (chunk * chunk_size),
                remote_tensor_ptr + (chunk * chunk_size),
                this_chunk_size * size_per_elem,
                peer,
            )
            if tidx == 0:
                tl.atomic_add(signal_ptr, 1)
            __syncthreads()
            expected += np
            offsets = tl.arange(0, 1)
            mask = offsets == 0
            r = 0
            while r < expected:
                signals = tl.load(signal_ptr + offsets, mask=mask, volatile=True)
                r = tl.max(signals)
            if tidx == 0 and pid == 0:
                quiet()
            __syncthreads()

            if tidx == 0:
                tl.atomic_add(signal_ptr, 1)
            __syncthreads()
            expected += np
            offsets = tl.arange(0, 1)
            mask = offsets == 0
            r = 0
            while r < expected:
                signals = tl.load(signal_ptr + offsets, mask=mask, volatile=True)
                r = tl.max(signals)


shaped_buffer = {}
buffer_splitter = BufferSplitter()


def all_gather_into_tensor(output_tensor: Tensor, input_tensor: Tensor, pg: dist.ProcessGroup):
    buf_size = buffer_splitter.get_global_buffer_size(output_tensor.shape)
    buffer_shape = (buf_size,)
    output_size = output_tensor.numel()
    assert output_size >= buf_size, f"output_size: {output_size} < buf_size: {buf_size}"

    if (buffer_shape, output_tensor.dtype) not in shaped_buffer:
        print(
            f"Rank {torch.distributed.get_rank()} create buffer: output_size: {output_size} num_sub_buffers: {math.ceil(output_size / buf_size)} buf_size: {buf_size}"
        )
        shaped_buffer[
            (buffer_shape, output_tensor.dtype)
        ] = SymmBufferRegistry.get_instance().allocate_symm_buffer(
            f"ag_buffer_{buffer_shape}_{output_tensor.dtype}", buffer_shape, output_tensor.dtype
        )
    target_tensor = shaped_buffer[(buffer_shape, output_tensor.dtype)]

    # peers = SymmBufferRegistry.get_instance().get_peer_tensors(input_tensor)
    grid = (torch.distributed.get_world_size(pg),)
    # print(f"Rank {torch.distributed.get_rank(pg)} grid: {grid}")
    assert (input_tensor.numel() * input_tensor.element_size()) % (
        2**6
    ) == 0, "better align to 64 for efficiency"
    chunk_size = 2**20 // input_tensor.element_size()
    # assert input_tensor.numel() % chunk_size == 0

    world_size = torch.distributed.get_world_size(pg)
    get_comm_stream().wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(get_comm_stream()):
        output_tensor_split = output_tensor.view(world_size, -1)
        assert buf_size % world_size == 0
        local_buf_size = buf_size // world_size
        signal_ptr = torch.empty(1, dtype=torch.int32, device="cuda")
        for start in range(0, input_tensor.numel(), local_buf_size):
            size = min(local_buf_size, input_tensor.numel() - start)
            sub_input_tensor = input_tensor.view(-1)[start : start + size]
            assert (sub_input_tensor.numel() * sub_input_tensor.element_size()) % (
                2**6
            ) == 0, "better align to 64 for efficiency"
            target_buf_size = size * world_size
            assert target_buf_size <= buf_size
            target_tensor_split = target_tensor[:target_buf_size].view(world_size, size)
            signal_ptr.fill_(0)
            assert world_size % 8 == 0 or world_size < 8
            grid_size = 8 if world_size == 32 else world_size
            nvshmem_device_producer_all_gather_2d_get_block_kernel_chunked_synced[(grid_size,)](
                sub_input_tensor,
                target_tensor_split.view(-1),
                sub_input_tensor.numel(),
                sub_input_tensor.element_size(),
                torch.distributed.get_rank(pg),
                torch.distributed.get_world_size(pg),
                chunk_size,
                signal_ptr,
                num_warps=32,
                extern_libs=NVSHMEM_EXTERN_LIBS,
            )
            if buf_size == output_size:
                output_tensor.copy_(target_tensor)
            else:
                for r in range(world_size):
                    output_tensor_split[r, start : start + size].copy_(target_tensor_split[r, :])
    torch.cuda.current_stream().wait_stream(get_comm_stream())
