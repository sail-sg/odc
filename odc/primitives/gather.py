import logging
import math

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch import Tensor

from odc.primitives import NVSHMEM_EXTERN_LIBS, __syncthreads, getmem_nbi_block, quiet, tid
from odc.primitives.utils import (
    BufferSplitter,
    SymmBufferRegistry,
    get_comm_stream,
    get_local_world_size,
    sync_cta,
)

logger = logging.getLogger(__name__)


@triton.jit
def nvshmem_device_producer_gather_2d_get_block_kernel_chunked_synced(
    remote_tensor_ptr,
    target_tensor_ptr,
    elem_per_rank,
    size_per_elem,
    rank,
    num_ranks_per_node,
    world_size,
    chunk_size,
    signal_ptr,
):
    pid = tl.program_id(axis=0)
    # np = tl.num_programs(axis=0)
    assert num_ranks_per_node == tl.num_programs(axis=0)
    np = num_ranks_per_node
    num_nodes = world_size // np

    tidx = tid(axis=0)
    expected = 0
    for i in range(1, num_nodes):
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
            expected += np
            sync_cta(signal_ptr, expected)
            if tidx == 0 and pid == 0:
                quiet()
            __syncthreads()

            expected += np
            sync_cta(signal_ptr, expected)


class GatherService:
    def __init__(self):
        self.shaped_buffer = {}
        self.buffer_splitter = BufferSplitter()
        self.chunk_size_bytes = 2**20
        self._shaped_buffer_memory_bytes = 0
        self._shaped_buffer_count = 0

    def get_chunk_size(self, buffer_dtype):
        return self.chunk_size_bytes // buffer_dtype.itemsize

    def gather_into_tensor(
        self, output_tensor: Tensor, input_tensor: Tensor, pg: dist.ProcessGroup
    ):
        buf_size = self.buffer_splitter.get_global_buffer_size(output_tensor.shape)
        buffer_shape = (buf_size,)
        output_size = output_tensor.numel()
        assert output_size >= buf_size, f"output_size: {output_size} < buf_size: {buf_size}"

        rank = torch.distributed.get_rank()
        new_shaped_buffer = False
        if (buffer_shape, output_tensor.dtype) not in self.shaped_buffer:
            logger.info(
                f"Rank {rank} create buffer: output_size: {output_size} num_sub_buffers: {math.ceil(output_size / buf_size)} buf_size: {buf_size}"
            )
            self.shaped_buffer[
                (buffer_shape, output_tensor.dtype)
            ] = SymmBufferRegistry.get_instance().allocate_symm_buffer(
                f"ag_buffer_{buffer_shape}_{output_tensor.dtype}",
                buffer_shape,
                output_tensor.dtype,
            )
            new_shaped_buffer = True
        target_tensor = self.shaped_buffer[(buffer_shape, output_tensor.dtype)]
        if new_shaped_buffer:
            shaped_bytes = target_tensor.numel() * target_tensor.element_size()
            self._shaped_buffer_memory_bytes += shaped_bytes
            self._shaped_buffer_count += 1
            logger.info(
                f"[ODC] Gather shaped buffer allocated: shape={buffer_shape}, dtype={output_tensor.dtype}, "
                f"size={shaped_bytes / 1e6:.2f}MB"
            )
            logger.info(
                f"[ODC] Gather shaped buffer totals: count={self._shaped_buffer_count}, "
                f"bytes={self._shaped_buffer_memory_bytes} "
                f"({self._shaped_buffer_memory_bytes / 1e6:.2f}MB)"
            )

        assert (input_tensor.numel() * input_tensor.element_size()) % (
            2**6
        ) == 0 or input_tensor.numel() < 2**6, "better align to 64 for efficiency"
        chunk_size = self.get_chunk_size(input_tensor.dtype)
        # assert input_tensor.numel() % chunk_size == 0

        registry = SymmBufferRegistry.get_instance()
        peer_tensors = registry.get_peer_tensors(input_tensor)

        group_world_size = torch.distributed.get_world_size(pg)
        local_world_size = get_local_world_size()
        assert group_world_size in (
            torch.distributed.get_world_size(),
            local_world_size,
        ), f"{group_world_size=} {torch.distributed.get_world_size()=} {local_world_size=}"

        get_comm_stream().wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(get_comm_stream()):
            output_tensor_split = output_tensor.view(group_world_size, -1)
            assert local_world_size == len(peer_tensors)
            local_rank = rank % local_world_size
            rank_same_node_start = rank - local_rank
            rank_same_node_end = rank_same_node_start + local_world_size
            for r_offset in range(local_world_size):
                src_local_rank = (local_rank + r_offset) % local_world_size
                if group_world_size == local_world_size:
                    output_tensor_split[src_local_rank].copy_(peer_tensors[src_local_rank])
                else:
                    src_rank = rank_same_node_start + src_local_rank
                    output_tensor_split[src_rank].copy_(peer_tensors[src_local_rank])

            assert buf_size % group_world_size == 0
            local_buf_size = buf_size // group_world_size
            signal_ptr = torch.empty(1, dtype=torch.int32, device="cuda")
            for start in range(0, input_tensor.numel(), local_buf_size):
                if local_world_size == group_world_size:
                    continue
                size = min(local_buf_size, input_tensor.numel() - start)
                sub_input_tensor = input_tensor.view(-1)[start : start + size]
                assert (sub_input_tensor.numel() * sub_input_tensor.element_size()) % (
                    2**6
                ) == 0 or sub_input_tensor.numel() < 2**6, "better align to 64 for efficiency"
                target_buf_size = size * group_world_size
                assert target_buf_size <= buf_size
                target_tensor_split = target_tensor[:target_buf_size].view(group_world_size, size)

                signal_ptr.fill_(0)
                assert group_world_size % 8 == 0 or group_world_size < 8
                # grid_size = 8 if world_size == 32 else world_size
                grid_size = local_world_size
                nvshmem_device_producer_gather_2d_get_block_kernel_chunked_synced[(grid_size,)](
                    remote_tensor_ptr=sub_input_tensor,
                    target_tensor_ptr=target_tensor_split.view(-1),
                    elem_per_rank=sub_input_tensor.numel(),
                    size_per_elem=sub_input_tensor.element_size(),
                    rank=rank,
                    num_ranks_per_node=local_world_size,
                    world_size=group_world_size,
                    chunk_size=chunk_size,
                    signal_ptr=signal_ptr,
                    num_warps=32,
                    extern_libs=NVSHMEM_EXTERN_LIBS,
                )
                if buf_size == output_size:
                    local_world_data_size = size * local_world_size
                    local_world_idx = rank // local_world_size
                    data_start_idx = local_world_data_size * local_world_idx
                    data_end_idx = data_start_idx + local_world_data_size
                    output_tensor[:data_start_idx].copy_(target_tensor[:data_start_idx])
                    output_tensor[data_end_idx:].copy_(target_tensor[data_end_idx:])
                    # output_tensor.copy_(target_tensor)
                else:
                    for r in range(group_world_size):
                        if rank_same_node_start <= r < rank_same_node_end:
                            continue
                        output_tensor_split[r, start : start + size].copy_(
                            target_tensor_split[r, :]
                        )
        torch.cuda.current_stream().wait_stream(get_comm_stream())
