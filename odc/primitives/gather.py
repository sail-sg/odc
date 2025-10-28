import logging
import math

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch import Tensor

from odc.primitives import NVSHMEM_EXTERN_LIBS, __syncthreads, getmem_nbi_block, quiet, tid
from odc.primitives.utils import (
    PROCESS_GROUP_RANKS_TENSORS,
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
    group_rank,
    num_ranks_per_node,
    group_world_size,
    pg_ranks_ptr,
    chunk_size,
    signal_ptr,
):
    pid = tl.program_id(axis=0)
    # np = tl.num_programs(axis=0)
    assert num_ranks_per_node == tl.num_programs(axis=0)
    np = num_ranks_per_node
    num_nodes = group_world_size // np

    tidx = tid(axis=0)
    expected = 0
    for i in range(1, num_nodes):
        peer_node = (i + group_rank // np) % num_nodes
        peer = (pid + peer_node * np) % group_world_size
        # chunk_size = elem_per_rank // num_chunks
        num_chunks = tl.cdiv(elem_per_rank, chunk_size)
        for chunk in range(num_chunks):
            this_chunk_size = chunk_size
            if chunk == num_chunks - 1:
                this_chunk_size = elem_per_rank - chunk * chunk_size
            global_peer = tl.load(pg_ranks_ptr + peer)
            getmem_nbi_block(
                target_tensor_ptr + peer * elem_per_rank + (chunk * chunk_size),
                remote_tensor_ptr + (chunk * chunk_size),
                this_chunk_size * size_per_elem,
                global_peer,
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

    def gather_into_tensor(
        self, output_tensor: Tensor, input_tensor: Tensor, pg: dist.ProcessGroup
    ):
        buf_size = self.buffer_splitter.get_global_buffer_size(output_tensor.shape)
        buffer_shape = (buf_size,)
        output_size = output_tensor.numel()
        assert output_size >= buf_size, f"output_size: {output_size} < buf_size: {buf_size}"

        group_rank = torch.distributed.get_rank(group=pg)
        if (buffer_shape, output_tensor.dtype) not in self.shaped_buffer:
            logger.info(
                f"Rank {torch.distributed.get_rank()} create buffer: output_size: {output_size} num_sub_buffers: {math.ceil(output_size / buf_size)} buf_size: {buf_size}"
            )
            self.shaped_buffer[
                (buffer_shape, output_tensor.dtype)
            ] = SymmBufferRegistry.get_instance().allocate_symm_buffer(
                f"ag_buffer_{buffer_shape}_{output_tensor.dtype}",
                buffer_shape,
                output_tensor.dtype,
                group_rank,
            )
        target_tensor = self.shaped_buffer[(buffer_shape, output_tensor.dtype)]

        assert (input_tensor.numel() * input_tensor.element_size()) % (
            2**6
        ) == 0 or input_tensor.numel() < 2**6, "better align to 64 for efficiency"
        chunk_size = 2**20 // input_tensor.element_size()
        # assert input_tensor.numel() % chunk_size == 0

        pg_ranks_tensor = PROCESS_GROUP_RANKS_TENSORS.get_pg_ranks_tensor(pg)
        registry = SymmBufferRegistry.get_instance()
        peer_tensors = registry.get_peer_tensors(input_tensor)

        group_world_size = torch.distributed.get_world_size(pg)
        get_comm_stream().wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(get_comm_stream()):
            output_tensor_split = output_tensor.view(group_world_size, -1)
            local_world_size = get_local_world_size()
            assert local_world_size == len(peer_tensors)
            local_rank = group_rank % local_world_size
            rank_same_node_start = group_rank - local_rank
            rank_same_node_end = rank_same_node_start + local_world_size
            for r_offset in range(local_world_size):
                src_local_rank = (local_rank + r_offset) % local_world_size
                src_group_rank = rank_same_node_start + src_local_rank
                output_tensor_split[src_group_rank].copy_(peer_tensors[src_local_rank])

            assert buf_size % group_world_size == 0
            local_buf_size = buf_size // group_world_size
            signal_ptr = torch.empty(1, dtype=torch.int32, device="cuda")
            for start in range(0, input_tensor.numel(), local_buf_size):
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
                grid_size = get_local_world_size()
                # logger.warning(f"Rank {torch.distributed.get_rank()} group_rank: {group_rank} group_size: {group_world_size} grid_size: {grid_size}")
                nvshmem_device_producer_gather_2d_get_block_kernel_chunked_synced[(grid_size,)](
                    remote_tensor_ptr=sub_input_tensor,
                    target_tensor_ptr=target_tensor_split.view(-1),
                    elem_per_rank=sub_input_tensor.numel(),
                    size_per_elem=sub_input_tensor.element_size(),
                    group_rank=group_rank,
                    num_ranks_per_node=get_local_world_size(),
                    group_world_size=group_world_size,
                    pg_ranks_ptr=pg_ranks_tensor,
                    chunk_size=chunk_size,
                    signal_ptr=signal_ptr,
                    num_warps=32,
                    extern_libs=NVSHMEM_EXTERN_LIBS,
                )
                if buf_size == output_size:
                    local_world_data_size = size * local_world_size
                    local_world_idx = group_rank // local_world_size
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
