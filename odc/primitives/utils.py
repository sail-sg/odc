import logging
import os
from functools import reduce
from typing import List

import nvshmem.core
import torch
import triton
import triton.language as tl
from cuda.core.experimental import Device

from odc.primitives import __syncthreads, tid

logger = logging.getLogger(__name__)


# From triton_dist.utils
def init_nvshmem():
    """
    Initialize NVSHMEM on the global process group.
    The symmetric tensors are initialized on the global process group.
    When using other process groups,
    we need to convert the group rank to the global rank to use nvshmem functions.
    """
    assert "NVSHMEM_HOME" in os.environ
    current_lib_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if os.environ["NVSHMEM_HOME"] not in current_lib_paths:
        current_lib_paths.insert(0, os.environ["NVSHMEM_HOME"] + "/lib")
    os.environ["LD_LIBRARY_PATH"] = ":".join(current_lib_paths)

    logger.debug(f"init_nvshmem: {os.environ}")
    assert torch.distributed.is_initialized()
    # Extract rank, nranks from process group
    num_ranks = torch.distributed.get_world_size()
    rank_id = torch.distributed.get_rank()

    pg = torch.distributed.group.WORLD

    # Create an empty uniqueid for all ranks
    broadcast_objects = [nvshmem.core.get_unique_id(empty=rank_id != 0)]
    torch.distributed.broadcast_object_list(broadcast_objects, src=0, group=pg)
    torch.distributed.barrier(group=pg)
    nvshmem.core.init(
        device=Device(torch.cuda.current_device()),
        uid=broadcast_objects[0],
        rank=rank_id,
        nranks=num_ranks,
        initializer_method="uid",
    )

    # nvshmem.core.utils._configure_logging("DEBUG")


def nvshmem_create_tensor(shape, dtype) -> torch.Tensor:
    torch.cuda.synchronize()
    tensor = nvshmem.core.tensor(shape, dtype=dtype)
    torch.cuda.synchronize()
    return tensor


def get_same_node_tensors(tensor, rank, local_world_size) -> List[torch.Tensor]:
    def get_same_node_peer_tensor(t, peer) -> torch.Tensor:
        # avoid create tensor on the same buf again. nvshmem4py can't handle multiple reference with grace. so we handle it here.
        # https://forums.developer.nvidia.com/t/nvshmem4py-nvshmem-core-finalize-does-not-handle-everything/337979
        if peer == rank:
            return t
        return nvshmem.core.get_peer_tensor(t, peer)

    local_rank = rank % local_world_size
    rank_on_same_node_start = rank - local_rank
    rank_on_same_node_end = rank_on_same_node_start + local_world_size
    return [
        get_same_node_peer_tensor(tensor, peer)
        for peer in range(rank_on_same_node_start, rank_on_same_node_end)
    ]


def nvshmem_free_tensor_sync(tensor):
    torch.cuda.synchronize()
    nvshmem.core.free_tensor(tensor)
    torch.cuda.synchronize()


def finalize_distributed():
    nvshmem.core.finalize()


def get_odc_hybrid_group_size():
    return int(os.environ.get("ODC_HYBRID_GROUP_SIZE", get_local_world_size()))


def check_odc_hybrid_group_ranks(ranks: List[int]):
    odc_hybrid_group_size = get_odc_hybrid_group_size()
    assert len(ranks) == odc_hybrid_group_size
    min_rank = min(ranks)
    for i in range(odc_hybrid_group_size):
        assert ranks[i] == min_rank + i


class SymmBufferRegistry:
    def __init__(self):
        self.local_tensor = {}
        self.local_tensor_to_keys = {}
        self.updated = set()
        self.peer_tensors = {}
        self.allocations = []

    @classmethod
    def get_instance(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = SymmBufferRegistry()
        return cls._instance

    # we'll mark all symm buffer as dirty, and next update_symm_buffer will copy the data to the symm buffer
    def flush(self):
        self.updated.clear()

    def update_symm_buffer(self, buffer_key, values, pg: torch.distributed.ProcessGroup):
        values = values.contiguous()
        if buffer_key not in self.local_tensor:
            self.allocate_symm_buffer(buffer_key, values.shape, values.dtype, pg)

        if buffer_key not in self.updated:
            self.updated.add(buffer_key)
            self.local_tensor[buffer_key].copy_(values)
            # Make sure updated buffer is visible to all ranks
            torch.distributed.barrier()
        return self.local_tensor[buffer_key]

    def allocate_symm_buffer(self, key, shape, dtype, pg: torch.distributed.ProcessGroup):
        assert key not in self.local_tensor
        group_rank = torch.distributed.get_rank(pg)
        local_world_size = get_local_world_size()
        odc_hybrid_group_size = get_odc_hybrid_group_size()
        peer_tensors = []
        for _node_rank in range(odc_hybrid_group_size // local_world_size):
            tensor = nvshmem_create_tensor(shape, dtype)
            same_node_tensors = get_same_node_tensors(tensor, group_rank, local_world_size)
            self.allocations.append(tensor)
            peer_tensors.extend(same_node_tensors)
        assert len(peer_tensors) == odc_hybrid_group_size
        local_group_size = len(peer_tensors)
        # ranks inside hybrid group must be contiguous
        # TODO: maybe we should accept the process group as parameter in this method
        # to tell the ranks inside the hybrid group.
        local_group_rank = group_rank % local_group_size
        # tensors = nvshmem_create_tensors(shape, dtype, rank, local_world_size)
        self.local_tensor[key] = peer_tensors[local_group_rank]
        self.peer_tensors[key] = peer_tensors

        self.local_tensor_to_keys[self.local_tensor[key].data_ptr()] = key
        logger.info(
            f"Rank {torch.distributed.get_rank()} create tensor {key} with shape {shape} and dtype {dtype} and ptr {self.local_tensor[key].data_ptr()}"
        )
        return self.local_tensor[key]

    def get_local_peer_tensors(self, local_tensor):
        peer_tensors = self.get_peer_tensors(local_tensor)
        local_world_size = get_local_world_size()
        local_rank = torch.distributed.get_rank() % local_world_size
        num_nodes = torch.distributed.get_world_size() // local_world_size
        return [peer_tensors[local_rank + i * local_world_size] for i in range(num_nodes)]

    def has_key(self, key):
        return key in self.local_tensor

    def get_peer_tensors(self, local_tensor):
        # Returns tensors in the same node.
        buffer_key = self.local_tensor_to_keys[local_tensor.data_ptr()]
        return self.peer_tensors[buffer_key]

    def finalize(self):
        for t in self.allocations:
            nvshmem_free_tensor_sync(t)
        self.local_tensor.clear()
        self.local_tensor_to_keys.clear()
        self.updated.clear()
        self.peer_tensors.clear()


same_local_rank_pg = None


# TODO: support hybrid mode, where pg is only a subset of the world
def get_same_local_rank_pg(pg: torch.distributed.ProcessGroup):
    local_world_size = get_local_world_size()
    assert torch.distributed.get_world_size() == torch.distributed.get_world_size(
        group=pg
    ), "Cached AG only supports pure data parallelism"
    assert (
        local_world_size != torch.distributed.get_world_size()
    ), "No need to call this for single node"
    local_rank = torch.distributed.get_rank() % local_world_size
    global same_local_rank_pg
    if same_local_rank_pg is None:
        for i in range(local_world_size):
            ranks = [
                i + j * local_world_size
                for j in range(torch.distributed.get_world_size() // local_world_size)
            ]
            new_gp = torch.distributed.new_group(ranks=ranks, backend="nccl")
            if i == local_rank:
                same_local_rank_pg = new_gp
    assert same_local_rank_pg is not None
    return same_local_rank_pg


local_world_pg = None


def get_local_world_pg(pg: torch.distributed.ProcessGroup):
    local_world_size = get_local_world_size()
    assert torch.distributed.get_world_size() == torch.distributed.get_world_size(
        group=pg
    ), "Cached AG only supports pure data parallelism"
    rank = torch.distributed.get_rank()
    global local_world_pg
    if local_world_pg is None:
        for i in range(0, torch.distributed.get_world_size(), local_world_size):
            ranks = list(range(i, i + local_world_size))
            new_gp = torch.distributed.new_group(ranks=ranks, backend="nccl")
            if rank in ranks:
                local_world_pg = new_gp
    assert local_world_pg is not None
    return local_world_pg


def get_local_world_size():
    if "RAY_LOCAL_WORLD_SIZE" in os.environ:
        return int(os.environ["RAY_LOCAL_WORLD_SIZE"])
    else:
        return int(os.environ["LOCAL_WORLD_SIZE"])


stream = None


def get_comm_stream():
    global stream
    if stream is None:
        stream = torch.cuda.Stream()
    return stream


class BufferSplitter:
    def get_max_global_buffer_size(self):
        DEFAULT_MAX_BUFFER_SIZE = 64 * 1024 * 1024
        max_buffer_size = int(os.environ.get("ODC_MAX_BUFFER_SIZE", DEFAULT_MAX_BUFFER_SIZE))
        return max_buffer_size

    def get_global_buffer_size(self, original_buffer_shape):
        original_size = reduce(lambda x, y: x * y, original_buffer_shape)
        max_buffer_size = self.get_max_global_buffer_size()
        if max_buffer_size <= 0:
            return original_size
        buf_size = min(max_buffer_size, original_size)
        return buf_size

    def get_local_buffer_size(self, original_buffer_shape, world_size):
        original_size = reduce(lambda x, y: x * y, original_buffer_shape)
        max_buffer_size = self.get_max_global_buffer_size()
        if max_buffer_size <= 0:
            return original_size
        assert (
            max_buffer_size % world_size == 0
        ), f"ODC_MAX_BUFFER_SIZE: {max_buffer_size} % world_size: {world_size} != 0"
        local_max_buffer_size = max_buffer_size // world_size
        buf_size = min(local_max_buffer_size, original_size)
        return buf_size


@triton.jit
def sync_cta(signal_ptr, expected):
    tidx = tid(axis=0)
    if tidx == 0:
        tl.atomic_add(signal_ptr, 1)
    __syncthreads()
    offsets = tl.arange(0, 1)
    mask = offsets == 0
    r = 0
    while r < expected:
        signals = tl.load(signal_ptr + offsets, mask=mask, volatile=True)
        r = tl.max(signals)


class ProcessGroupRanksTensors:
    def __init__(self):
        self.pg_ranks_cache = {}

    def get_pg_ranks_tensor(self, pg: torch.distributed.ProcessGroup):
        if pg not in self.pg_ranks_cache:
            pg_ranks = torch.distributed.get_process_group_ranks(group=pg)
            pg_ranks_tensor = torch.tensor(pg_ranks, dtype=torch.int32, device="cuda")
            self.pg_ranks_cache[pg] = pg_ranks_tensor
        return self.pg_ranks_cache[pg]


PROCESS_GROUP_RANKS_TENSORS = ProcessGroupRanksTensors()
