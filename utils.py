import torch
import nvshmem
from typing import List


# From triton_dist.utils
def init_nvshmem_by_torch_process_group(pg: torch.distributed.ProcessGroup):
    # Extract rank, nranks from process group
    num_ranks = pg.size()
    rank_id = pg.rank()

    # Create an empty uniqueid for all ranks
    broadcast_objects = [nvshmem.core.get_unique_id(empty=rank_id != 0)]
    torch.distributed.broadcast_object_list(broadcast_objects, src=0, group=pg)
    torch.distributed.barrier(group=pg)
    from cuda.core.experimental import Device
    nvshmem.core.init(device=Device(torch.cuda.current_device()), uid=broadcast_objects[0], rank=rank_id,
                      nranks=num_ranks, initializer_method="uid")
    # nvshmem.core.utils._configure_logging("DEBUG")


def nvshmem_create_tensor(shape, dtype) -> torch.Tensor:
    torch.cuda.synchronize()
    tensor = nvshmem.core.tensor(shape, dtype=dtype)
    torch.cuda.synchronize()
    return tensor


def nvshmem_create_tensors(shape, dtype, rank, local_world_size, local_buffer=None) -> List[torch.Tensor]:

    def get_peer_tensor(t, peer) -> torch.Tensor:
        # avoid create tensor on the same buf again. nvshmem4py can't handle multiple reference with grace. so we handle it here.
        # https://forums.developer.nvidia.com/t/nvshmem4py-nvshmem-core-finalize-does-not-handle-everything/337979
        if peer == rank:
            return t
        return nvshmem.core.get_peer_tensor(t, peer)

    local_rank = rank % local_world_size
    rank_on_same_node_start = rank - local_rank
    rank_on_same_node_end = rank_on_same_node_start + local_world_size
    torch.cuda.synchronize()
    if local_buffer is None:
        tensor = nvshmem_create_tensor(shape, dtype=dtype)
    else:
        tensor = local_buffer
    torch.cuda.synchronize()
    return [get_peer_tensor(tensor, peer) for peer in range(rank_on_same_node_start, rank_on_same_node_end)]


def nvshmem_free_tensor_sync(tensor):
    torch.cuda.synchronize()
    nvshmem.core.free_tensor(tensor)
    torch.cuda.synchronize()


def finalize_distributed():
    nvshmem.core.finalize()


def init_nvshmem():
    assert torch.distributed.is_initialized()
    init_nvshmem_by_torch_process_group(torch.distributed.group.WORLD)


class SymmBufferRegistry:
    def __init__(self):
        self.local_tensor = {}
        self.local_tensor_to_keys = {}
        self.updated = set()
        self.peer_tensors = {}
        self.cached_local_buffer = {}

    @classmethod
    def get_instance(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = SymmBufferRegistry()
        return cls._instance
    
    # we'll mark all symm buffer as dirty, and next update_symm_buffer will copy the data to the symm buffer
    def flush(self):
        
        self.updated.clear()

    def update_symm_buffer(self, buffer_key, values):
        values = values.contiguous()
        if buffer_key not in self.local_tensor:
            self.allocate_symm_buffer(buffer_key, values.shape, values.dtype)

        if buffer_key not in self.updated:
            self.updated.add(buffer_key)
            self.local_tensor[buffer_key].copy_(values)
            # Make sure updated buffer is visible to all ranks
            torch.distributed.barrier()
        return self.local_tensor[buffer_key]
  
    def get_cached_local_buffer(self, shape, dtype):
        key = (tuple(shape), dtype)
        if key not in self.cached_local_buffer:
            self.cached_local_buffer[key] = nvshmem_create_tensor(shape, dtype=dtype)
        return self.cached_local_buffer[key]

    def allocate_symm_buffer(self, key, shape, dtype, local_single_buffer=False):
        assert key not in self.local_tensor
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        local_buffer = None
        if local_single_buffer:
            local_buffer = self.get_cached_local_buffer(shape, dtype)
        self.peer_tensors[key] = nvshmem_create_tensors(shape, dtype, rank, world_size, local_buffer)
        self.local_tensor[key] = self.peer_tensors[key][rank]
        assert self.local_tensor[key].data_ptr() == local_buffer.data_ptr() if local_buffer is not None else True
        self.local_tensor_to_keys[self.local_tensor[key].data_ptr()] = key
        print(f"Rank {torch.distributed.get_rank()} create tensor {key} with shape {shape} and dtype {dtype} and ptr {self.local_tensor[key].data_ptr()}")
        return self.local_tensor[key]
    
    def has_key(self, key):
        return key in self.local_tensor
    
    def get_peer_tensors(self, local_tensor):
        buffer_key = self.local_tensor_to_keys[local_tensor.data_ptr()]
        return self.peer_tensors[buffer_key]
    
    def finalize(self):
        cached_local_buffer = set(self.cached_local_buffer.values())
        for buffer_key in self.local_tensor:
            if self.local_tensor[buffer_key] not in cached_local_buffer:
                nvshmem_free_tensor_sync(self.local_tensor[buffer_key])
        for buffer in cached_local_buffer:
            nvshmem_free_tensor_sync(buffer)
        self.cached_local_buffer.clear()
        self.local_tensor.clear()
        self.local_tensor_to_keys.clear()
        self.updated.clear()
        self.peer_tensors.clear()