import torch
import nvshmem
import os
from typing import List


# From triton_dist.utils
def init_nvshmem():
    # print(f"init_nvshmem: {os.environ}")
    assert torch.distributed.is_initialized()
    # Extract rank, nranks from process group
    num_ranks = torch.distributed.get_world_size()
    rank_id = torch.distributed.get_rank()
    local_world_size = get_local_world_size()
    local_rank = rank_id % get_local_world_size()

    # Create an empty uniqueid for all ranks

    from cuda.core.experimental import Device

    # broadcast_objects = [nvshmem.core.get_unique_id(empty=rank_id != 0)]
    # torch.distributed.broadcast_object_list(uid, src=0, group=pg)
    # torch.distributed.barrier(group=pg)
    # nvshmem.core.init(device=Device(torch.cuda.current_device()), uid=uid[0], rank=rank_id,
    #                   nranks=num_ranks, initializer_method="uid")
    
    
    # TODO: This is a hack as currently nvshmem doesn't work cross node. So we init nvshmem only within node.
    all_gather_objects = [nvshmem.core.get_unique_id(empty=local_rank != 0)]
    res = [None for _ in range(num_ranks)]
    torch.distributed.all_gather_object(res, all_gather_objects, group=torch.distributed.group.WORLD)
    torch.distributed.barrier(group=torch.distributed.group.WORLD)
    uid = res[rank_id // local_world_size * local_world_size]
    nvshmem.core.init(device=Device(torch.cuda.current_device()), uid=uid[0], rank=local_rank,
                      nranks=local_world_size, initializer_method="uid")
    

    
    
    
    # nvshmem.core.utils._configure_logging("DEBUG")


def nvshmem_create_tensor(shape, dtype) -> torch.Tensor:
    torch.cuda.synchronize()
    tensor = nvshmem.core.tensor(shape, dtype=dtype)
    torch.cuda.synchronize()
    return tensor


def nvshmem_create_tensors(shape, dtype, rank, local_world_size) -> List[torch.Tensor]:

    def get_peer_tensor(t, peer) -> torch.Tensor:
        # avoid create tensor on the same buf again. nvshmem4py can't handle multiple reference with grace. so we handle it here.
        # https://forums.developer.nvidia.com/t/nvshmem4py-nvshmem-core-finalize-does-not-handle-everything/337979
        if peer == rank:
            return t
        # return nvshmem.core.get_peer_tensor(t, peer)
        # TODO: This is a hack as currently nvshmem doesn't work cross node. So we init nvshmem only within node.
        return nvshmem.core.get_peer_tensor(t, peer % local_world_size)

    local_rank = rank % local_world_size
    rank_on_same_node_start = rank - local_rank
    rank_on_same_node_end = rank_on_same_node_start + local_world_size
    torch.cuda.synchronize()
    tensor = nvshmem_create_tensor(shape, dtype=dtype)
    torch.cuda.synchronize()
    return [get_peer_tensor(tensor, peer) for peer in range(rank_on_same_node_start, rank_on_same_node_end)]


def nvshmem_free_tensor_sync(tensor):
    torch.cuda.synchronize()
    nvshmem.core.free_tensor(tensor)
    torch.cuda.synchronize()


def finalize_distributed():
    nvshmem.core.finalize()


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
  
    def allocate_symm_buffer(self, key, shape, dtype):
        assert key not in self.local_tensor
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        local_world_size = get_local_world_size()
        local_rank = rank % local_world_size
        peer_tensors = []
        for node_rank in range(world_size // local_world_size):
            tensors = nvshmem_create_tensors(shape, dtype, rank, local_world_size)
            self.allocations.append(tensors[local_rank])
            peer_tensors.extend(tensors)
        assert len(peer_tensors) == world_size
        self.peer_tensors[key] = peer_tensors
        self.local_tensor[key] = self.peer_tensors[key][rank]
        self.local_tensor_to_keys[self.local_tensor[key].data_ptr()] = key
        # print(f"Rank {torch.distributed.get_rank()} create tensor {key} with shape {shape} and dtype {dtype} and ptr {self.local_tensor[key].data_ptr()}")
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
    assert torch.distributed.get_world_size() == torch.distributed.get_world_size(group=pg), "Cached AG only supports pure data parallelism"
    assert local_world_size != torch.distributed.get_world_size(), "No need to call this for single node"
    local_rank = torch.distributed.get_rank() % local_world_size
    global same_local_rank_pg
    if same_local_rank_pg is None:
      for i in range(local_world_size):
         ranks = [i + j * local_world_size for j in range(torch.distributed.get_world_size() // local_world_size)]
         new_gp = torch.distributed.new_group(ranks=ranks, backend="nccl")
         if i == local_rank:
            same_local_rank_pg = new_gp
    assert same_local_rank_pg is not None
    return same_local_rank_pg

local_world_pg = None
def get_local_world_pg(pg: torch.distributed.ProcessGroup):
    local_world_size = get_local_world_size()
    assert torch.distributed.get_world_size() == torch.distributed.get_world_size(group=pg), "Cached AG only supports pure data parallelism"
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
    
    
    