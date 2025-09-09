"""
This example initializes NVSHMEM4Py with the `torchrun` 
launcher and torch.distributed

It runs a kernel expressed with Triton

Run this program with `torchrun --nproc-per-node <NGPUs> torch_triton_interop.py`
"""

import torch.distributed as dist
import torch
import triton
import triton.language as tl
import nvshmem.core
import os
from typing import List
from torch import Tensor
from triton_dist.language.extra import libshmem_device
from odc.utils import SymmBufferRegistry, init_nvshmem, get_same_local_rank_pg, get_local_world_size, get_local_world_pg
###
#  Helper code from https://github.com/NVIDIA/cuda-python/blob/main/cuda_core/examples/pytorch_example.py
#  Used to extract PyTorch Stream into a cuda.core.Stream for NVSHMEM APIs
###


def all_gather_into_tensor(output_tensor: Tensor, input_tensor: Tensor, pg: dist.ProcessGroup):
    assert len(output_tensor.shape) == 1
    assert len(input_tensor.shape) == 1

    registry = SymmBufferRegistry.get_instance()
    # print(f"Rank {torch.distributed.get_rank()} input_tensor: dtype {input_tensor.dtype}, device {input_tensor.device} shape {input_tensor.shape} ptr {input_tensor.data_ptr()}")
    peer_tensors = registry.get_peer_tensors(input_tensor)

    # All ranks are global ranks for usage in nvshmem
    ranks = dist.get_process_group_ranks(group=pg)
    # print(f"Ranks: {ranks} on {dist.get_rank()}")
    group_idx = torch.distributed.get_rank(group=pg)

    size = input_tensor.numel()

    output_tensor_views = [
        output_tensor[r * size:(r + 1) * size] for r in range(len(ranks))
    ]

    for r_offset in range(len(ranks)):
        src_group_rank = (group_idx + r_offset) % len(ranks)
        src_rank = ranks[src_group_rank]
        output_tensor_views[src_group_rank].copy_(peer_tensors[src_rank])
    return output_tensor

def all_gather_into_tensor_nccl_comm(output_tensor: Tensor, input_tensor: Tensor, pg: dist.ProcessGroup):
    assert len(output_tensor.shape) == 1
    assert len(input_tensor.shape) == 1

    registry = SymmBufferRegistry.get_instance()
    # print(f"Rank {torch.distributed.get_rank()} input_tensor: dtype {input_tensor.dtype}, device {input_tensor.device} shape {input_tensor.shape} ptr {input_tensor.data_ptr()}")
    local_peer_tensors = registry.get_local_peer_tensors(input_tensor)

    size = input_tensor.numel()

    # output_tensor_views = [
    #     output_tensor[r * size:(r + 1) * size] for r in range(torch.distributed.get_world_size(group=pg))
    # ]

    local_world_pg = get_local_world_pg(pg)
    local_world_size = torch.distributed.get_world_size(group=local_world_pg)
    assert len(local_peer_tensors) * local_world_size == torch.distributed.get_world_size(group=pg)
    for i in range(0, torch.distributed.get_world_size(group=pg), local_world_size):
      dst_tensor = output_tensor[i * size:(i + local_world_size) * size]
      src_tensor = local_peer_tensors[i // local_world_size]
      torch.distributed.all_gather_into_tensor(dst_tensor, src_tensor, group=local_world_pg)
  

@triton.jit(do_not_specialize=["chunk", "total_chunks"])
def nvshmem_device_producer_all_gather_2d_get_block_kernel(
    remote_tensor_ptr,
    target_tensor_ptr,
    elem_per_rank,
    size_per_elem,
    rank,
    world_size,
    chunk,
    total_chunks,
):
    pid = tl.program_id(axis=0)
    if pid < world_size:
        np = tl.num_programs(axis=0)
        peer = (pid  + rank + 1) % world_size
        for c in range(total_chunks):
            libshmem_device.getmem_nbi_block(
                target_tensor_ptr + peer * elem_per_rank + (c * elem_per_rank // total_chunks),
                remote_tensor_ptr + (c * elem_per_rank // total_chunks),
                elem_per_rank * size_per_elem // total_chunks,
                peer,
            )
        libshmem_device.quiet()

@triton.jit
def nvshmem_device_producer_all_gather_2d_get_block_kernel_chunked(
    remote_tensor_ptr,
    target_tensor_ptr,
    elem_per_rank,
    size_per_elem,
    rank,
    world_size,
    chunk_size,
):
    pid = tl.program_id(axis=0)
    peer = (pid + rank + 1) % world_size
    # chunk_size = elem_per_rank // num_chunks
    num_chunks = elem_per_rank // chunk_size
    
    for chunk in range(num_chunks):
        libshmem_device.getmem_block(
            target_tensor_ptr + peer * elem_per_rank + (chunk * chunk_size),
            remote_tensor_ptr + (chunk * chunk_size),
            chunk_size * size_per_elem,
            peer,
        )

@triton.jit
def nvshmem_device_producer_all_gather_2d_put_block_kernel(
    remote_tensor_ptr,
    target_tensor_ptr,
    elem_per_rank,
    size_per_elem,
    rank,
    world_size,
):
    pid = tl.program_id(axis=0)
    if pid < world_size:
        np = tl.num_programs(axis=0)
        peer = (pid  + rank + 1) % world_size

        libshmem_device.putmem_block(
            target_tensor_ptr + rank * elem_per_rank,
            remote_tensor_ptr,
            elem_per_rank * size_per_elem,
            peer,
        )

# @triton.jit
# def nvshmem_device_producer_all_gather_2d_get_block_kernel(
#     remote_tensor_ptr,
#     target_tensor_ptr,
#     elem_per_rank,
#     size_per_elem,
#     local_rank,
#     world_size,
# ):
#     pid = tl.program_id(axis=0)
#     np = tl.num_programs(axis=0)
#     peer = (local_rank + pid * world_size // np) % world_size
#     blocks_per_rank = np // world_size
#     offset = (pid  + local_rank * blocks_per_rank) % np
#     offset_remote = pid % blocks_per_rank

#     size = elem_per_rank //blocks_per_rank
#     libshmem_device.getmem_nbi_block(
#         target_tensor_ptr + offset * size,
#         remote_tensor_ptr + offset_remote * size,
#         size * size_per_elem,
#         peer,
#     )

shaped_buffer = {}
class PyTorchStreamWrapper:
    def __init__(self, pt_stream):
        self.pt_stream = pt_stream
        self.handle = pt_stream.cuda_stream

    def __cuda_stream__(self):
        stream_id = self.pt_stream.cuda_stream
        return (0, stream_id)  # Return format required by CUDA Python

def all_gather_into_tensor_multinode(output_tensor: Tensor, input_tensor: Tensor, pg: dist.ProcessGroup):
  if (output_tensor.shape, output_tensor.dtype) not in shaped_buffer:
    shaped_buffer[(output_tensor.shape, output_tensor.dtype)] = SymmBufferRegistry.get_instance().allocate_symm_buffer(f'ag_buffer_{output_tensor.shape}_{output_tensor.dtype}', output_tensor.shape, output_tensor.dtype)
  target_tensor = shaped_buffer[(output_tensor.shape, output_tensor.dtype)]

  # peers = SymmBufferRegistry.get_instance().get_peer_tensors(input_tensor)
  grid = (torch.distributed.get_world_size(pg),)
  # print(f"Rank {torch.distributed.get_rank(pg)} grid: {grid}")
  chunk_size = (2**20 // input_tensor.element_size())
  assert input_tensor.numel() % chunk_size == 0
  cupy_stream = PyTorchStreamWrapper(torch.cuda.current_stream())
  # print(f'chunk size {chunk_size}')
  
  nvshmem_device_producer_all_gather_2d_get_block_kernel_chunked[(torch.distributed.get_world_size(pg), )](
    input_tensor,
    target_tensor,
    input_tensor.numel(),
    input_tensor.element_size(),
    torch.distributed.get_rank(pg),
    torch.distributed.get_world_size(pg),
    chunk_size,
    num_warps=16,
  )
  # nvshmem.core.quiet(stream=cupy_stream)
  # for chunk in range(total_chunks):
  #     nvshmem_device_producer_all_gather_2d_get_block_kernel[grid](
  #       input_tensor,
  #       target_tensor,
  #       input_tensor.numel(),
  #       input_tensor.element_size(),
  #       torch.distributed.get_rank(pg),
  #       torch.distributed.get_world_size(pg),
  #       chunk,
  #       total_chunks,
  #       num_warps=16,
  #     )
  #     nvshmem.core.quiet(stream=cupy_stream)
  # nvshmem_device_producer_all_gather_2d_get_block_kernel[(16, )](
  #     input_tensor,
  #     target_tensor,
  #     input_tensor.numel(),
  #     input_tensor.element_size(),
  #     torch.distributed.get_rank(pg),
  #     torch.distributed.get_world_size(pg),
  #     num_warps=32,
  # )

  # nvshmem_device_producer_all_gather_2d_put_block_kernel[grid](
  #   input_tensor,
  #   target_tensor,
  #   input_tensor.numel(),
  #   input_tensor.element_size(),
  #   torch.distributed.get_rank(pg),
  #   torch.distributed.get_world_size(pg),
  #   num_warps=16,
  # )
  
  # torch.distributed.barrier(group=pg)
  # torch.cuda.synchronize()
  # print(f"Rank {torch.distributed.get_rank(pg)} target_tensor: {target_tensor} input_tensor: {input_tensor}")
  output_tensor.copy_(target_tensor)



def all_gather_sync_cache(input_tensor: Tensor, pg: dist.ProcessGroup):
  return
    


def all_gather_into_tensor_nccl(output_tensor: Tensor, input_tensor: Tensor, pg: dist.ProcessGroup):
    return dist.all_gather_into_tensor(output_tensor, input_tensor, group=pg)


if __name__ == "__main__":
    torch.cuda.cudart().cudaProfilerStart()
    try:
      torch.cuda.set_device(f"cuda:{int(os.environ['RANK']) % torch.cuda.device_count()}")
      torch.distributed.init_process_group("nccl")
      init_nvshmem()
      world_size = torch.distributed.get_world_size()
      rank = torch.distributed.get_rank()
      registry = SymmBufferRegistry.get_instance()
      cnt = 20
      size = 16 * (2**20)
      comp_sizes = torch.rand(cnt).tolist()
      dtype = torch.int64

      group_count = 1
      
      for i in range(group_count):
        group_ranks_ = range(i, world_size, group_count)
        group_ = torch.distributed.new_group(ranks=group_ranks_, backend="nccl")
        if rank in group_ranks_:
          group_ranks = group_ranks_
          group = group_
      group_size = len(group_ranks)
      print(f"Rank {rank} group: {group_ranks}")

        
      torch.cuda.synchronize()
      mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
      mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
      print(f"[Rank {rank}] CUDA memory allocated: {mem_allocated:.2f} MB, reserved: {mem_reserved:.2f} MB")
      compute_buffer = [torch.empty(int(x*16384),8192, dtype=torch.bfloat16, device="cuda") for x in comp_sizes]
      compute_param = torch.empty(8192, 8192, dtype=torch.bfloat16, device="cuda")




      src_tensors = [torch.empty(size, dtype=dtype, device="cuda") for _ in range(cnt)]
      for i in range(cnt):
        src_tensors[i].fill_(i + rank*2)
        src_tensors[i] = registry.update_symm_buffer(i, src_tensors[i])
        # all_gather_sync_cache(src_tensors[i], group)

      # for all_gather_func in [all_gather_into_tensor, all_gather_into_tensor_nccl_comm, all_gather_into_tensor_nccl, all_gather_into_tensor_multinode]:
      for all_gather_func in [all_gather_into_tensor_nccl, all_gather_into_tensor_multinode]:
        with torch.cuda.nvtx.range(all_gather_func.__name__):
          start_events = [torch.cuda.Event(enable_timing=True) for _ in range(cnt)]
          comm_events = [torch.cuda.Event(enable_timing=True) for _ in range(cnt)]
          compute_events = [torch.cuda.Event(enable_timing=True) for _ in range(cnt)]
          start = torch.cuda.Event(enable_timing=True)
          
          for i in range(cnt):
            if i == 1:
              start.record()
            dst = torch.empty(size * group_size, dtype=dtype, device="cuda")
            # dst_arr = [
            #   dst[r * size:(r + 1) * size]
            #   for r in range(world_size)
            # ]
            start_events[i].record()
            all_gather_func(dst, src_tensors[i], group)
            comm_events[i].record()
            # compute_buffer[i] @ compute_param
            compute_events[i].record()
            
            # print(dst)
            for r in range(group_size):
              expected = group_ranks[r] * 2 + i
              assert torch.eq(dst[r * size:(r + 1) * size], expected).all(), f"Rank {rank} cnt {i} r {r} dst: {dst[r * size:(r + 1) * size]}, expected: {expected}"
          end = torch.cuda.Event(enable_timing=True)
          end.record()
          dist.barrier()
          torch.cuda.synchronize()
          # print(f"Rank {rank} comm time: {[start_events[i].elapsed_time(comm_events[i]) for i in range(cnt)]}, compute time: {[comm_events[i].elapsed_time(compute_events[i]) for i in range(cnt)]}")
          all_gather_payload = size * (group_size - 1)* dtype.itemsize
          print(f"Rank {rank} all_gather bw: {all_gather_payload / 1024 ** 2 * (cnt - 1) / start.elapsed_time(end)}")
          print(f"Total time: {start.elapsed_time(end)}")
          # print(f"Rank {rank} dst: {dst}")

    except Exception as e:
      print(e)
      import traceback
      traceback.print_exc()
    finally:
      registry.finalize()

# for t in local_tensors:
#   nvshmem.core.free_tensor(t)
