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
from odc.utils import SymmBufferRegistry, init_nvshmem
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


def all_gather_into_tensor_nccl(output_tensor: Tensor, input_tensor: Tensor, pg: dist.ProcessGroup):
    return dist.all_gather_into_tensor(output_tensor, input_tensor, group=pg)


if __name__ == "__main__":
    try:
      torch.cuda.set_device(f"cuda:{os.environ['RANK']}")
      torch.distributed.init_process_group("nccl")
      init_nvshmem()
      world_size = torch.distributed.get_world_size()
      rank = torch.distributed.get_rank()
      registry = SymmBufferRegistry.get_instance()
      cnt = 20
      size = 16 * (1000 ** 2)
      comp_sizes = torch.rand(cnt).tolist()

      group_count = 2
      
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




      src_tensors = [torch.empty(size, dtype=torch.long, device="cuda") for _ in range(cnt)]
      for i in range(cnt):
        src_tensors[i].fill_(i + rank*100)
        src_tensors[i] = registry.update_symm_buffer(i, src_tensors[i])

      for all_gather_func in [all_gather_into_tensor, all_gather_into_tensor_nccl]:
        with torch.cuda.nvtx.range(all_gather_func.__name__):
          start_events = [torch.cuda.Event(enable_timing=True) for _ in range(cnt)]
          comm_events = [torch.cuda.Event(enable_timing=True) for _ in range(cnt)]
          compute_events = [torch.cuda.Event(enable_timing=True) for _ in range(cnt)]
          start = torch.cuda.Event(enable_timing=True)
          
          for i in range(cnt):
            if i == 1:
              start.record()
            dst = torch.empty(size * group_size, dtype=torch.long, device="cuda")
            # dst_arr = [
            #   dst[r * size:(r + 1) * size]
            #   for r in range(world_size)
            # ]
            start_events[i].record()
            all_gather_func(dst, src_tensors[i], group)
            comm_events[i].record()
            compute_buffer[i] @ compute_param
            compute_events[i].record()
            
            # print(dst)
            for r in range(group_size):
              expected = group_ranks[r] * 100 + i
              assert torch.eq(dst[r * size:(r + 1) * size], expected).all(), f"Rank {rank} cnt {i} r {r} dst: {dst[r * size:(r + 1) * size]}, expected: {expected}"
          end = torch.cuda.Event(enable_timing=True)
          end.record()
          dist.barrier()
          torch.cuda.synchronize()
          print(f"Rank {rank} comm time: {[start_events[i].elapsed_time(comm_events[i]) for i in range(cnt)]}, compute time: {[comm_events[i].elapsed_time(compute_events[i]) for i in range(cnt)]}")
          all_gather_payload = size * (group_size - 1)* torch.long.itemsize
          print(f"Rank {rank} all_gather bw: {[all_gather_payload / 1024 ** 2 / start_events[i].elapsed_time(comm_events[i]) for i in range(cnt)]}")
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