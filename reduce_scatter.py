import nvshmem.core
import torch
from cuda import cuda

import triton
import triton.language as tl
from triton_dist.language.extra import libshmem_device
from triton.language.extra.cuda.language_extra import __syncthreads, tid
from triton_dist.utils import (CUDA_CHECK, dist_print, initialize_distributed, nvshmem_barrier_all_on_stream,
                               NVSHMEM_SIGNAL_DTYPE, nvshmem_create_tensors, nvshmem_create_tensor, nvshmem_free_tensor_sync)
from typing import List
import time
from odc.utils import SymmBufferRegistry, init_nvshmem
import torch.distributed as dist

### NVSHMEM kernels are used by clients to communicate with reduction servers
@triton.jit(do_not_specialize=["target_rank", "lock_id"])
def nvshmem_poll_lock_kernel(
    lock_buffer_ptr,
    target_rank,
    lock_id,
):
    pid = tl.program_id(axis=0)
    tidx = tid(axis=0)
    if pid == 0 and tidx == 0:
        r = 1
        while r != 0:
            r = libshmem_device.atomic_compare_swap(lock_buffer_ptr + lock_id, 0, 1, target_rank)
    __syncthreads()


@triton.jit(do_not_specialize=["target_rank", "lock_id", "value"])
def nvshmem_set_kernel(
    lock_buffer_ptr,
    target_rank,
    lock_id,
    value,
):
    pid = tl.program_id(axis=0)
    tidx = tid(axis=0)
    if pid == 0 and tidx == 0:
        libshmem_device.atomic_swap(lock_buffer_ptr + lock_id, value, target_rank)
    __syncthreads()


### Plain triton kernels are used within reduction servers themselves, locally.

@triton.jit(do_not_specialize=["num_locks"])
def reduction_watcher_kernel(
    lock_buffer_ptr,
    num_locks,
    BLOCK_SIZE: tl.constexpr,
):
   pid = tl.program_id(axis=0)
   tidx = tid(axis=0)
   if pid == 0:
      r = 0
      offsets = tl.arange(0, BLOCK_SIZE)
      mask = offsets < num_locks
      while r < 2:
          data = tl.load(lock_buffer_ptr + offsets, mask=mask, volatile=True)
          r = tl.max(data)
          __syncthreads()

@triton.jit(do_not_specialize=["lock_id"])
def reset_lock_kernel(
    lock_buffer_ptr,
    lock_id):
    pid = tl.program_id(axis=0)
    tidx = tid(axis=0)
    if pid == 0 and tidx == 0:
        tl.atomic_xchg(lock_buffer_ptr + lock_id, 0)
    __syncthreads()

class DistLock:
    def __init__(self, num_locks):
        self.num_locks = num_locks
        self.lock_buffers = nvshmem_create_tensor(self.num_locks, torch.int32)
        self.lock_buffers.fill_(0)
        self.cpu_lock_buffers = torch.empty_like(self.lock_buffers, device="cpu").pin_memory()
        

    def lock(self, target_rank, lock_id):
        nvshmem_poll_lock_kernel[(1, )](self.lock_buffers, target_rank, lock_id)
    
    def notify_data(self, target_rank, lock_id):
        nvshmem_set_kernel[(1, )](self.lock_buffers, target_rank, lock_id, 2)

class ReductionWatcher:
    def __init__(self, accumulations: List[torch.Tensor], buffers: List[torch.Tensor], lock_buffers: torch.Tensor):
        self.accumulations = accumulations
        self.buffers = buffers
        self.lock_buffers = lock_buffers
        self.cpu_lock_buffers = torch.empty_like(self.lock_buffers, device="cpu").pin_memory()
        self.num_locks = len(lock_buffers)
        self.running = True
        self.task_count = 0

    def stop(self):
        self.running = False

    def wait_and_reset_task_count(self, expected):
        while self.task_count < expected:
            time.sleep(0)
        self.task_count = 0

    def add_buffer(self, accumulation, buffer):
        # print(f"Rank {dist.get_rank()} adding buffer {accumulation} {buffer}")
        self.accumulations.append(tensor_from_handle(*accumulation))
        self.buffers.append(tensor_from_handle(*buffer))

    def run(self):
        while self.running:
            block_size = triton.next_power_of_2(self.num_locks)
            self.cpu_lock_buffers.fill_(0)
            reduction_watcher_kernel[(1, )](self.lock_buffers, self.num_locks, BLOCK_SIZE=block_size)

            self.cpu_lock_buffers.copy_(self.lock_buffers, non_blocking=True)
            torch.cuda.current_stream().synchronize()
            if not self.running:
                break

            for idx, flag in enumerate(self.cpu_lock_buffers):
                if flag == 2:
                    self.accumulations[idx].add_(self.buffers[idx])
                    self.task_count += 1
                    # print(f"adding buffer {idx} {self.accumulations[idx]} {self.buffers[idx]}")
                    reset_lock_kernel[(1, )](self.lock_buffers, idx)

def tensor_from_handle(handle, size, dtype):
    from tensor_ipc import reconstruct_tensor
    return reconstruct_tensor(handle, (size,), dtype)

def reduction_watcher_function(device_id, accumulations, buffers, lock_buffers, cmd_queue, response_queue):
    torch.cuda.set_device(device_id)
    import sys
    
    buffers = [tensor_from_handle(*buffer) for buffer in buffers]
    accumulations = [tensor_from_handle(*acc) for acc in accumulations]
    lock_buffers = tensor_from_handle(*lock_buffers)

    watcher = ReductionWatcher(accumulations, buffers, lock_buffers)

    from threading import Thread
    def cmd_thread():
        torch.cuda.set_device(device_id)
        while True:
            data = cmd_queue.get()
            cmd = data[0]
            args = data[1:]
            response_queue.put(getattr(watcher, cmd)(*args))
            if cmd == 'stop':
                break

    cmd_thread = Thread(target=cmd_thread)
    cmd_thread.start()
    watcher.run()
    cmd_thread.join()

def start_reduction_watcher(accumulations, buffers, lock_buffers):
    from torch.multiprocessing import Process
    
    ctx = torch.multiprocessing.get_context("spawn")
    cmd_queue = ctx.Queue()
    response_queue = ctx.Queue()
    device_id = torch.cuda.current_device()
    process = ctx.Process(target=reduction_watcher_function,
                       args=(device_id, accumulations, buffers, lock_buffers, cmd_queue, response_queue))
    process.start()
    return cmd_queue, response_queue

def call_watcher(watcher_handle, cmd, *args):
    cmd_queue, response_queue = watcher_handle
    cmd_queue.put((cmd, *args))
    return response_queue.get()

def get_nvshmem_handle(tensor):
    from tensor_ipc import get_ipc_handle
    handle = get_ipc_handle(tensor)
    return handle, tensor.numel(), tensor.dtype

class ReductionService:
    def __init__(self, accumulation_dtype=None):
        self.buffers = []
        self.lock = None
        self.reduction_watcher = None
        self.indices = {}
        self.dispatched_tasks = 0
        self.accumulation_dtype = accumulation_dtype
        pass
    
    def register(self, key, output_tensor_shape, grad_dtype,reduction_dtype):
        if self.reduction_watcher is None:
            self.lock = DistLock(1024)
            lock_buffers_handle = get_nvshmem_handle(self.lock.lock_buffers)

            # Make sure changes are visible to all reduction watchers
            torch.distributed.barrier()
            torch.cuda.synchronize()

            self.reduction_watcher = start_reduction_watcher([], [], lock_buffers_handle)

        buffer_key = f'rs_buffer_{key}'
        accumulation_key = f'rs_accumulation_{key}'
        assert len(output_tensor_shape) == 1
        registry = SymmBufferRegistry.get_instance()
        assert not registry.has_key(accumulation_key)
        # assert self.reduction_watcher is None, "Reduction watcher is already running"
        
        acc = registry.allocate_symm_buffer(accumulation_key, output_tensor_shape, grad_dtype)
        acc.fill_(0)
        buffer = registry.allocate_symm_buffer(buffer_key, output_tensor_shape, reduction_dtype)
            
        
        current_size = len(self.buffers)
        self.buffers.append((acc, buffer))
        self.indices[key] = current_size

        acc_handle = get_nvshmem_handle(acc)
        buffer_handle = get_nvshmem_handle(buffer)
        call_watcher(self.reduction_watcher, 'add_buffer', acc_handle, buffer_handle)

        # Make sure changes are visible to all reduction watchers
        torch.distributed.barrier()
        torch.cuda.synchronize()

    def clear_accumulations(self):
        for acc, _ in self.buffers:
            acc.fill_(0)
    
    def infer_output_shape(self, input_tensor, pg: dist.ProcessGroup):
        assert len(input_tensor.shape) == 1
        assert input_tensor.shape[0] % dist.get_world_size(pg) == 0
        return (input_tensor.shape[0] // dist.get_world_size(pg),)

    def reduce_scatter_accumulation(self, key, input_tensor, pg: dist.ProcessGroup):
        if key not in self.indices:
            accum_dtype = self.accumulation_dtype if self.accumulation_dtype is not None else input_tensor.dtype
            self.register(key, self.infer_output_shape(input_tensor, pg), input_tensor.dtype, accum_dtype)

        acc, buffer = self.buffers[self.indices[key]]

        peer_buffers = SymmBufferRegistry.get_instance().get_peer_tensors(buffer)

        group_size = dist.get_world_size(pg)
        group_ranks = dist.get_process_group_ranks(pg)
        group_idx = dist.get_rank(pg)

        key_id = self.indices[key]

        size = buffer.numel()
        assert input_tensor.numel() == size * group_size
        
        for r_offset in range(0, group_size):
            dst_group_idx = (group_idx + r_offset) % group_size
            dst_buffer = peer_buffers[group_ranks[dst_group_idx]]
            self.lock.lock(target_rank=group_ranks[dst_group_idx], lock_id=key_id)

            dst_buffer.copy_(input_tensor[dst_group_idx * size:(dst_group_idx + 1) * size])
            self.lock.notify_data(target_rank=group_ranks[dst_group_idx], lock_id=key_id)
        
        self.dispatched_tasks += 1
    
    def get_accumulation(self, key):
        acc, _ = self.buffers[self.indices[key]]
        return acc
    
    def sync(self, pg: dist.ProcessGroup):
        group_size = dist.get_world_size(pg)
        call_watcher(self.reduction_watcher, 'wait_and_reset_task_count', self.dispatched_tasks * group_size)
        self.dispatched_tasks = 0

    def stop(self):
        if self.reduction_watcher is not None:
            call_watcher(self.reduction_watcher, 'stop')
            torch.distributed.barrier()
            torch.cuda.synchronize()
            self.lock.lock_buffers.fill_(2)
            nvshmem_free_tensor_sync(self.lock.lock_buffers)



if __name__ == "__main__":
    import os
    try:
      torch.cuda.set_device(f"cuda:{os.environ['RANK']}")
      torch.distributed.init_process_group("nccl")
      init_nvshmem()
      world_size = torch.distributed.get_world_size()
      rank = torch.distributed.get_rank()

      accum_dtype = torch.bfloat16
      grad_dtype = torch.bfloat16

      reduction_service = ReductionService(accumulation_dtype=accum_dtype)
      cnt = 20
      times = 10
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

      data = torch.rand(cnt * times, size, dtype=grad_dtype, device="cuda") / group_size / times
      # data = torch.arange(cnt * times * size, dtype=grad_dtype, device="cuda").reshape(cnt * times, size) / group_size / times
      # data = torch.ones(cnt * times, size, dtype=grad_dtype, device="cuda") * rank

      # for i in range(cnt):
      #   reduction_service.register(i, (size // group_size,), grad_dtype, accum_dtype)

        
      torch.cuda.synchronize()
      mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
      mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
      print(f"[Rank {rank}] CUDA memory allocated: {mem_allocated:.2f} MB, reserved: {mem_reserved:.2f} MB")
      compute_buffer = [torch.empty(int(x*16384),8192, dtype=torch.bfloat16, device="cuda") for x in comp_sizes]
      compute_param = torch.empty(8192, 8192, dtype=torch.bfloat16, device="cuda")

      nccl_accumulations = [torch.zeros(size // group_size, dtype=accum_dtype, device="cuda") for _ in range(cnt)]
      def reduce_scatter_accumulation_nccl(src_tensor, dest_idx, pg: dist.ProcessGroup):
        output = torch.empty((src_tensor.numel() // dist.get_world_size(pg),), dtype=src_tensor.dtype, device="cuda")
        torch.distributed.reduce_scatter_tensor(output, src_tensor, op=torch.distributed.ReduceOp.SUM, group=pg)
        nccl_accumulations[dest_idx].add_(output)

      def reduce_scatter_accumulation(src_tensor, dest_idx, pg: dist.ProcessGroup):
        reduction_service.reduce_scatter_accumulation(dest_idx, src_tensor, pg)


      for reduce_scatter_func in [reduce_scatter_accumulation, reduce_scatter_accumulation_nccl]:
        with torch.cuda.nvtx.range(reduce_scatter_func.__name__):
          start_events = [torch.cuda.Event(enable_timing=True) for _ in range(cnt * times)]
          comm_events = [torch.cuda.Event(enable_timing=True) for _ in range(cnt * times)]
          compute_events = [torch.cuda.Event(enable_timing=True) for _ in range(cnt * times)]
          start = torch.cuda.Event(enable_timing=True)
          
          for i in range(cnt * times):
            dst_idx = i % cnt
            if i == 1:
              start.record()
            
            # dst_arr = [
            #   dst[r * size:(r + 1) * size]
            #   for r in range(world_size)
            # ]
            start_events[i].record()
            reduce_scatter_func(data[i], dst_idx, group)
            comm_events[i].record()
            compute_buffer[dst_idx] @ compute_param
            compute_events[i].record()
            
            # print(dst)
          
          
          
          if reduce_scatter_func == reduce_scatter_accumulation:
            reduction_service.sync(group)
            # print(f"Rank {rank} reduction_service: {reduction_service.buffers[0][0]}")
          else:
            # print(f"Rank {rank} nccl_accumulations: {nccl_accumulations[0]}")
            
            for i in range(cnt):
              print(f"Rank {rank} nccl_accumulations: {nccl_accumulations[i]} reduction_service: {reduction_service.buffers[i][0]}")
              torch.testing.assert_close(nccl_accumulations[i], reduction_service.buffers[i][0], rtol=5e-2, atol=5e-2)
          end = torch.cuda.Event(enable_timing=True)
          end.record()
          dist.barrier()
          torch.cuda.synchronize()
          print(f"Rank {rank} comm time: {[start_events[i].elapsed_time(comm_events[i]) for i in range(cnt * times)]}, compute time: {[comm_events[i].elapsed_time(compute_events[i]) for i in range(cnt * times)]}")
          reduce_scatter_payload = size // group_size* (group_size - 1)* data.dtype.itemsize
          print(f"Rank {rank} reduce_scatter bw: {[reduce_scatter_payload / 1024 ** 2 / start_events[i].elapsed_time(comm_events[i]) for i in range(cnt * times)]}")
          print(f"Rank {rank} Total time: {start.elapsed_time(end)}")
          # print(f"Rank {rank} dst: {dst}")

      reduction_service.stop()

    except Exception as e:
      print(e)
      import traceback
      traceback.print_exc()
    finally:
      SymmBufferRegistry.get_instance().finalize()
