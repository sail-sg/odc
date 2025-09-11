import json
from collections import defaultdict
from functools import reduce
import nvshmem.core
import torch
import os
from cuda import cuda

import triton
import triton.language as tl
from triton_dist.language.extra import libshmem_device
from triton.language.extra.cuda.language_extra import __syncthreads, tid
from triton_dist.utils import (CUDA_CHECK, dist_print, initialize_distributed, nvshmem_barrier_all_on_stream,
                               NVSHMEM_SIGNAL_DTYPE, nvshmem_create_tensors, nvshmem_create_tensor, nvshmem_free_tensor_sync)
from typing import List
import time
from utils import SymmBufferRegistry, init_nvshmem, get_same_local_rank_pg, get_local_world_size, get_local_world_pg
import torch.distributed as dist

### NVSHMEM kernels are used by clients to communicate with reduction servers
@triton.jit(do_not_specialize=["target_rank", "lock_id", "value"])
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
            r = libshmem_device.atomic_compare_swap(lock_buffer_ptr + lock_id, 0, -1, target_rank)
    __syncthreads()


@triton.jit(do_not_specialize=["target_rank", "lock_id", "value"])
def nvshmem_set_kernel(
    lock_buffer_ptr,
    target_rank,
    lock_id,
    value,
    accumulation_start_indices_ptr,
    accumulation_start_index,
):
    pid = tl.program_id(axis=0)
    tidx = tid(axis=0)
    if pid == 0 and tidx == 0:
        libshmem_device.atomic_swap(accumulation_start_indices_ptr + lock_id, accumulation_start_index, target_rank)
        libshmem_device.quiet()
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
      while r < 1:
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
        self.accumulation_start_indices = nvshmem_create_tensor(self.num_locks, torch.int32)
        self.accumulation_start_indices.fill_(0)
        self.cpu_lock_buffers = torch.empty_like(self.lock_buffers, device="cpu").pin_memory()
        

    def lock(self, target_rank, buffer_id):
        assert buffer_id < self.num_locks
        # TODO: This is a hack as currently nvshmem doesn't work cross node. So we init nvshmem only within node.
        # nvshmem_poll_lock_kernel[(1, )](self.lock_buffers, target_rank, buffer_id)
        nvshmem_poll_lock_kernel[(1, )](self.lock_buffers, target_rank % get_local_world_size(), buffer_id)
    
    def notify_data(self, target_rank, buffer_id, accumulation_id, accumulation_start_index):
        assert buffer_id < self.num_locks
        assert accumulation_id > 0
        # TODO: This is a hack as currently nvshmem doesn't work cross node. So we init nvshmem only within node.
        # nvshmem_set_kernel[(1, )](self.lock_buffers, target_rank, buffer_id, accumulation_id)
        assert accumulation_start_index >= 0
        nvshmem_set_kernel[(1, )](self.lock_buffers, target_rank % get_local_world_size(), buffer_id, accumulation_id,
                                  self.accumulation_start_indices, accumulation_start_index)

class ReductionWatcher:
    def __init__(self, accumulations: List[torch.Tensor], buffers: List[torch.Tensor], lock_buffers: torch.Tensor, accumulation_start_indices: torch.Tensor):
        self.accumulations = accumulations
        self.buffers = buffers
        self.lock_buffers = lock_buffers
        self.accumulation_start_indices = accumulation_start_indices
        self.cpu_lock_buffers = torch.empty_like(self.lock_buffers, device="cpu").pin_memory()
        self.cpu_accumulation_start_indices = torch.empty_like(self.accumulation_start_indices, device="cpu").pin_memory()
        self.num_locks = len(lock_buffers)
        self.running = True
        self.task_count = 0

    def stop(self):
        self.running = False

    def wait_and_reset_task_count(self, expected):
        while self.task_count < expected:
            time.sleep(0)
            # print(f"Rank {torch.cuda.current_device()} waiting for task count {self.task_count} < {expected}")
        self.task_count = 0

    def add_buffer(self, buffers):
        # print(f"Rank {dist.get_rank()} adding buffer {accumulation} {buffer}")
        self.buffers.append([tensor_from_handle(*buffer) for buffer in buffers])
    
    def add_accumulation(self, accumulations):
        self.accumulations.append([tensor_from_handle(*acc) for acc in accumulations])

    def run(self):
        while self.running:
            block_size = triton.next_power_of_2(self.num_locks)

            self.cpu_lock_buffers.fill_(0)
            self.cpu_accumulation_start_indices.fill_(0)
            time.sleep(1/10000)
            # reduction_watcher_kernel[(1, )](self.lock_buffers, self.num_locks, BLOCK_SIZE=block_size)

            self.cpu_lock_buffers.copy_(self.lock_buffers, non_blocking=True)
            self.cpu_accumulation_start_indices.copy_(self.accumulation_start_indices, non_blocking=True)
            torch.cuda.current_stream().synchronize()
            if not self.running:
                break
            
            nonzeros = torch.nonzero(self.cpu_lock_buffers, as_tuple=False).squeeze(1).tolist()
            
            for buf_id in nonzeros:
                acc_id = self.cpu_lock_buffers[buf_id]
                start = self.cpu_accumulation_start_indices[buf_id]
                if acc_id > 0:
                    # print(f"Rank {torch.cuda.current_device()} adding buffer {buf_id} -> {acc_id - 1}")
                    for acc, buf in zip(self.accumulations[acc_id - 1], self.buffers[buf_id]):
                        size = min(buf.numel(), acc.numel() - start)
                        acc[start:start + size].add_(buf[:size])
                        # print(f"Rank {torch.cuda.current_device()} adding buffer {buf_id} -> {acc_id - 1} start {start} size {size} acc.numel() {acc.numel()}")
                    if start + size >= acc.numel():
                        assert start + size == acc.numel()
                        self.task_count += 1
                    
                    # print(f"adding buffer {idx} {self.accumulations[idx]} {self.buffers[idx]}")
                    reset_lock_kernel[(1, )](self.lock_buffers, buf_id)

def tensor_from_handle(handle, size, dtype):
    from tensor_ipc import reconstruct_tensor
    return reconstruct_tensor(handle, (size,), dtype)

def reduction_watcher_function(device_id, accumulations, buffers, lock_buffers, accumulation_start_indices, cmd_queue, response_queue):
    torch.cuda.set_device(device_id)
    import sys
    # torch.cuda.cudart().cudaProfilerStart()
    buffers = [tensor_from_handle(*buffer) for buffer in buffers]
    accumulations = [tensor_from_handle(*acc) for acc in accumulations]
    lock_buffers = tensor_from_handle(*lock_buffers)
    accumulation_start_indices = tensor_from_handle(*accumulation_start_indices)

    watcher = ReductionWatcher(accumulations, buffers, lock_buffers, accumulation_start_indices)

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

def start_reduction_watcher(accumulations, buffers, lock_buffers, accumulation_start_indices):
    from torch.multiprocessing import Process

    # original_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    # if original_visible_devices is not None:
    #     del os.environ['CUDA_VISIBLE_DEVICES']
    
    ctx = torch.multiprocessing.get_context("spawn")
    cmd_queue = ctx.Queue()
    response_queue = ctx.Queue()
    device_id = torch.distributed.get_rank() % get_local_world_size()
    process = ctx.Process(target=reduction_watcher_function,
                       args=(device_id, accumulations, buffers, lock_buffers, accumulation_start_indices, cmd_queue, response_queue))
    process.start()
    # if original_visible_devices is not None:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = original_visible_devices
    # elif 'CUDA_VISIBLE_DEVICES' in os.environ:
    #     del os.environ['CUDA_VISIBLE_DEVICES']
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
        self.accumulations = []
        self.buffers = []
        self.lock = None
        self.reduction_watcher = None
        self.accumulation_indices = {}
        self.buffer_indices = {}
        self.shared_buffer = {}
        self.fixed_buffers = {}
        self.dispatched_tasks = 0
        self.accumulation_dtype = accumulation_dtype
        self.rank_streams = defaultdict(lambda: torch.cuda.Stream())
    
    def pre_register(self, key, input_tensor, pg: dist.ProcessGroup):
        assert key not in self.accumulation_indices
        accum_dtype = self.accumulation_dtype if self.accumulation_dtype is not None else input_tensor.dtype
        self.register(key, self.infer_output_shape(input_tensor, pg), input_tensor.dtype, accum_dtype)
    
    def register(self, key, output_tensor_shape, grad_dtype,reduction_dtype):
        if self.reduction_watcher is None:
            self.lock = DistLock(128)
            lock_buffers_handle = get_nvshmem_handle(self.lock.lock_buffers)
            accumulation_start_indices_handle = get_nvshmem_handle(self.lock.accumulation_start_indices)
            # Make sure changes are visible to all reduction watchers
            torch.distributed.barrier()
            torch.cuda.synchronize()

            self.reduction_watcher = start_reduction_watcher([], [], lock_buffers_handle, accumulation_start_indices_handle)

        buffer_key = f'rs_buffer_{key}'
        accumulation_key = f'rs_accumulation_{key}'
        assert len(output_tensor_shape) == 1
        registry = SymmBufferRegistry.get_instance()
        assert not registry.has_key(accumulation_key)
        # assert self.reduction_watcher is None, "Reduction watcher is already running"
        
        def create_and_register(key, shape, dtype, add_func):
            buffer = registry.allocate_symm_buffer(key, shape, dtype)
            local_handles = []
            for local_tensor in registry.get_local_peer_tensors(buffer):
                local_tensor.fill_(0)
                local_handles.append(get_nvshmem_handle(local_tensor))
            call_watcher(self.reduction_watcher, add_func, local_handles)
            return buffer
            
           
        # acc = registry.allocate_symm_buffer(accumulation_key, output_tensor_shape, reduction_dtype)
        acc = create_and_register(accumulation_key, output_tensor_shape, reduction_dtype, 'add_accumulation')
        self.accumulation_indices[key] = len(self.accumulations)
        self.accumulations.append(acc)

        # local_acc_handles = []  
        # for local_acc in registry.get_local_peer_tensors(acc):
        #     local_acc.fill_(0)
        #     local_acc_handles.append(get_nvshmem_handle(local_acc))
        # call_watcher(self.reduction_watcher, 'add_accumulation', local_acc_handles)
        buffer_shape = self.get_buffer_shape(output_tensor_shape)
        
        if os.environ.get('ODC_SINGLE_BUFFER', '0') != '1' and os.environ.get('ODC_NUM_BUFFERS', '0') == '0':
            buffer = create_and_register(buffer_key, buffer_shape, grad_dtype, 'add_buffer')
            self.buffer_indices[key] = len(self.buffers)
            self.buffers.append(buffer)
        elif os.environ.get('ODC_NUM_BUFFERS', '0') != '0':
            num_buffers = int(os.environ.get('ODC_NUM_BUFFERS', '0'))
            assert num_buffers > 0, f"Invalid ODC_NUM_BUFFERS {num_buffers}"
            fixed_buffer_key = (grad_dtype, buffer_shape)
            if fixed_buffer_key not in self.fixed_buffers:
                bufs = []
                for i in range(num_buffers):
                    buffer_key = self.get_fixed_buffer_key(buffer_shape, grad_dtype, i)
                    buffer = create_and_register(buffer_key, buffer_shape, grad_dtype, 'add_buffer')
                    bufs.append(buffer)
                    self.buffer_indices[buffer_key] = len(self.buffers)
                    self.buffers.append(buffer)
                self.fixed_buffers[fixed_buffer_key] = bufs
        else:
            shared_buffer_key = (grad_dtype, buffer_shape)
            if shared_buffer_key not in self.shared_buffer:
                cnt = len(self.shared_buffer)
                buffer = create_and_register(f'shared_buffer_{cnt}', buffer_shape, grad_dtype, 'add_buffer')
                self.shared_buffer[shared_buffer_key] = (cnt, buffer)
                
                self.buffers.append(buffer)
            buffer = self.shared_buffer[shared_buffer_key][1]
            self.buffer_indices[key] = self.shared_buffer[shared_buffer_key][0]

        # Make sure changes are visible to all reduction watchers
        torch.distributed.barrier()
        torch.cuda.synchronize()

    def get_buffer_shape(self, output_tensor_shape):
        # MAX_BUFFER_SIZE = 64 * 1000 * 1000
        MAX_BUFFER_SIZE = 8 * 1000 * 1000
        return (min(MAX_BUFFER_SIZE, reduce(lambda x, y: x * y, output_tensor_shape)),)

    def get_fixed_buffer_key(self, output_tensor_shape, grad_dtype, index):
        numel = reduce(lambda x, y: x * y, output_tensor_shape)
        return f'rs_fixed_buffer_{numel}_{grad_dtype}_{index}'

    def rank_to_fixed_buffer_index(self, rank):
        assert "ODC_NUM_BUFFERS" in os.environ, "ODC_NUM_BUFFERS must be set"
        num_buffers = int(os.environ.get('ODC_NUM_BUFFERS', '0'))
        return rank % num_buffers

    def clear_accumulations(self):
        for acc in self.accumulations:
            acc.fill_(0)
    
    def infer_output_shape(self, input_tensor, pg: dist.ProcessGroup):
        assert len(input_tensor.shape) == 1
        assert input_tensor.shape[0] % dist.get_world_size(pg) == 0
        return (input_tensor.shape[0] // dist.get_world_size(pg),)

    def reduce_scatter_accumulation_nccl_comm(self, key, input_tensor, pg: dist.ProcessGroup):
        output_tensor_shape = self.infer_output_shape(input_tensor, pg)
        if key not in self.accumulation_indices:
            accum_dtype = self.accumulation_dtype if self.accumulation_dtype is not None else input_tensor.dtype
            self.register(key, output_tensor_shape, input_tensor.dtype, accum_dtype)

        acc = self.accumulations[self.accumulation_indices[key]]

        buffer_shape = self.get_buffer_shape(output_tensor_shape)
        # buffer = self.buffers[self.buffer_indices[key]]
        if os.environ.get('ODC_NUM_BUFFERS', '0') != '0':
            index = self.rank_to_fixed_buffer_index(torch.distributed.get_rank())
            buffer_key = self.get_fixed_buffer_key(buffer_shape, input_tensor.dtype, index)
            assert buffer_key in self.buffer_indices, f"Buffer key {buffer_key} not found in buffer_indices {self.buffer_indices} fixed_buffers {self.fixed_buffers}"
            buffer_id = self.buffer_indices[buffer_key]
        else:
            buffer_id = self.buffer_indices[key]
        buffer = self.buffers[buffer_id]
        assert buffer.numel() == reduce(lambda x, y: x * y, buffer_shape)
        # buffer = buffer.view(*output_tensor_shape)

        local_peer_accs = SymmBufferRegistry.get_instance().get_local_peer_tensors(acc)

        size = self.infer_output_shape(input_tensor, pg)[0]

        local_world_pg = get_local_world_pg(pg)
        local_world_size = torch.distributed.get_world_size(group=local_world_pg)
        assert len(local_peer_accs) * local_world_size == torch.distributed.get_world_size(group=pg)
        for i in range(0, torch.distributed.get_world_size(group=pg), local_world_size):
          src_tensor = input_tensor[i * size:(i + local_world_size) * size]
          torch.distributed.reduce_scatter_tensor(buffer, src_tensor, group=local_world_pg)
          local_peer_accs[i // local_world_size].add_(buffer)

    def reduce_scatter_accumulation(self, key, input_tensor, pg: dist.ProcessGroup):
        output_tensor_shape = self.infer_output_shape(input_tensor, pg)
        if key not in self.accumulation_indices:
            accum_dtype = self.accumulation_dtype if self.accumulation_dtype is not None else input_tensor.dtype
            self.register(key, output_tensor_shape, input_tensor.dtype, accum_dtype)

        acc = self.accumulations[self.accumulation_indices[key]]

        buffer_shape = self.get_buffer_shape(output_tensor_shape)
        if os.environ.get('ODC_NUM_BUFFERS', '0') != '0':
            index = self.rank_to_fixed_buffer_index(torch.distributed.get_rank())
            buffer_key = self.get_fixed_buffer_key(buffer_shape, input_tensor.dtype, index)
            assert buffer_key in self.buffer_indices, f"Buffer key {buffer_key} not found in buffer_indices {self.buffer_indices} fixed_buffers {self.fixed_buffers}"
            buffer_id = self.buffer_indices[buffer_key]
        else:
            buffer_id = self.buffer_indices[key]
        buffer = self.buffers[buffer_id]
        assert buffer.numel() == reduce(lambda x, y: x * y, buffer_shape)

        peer_buffers = SymmBufferRegistry.get_instance().get_peer_tensors(buffer)

        group_size = dist.get_world_size(pg)
        group_ranks = dist.get_process_group_ranks(pg)
        group_idx = dist.get_rank(pg)
        accumulation_id = self.accumulation_indices[key] + 1

        # size = buffer.numel()
        size = acc.numel()
        assert input_tensor.numel() == size * group_size
        
        local_world_size = get_local_world_size()

        rank_info = [(r, group_ranks[r], group_ranks[r] % local_world_size) for r in range(group_size)]
                
        for local_r_offset in range(0, local_world_size):
            dst_local_rank = (torch.distributed.get_rank() + local_r_offset) % local_world_size
            dst_rank_infos = [r for r in rank_info if r[2] == dst_local_rank]
            matching_rank_in_local_world = torch.distributed.get_rank() // local_world_size * local_world_size + dst_local_rank

            if len(dst_rank_infos) == 0:
                continue
            dst_group_idx, _, dst_local_rank = dst_rank_infos[0]
            dst_buffer = peer_buffers[dst_group_idx]
            buf_size = dst_buffer.numel()
            data_size = input_tensor[dst_group_idx * size:(dst_group_idx + 1) * size].numel()

            for start in range(0, data_size, buf_size):
                self.lock.lock(target_rank=matching_rank_in_local_world, buffer_id=buffer_id)
                copy_size = min(buf_size, data_size - start)
                # print(f"Rank {torch.distributed.get_rank()} copying start {start} size {copy_size} full size {data_size}")
                for (dst_group_idx, dst_rank, dst_local_rank) in dst_rank_infos:
                    dst_buffer = peer_buffers[dst_group_idx]
                    input_data = input_tensor[dst_group_idx * size:(dst_group_idx + 1) * size]
                    dst_buffer[:copy_size].copy_(input_data[start:start+copy_size])
                self.lock.notify_data(target_rank=matching_rank_in_local_world, buffer_id=buffer_id, accumulation_id=accumulation_id, accumulation_start_index=start)
        self.dispatched_tasks += 1
    
    def get_accumulation(self, key):
        acc = self.accumulations[self.accumulation_indices[key]]
        return acc
    
    def sync(self, pg: dist.ProcessGroup):
        # TODO: This actually only syncs CPU of reduction workers, it's possible that the last reduction is on-the-fly.
        dispatched_task_list = [None for _ in range(dist.get_world_size(pg))]
        torch.distributed.all_gather_object(dispatched_task_list, self.dispatched_tasks, group=pg)
        torch.cuda.synchronize()

        local_world_size = get_local_world_size()
        if local_world_size == torch.distributed.get_world_size():
           target = sum(dispatched_task_list)
        else:
           assert torch.distributed.get_world_size(pg) == torch.distributed.get_world_size(), "Cached AG only supports pure data parallelism"
           local_world_start = torch.distributed.get_rank() // local_world_size * local_world_size
           local_world_end = local_world_start + local_world_size
           target = sum(dispatched_task_list[local_world_start:local_world_end])

        call_watcher(self.reduction_watcher, 'wait_and_reset_task_count', target)
        self.dispatched_tasks = 0

        for acc in self.accumulations:
          self.reduce_scatter_sync_cache(acc, pg)

    def reduce_scatter_sync_cache(self, accumulation, pg: dist.ProcessGroup):
        local_world_size = get_local_world_size()
        if local_world_size == torch.distributed.get_world_size():
          return
        
        same_local_rank_pg = get_same_local_rank_pg(pg)
        same_local_rank_pg_ranks = dist.get_process_group_ranks(group=same_local_rank_pg)

        registry = SymmBufferRegistry.get_instance()
        local_peer_tensors = registry.get_local_peer_tensors(accumulation)
        local_tensor = accumulation
        assert local_tensor is local_peer_tensors[torch.distributed.get_rank(group=same_local_rank_pg)]
        torch.distributed.reduce_scatter(local_tensor, local_peer_tensors, group=same_local_rank_pg)
        for tensor in local_peer_tensors:
          if tensor is not accumulation:
            tensor.fill_(0)

    def stop(self):
        if self.reduction_watcher is not None:
            call_watcher(self.reduction_watcher, 'stop')
            torch.distributed.barrier()
            torch.cuda.synchronize()
            self.lock.lock_buffers.fill_(2)
            nvshmem_free_tensor_sync(self.lock.lock_buffers)
            nvshmem_free_tensor_sync(self.lock.accumulation_start_indices)



def size_str_to_int(size_str):
    size_str = size_str.lower()
    if size_str.endswith('kb'):
        return int(size_str[:-2]) * (1000)
    elif size_str.endswith('mb'):
        return int(size_str[:-2]) * (1000 ** 2)
    elif size_str.endswith('gb'):
        return int(size_str[:-2]) * (1000 ** 3)
    else:
        return int(size_str)


if __name__ == "__main__":
    import os
    data_size_str = os.environ.get('DATA_SIZE', '256mb')
    data_size = size_str_to_int(data_size_str)
    data_dir = os.environ.get('DATA_DIR', 'profile')
    data_dir = os.path.join(data_dir, data_size_str)
    print(f"Data size: {data_size_str}, Data dir: {data_dir}")
    assert data_size > 0
    os.makedirs(data_dir, exist_ok=True)

    torch.cuda.cudart().cudaProfilerStart()
    try:
      torch.cuda.set_device(f"cuda:{int(os.environ['RANK']) % torch.cuda.device_count()}")
      torch.distributed.init_process_group("nccl")
      init_nvshmem()
      world_size = torch.distributed.get_world_size()
      rank = torch.distributed.get_rank()

      accum_dtype = torch.float32
      grad_dtype = torch.float32

      reduction_service = ReductionService(accumulation_dtype=accum_dtype)
      cnt = 10
      times = 10
    #   size = 64 * (1000 ** 2)
      size = data_size // accum_dtype.itemsize
      # cnt = 1
      # times = 2
      # size = 16 * (1000 ** 0)
      comp_sizes = torch.rand(cnt).tolist()

      
      
      
      
      

      group_count = 1
      
      for i in range(group_count):
        group_ranks_ = range(i, world_size, group_count)
        group_ = torch.distributed.new_group(ranks=group_ranks_, backend="nccl")
        if rank in group_ranks_:
          group_ranks = group_ranks_
          group = group_
      group_size = len(group_ranks)
      print(f"Rank {rank} group: {group_ranks}")

      data = [
        torch.rand(size, dtype=grad_dtype, device="cuda")
        for _ in range(cnt * times)
      ]
      # data = torch.arange(cnt * times * size, dtype=grad_dtype, device="cuda").reshape(cnt * times, size) / group_size / times
      # print(f"Rank {rank} data: {data}")
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

      def reduce_scatter_accumulation_nccl_comm(src_tensor, dest_idx, pg: dist.ProcessGroup):
        reduction_service.reduce_scatter_accumulation_nccl_comm(dest_idx, src_tensor, pg)

    #   for dst_idx in range(cnt):
    #       reduction_service.pre_register(dst_idx, data[dst_idx], group)
    #   torch.cuda.synchronize()
    #   print(f"Rank {rank} pre_register done")

      dist.barrier()
      torch.cuda.synchronize()
      # warmup
    #   for reduce_scatter_func in [reduce_scatter_accumulation_nccl, reduce_scatter_accumulation_nccl_comm, reduce_scatter_accumulation]:
      for reduce_scatter_func in [reduce_scatter_accumulation_nccl, reduce_scatter_accumulation]:
        reduction_service.clear_accumulations()
        torch.cuda.current_stream().synchronize()
        with torch.cuda.nvtx.range(reduce_scatter_func.__name__):
          for i in range(cnt * times):
            dst_idx = i % cnt
            reduce_scatter_func(data[i], dst_idx, group)
            compute_buffer[dst_idx] @ compute_param
          if reduce_scatter_func == reduce_scatter_accumulation or reduce_scatter_func == reduce_scatter_accumulation_nccl_comm:
            reduction_service.sync(group)
            for i in range(cnt):
              torch.testing.assert_close(nccl_accumulations[i], reduction_service.accumulations[i], rtol=5e-3, atol=5e-3)
          dist.barrier()
          torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
      for i in range(cnt):
          nccl_accumulations[i].zero_()
    #   for reduce_scatter_func in [reduce_scatter_accumulation_nccl, reduce_scatter_accumulation_nccl_comm, reduce_scatter_accumulation]:
      for reduce_scatter_func in [reduce_scatter_accumulation_nccl, reduce_scatter_accumulation]:
        reduction_service.clear_accumulations()
        with torch.cuda.nvtx.range(reduce_scatter_func.__name__):
          start_events = [torch.cuda.Event(enable_timing=True) for _ in range(cnt * times)]
          comm_events = [torch.cuda.Event(enable_timing=True) for _ in range(cnt * times)]
          compute_events = [torch.cuda.Event(enable_timing=True) for _ in range(cnt * times)]
          start = torch.cuda.Event(enable_timing=True)
          
          start.record()
          for i in range(cnt * times):
            dst_idx = i % cnt
            # if i == cnt:
            #   start.record()
            
            # dst_arr = [
            #   dst[r * size:(r + 1) * size]
            #   for r in range(world_size)
            # ]
            start_events[i].record()
            reduce_scatter_func(data[i], dst_idx, group)
            comm_events[i].record()
            # compute_buffer[dst_idx] @ compute_param
            compute_events[i].record()
            
            # print(dst)
          end = torch.cuda.Event(enable_timing=True)
          end.record()
          
          
          
          if reduce_scatter_func == reduce_scatter_accumulation or reduce_scatter_func == reduce_scatter_accumulation_nccl_comm:
            reduction_service.sync(group)
            for i in range(cnt):
              # print(f"Rank {rank} nccl_accumulations: {nccl_accumulations[i]} reduction_service: {reduction_service.accumulations[i]}")
              torch.testing.assert_close(nccl_accumulations[i], reduction_service.accumulations[i], rtol=5e-3, atol=5e-3)
            # print(f"Rank {rank} reduction_service: {reduction_service.buffers[0][0]}")
          else:
            pass
            # print(f"Rank {rank} nccl_accumulations: {nccl_accumulations[0]}")
          dist.barrier()
          torch.cuda.synchronize()
          # print(f"Rank {rank} comm time: {[start_events[i].elapsed_time(comm_events[i]) for i in range(cnt * times)]}, compute time: {[comm_events[i].elapsed_time(compute_events[i]) for i in range(cnt * times)]}")
          reduce_scatter_payload = size // group_size* (group_size - 1)* data[0].dtype.itemsize
          print(f"Rank {rank} {reduce_scatter_func.__name__} reduce_scatter bw: {reduce_scatter_payload / 1024 ** 2 * (cnt * (times - 0)) / start.elapsed_time(end)}")
          print(f"Rank {rank} {reduce_scatter_func.__name__} Total time: {start.elapsed_time(end)}")
          # print(f"Rank {rank} dst: {dst}")
        profile_data = {
            "payload": reduce_scatter_payload,
            "comm_time": [start_events[i].elapsed_time(comm_events[i]) for i in range(cnt * times)],
            "total_time": start.elapsed_time(end),
        }
        with open(os.path.join(data_dir, f"{reduce_scatter_func.__name__}-{data_size}-{rank}.json"), "w") as f:
            json.dump(profile_data, f)

      reduction_service.stop()

    except Exception as e:
      print(e)
      import traceback
      traceback.print_exc()
    finally:
      SymmBufferRegistry.get_instance().finalize()
    torch.cuda.cudart().cudaProfilerStop()
