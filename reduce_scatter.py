import math
from typing import Mapping, Tuple
from functools import reduce

import nvshmem.core
import torch
import os
from cuda import cuda

import triton
import triton.language as tl
# from triton_dist.language.extra import libshmem_device
# from triton.language.extra.cuda.language_extra import __syncthreads, tid
# from triton_dist.utils import (CUDA_CHECK, dist_print, initialize_distributed, nvshmem_barrier_all_on_stream,
#                                NVSHMEM_SIGNAL_DTYPE, nvshmem_create_tensors, nvshmem_create_tensor, nvshmem_free_tensor_sync)
import nvshmem_triton
from nvshmem_triton import tid, __syncthreads, NVSHMEM_EXTERN_LIBS
from typing import List
import time
from dataclasses import dataclass
from odc.utils import SymmBufferRegistry, init_nvshmem, get_same_local_rank_pg, get_local_world_size, get_local_world_pg, get_comm_stream, BufferSplitter, nvshmem_create_tensor
import torch.distributed as dist
from collections import defaultdict


MAX_REQUEST_COUNT = 2 * 100000


@triton.jit(do_not_specialize=["server_rank", "command", "request_id"])
def nvshmem_request_wait_kernel(
  request_buffer_ptr,
  response_buffer_ptr,
  client_rank,
  server_rank,
  command,
  request_id):
  pid = tl.program_id(axis=0)
  tidx = tid(axis=0)
  # if pid == 0 and tidx == 0:
  #   nvshmem_triton.int_p(request_buffer_ptr + client_rank, command, server_rank)
  # nvshmem_triton.quiet()
  # __syncthreads()
  if pid == 0 and tidx == 0:
    r=request_id-1
    while r != request_id:
      nvshmem_triton.quiet()
      r=nvshmem_triton.int_g(response_buffer_ptr + client_rank, server_rank)
  __syncthreads()


@triton.jit(do_not_specialize=["next_request_id", "accumulation_command", "need_accumulation"])
def nvshmem_reduce_scatter_kernel(
    input_tensor_ptr,
    output_tensor_ptr,
    request_buffer_ptr,
    response_buffer_ptr,
    elem_per_rank,
    size_per_elem,
    rank,
    world_size,
    chunk_size,
    next_request_id,
    accumulation_command,
    signal_ptr,
    need_accumulation,
):
    pid = tl.program_id(axis=0)
    tidx = tid(axis=0)
    np = tl.num_programs(axis=0)
    num_nodes = world_size // np
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
          nvshmem_triton.putmem_nbi_block(
              output_tensor_ptr + (chunk * chunk_size),
              input_tensor_ptr + peer * elem_per_rank + (chunk * chunk_size),
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
              nvshmem_triton.quiet()
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

    if pid == 0 and need_accumulation:
        if tidx == 0:
          nvshmem_triton.quiet()
        __syncthreads()


        if tidx == 0:
            for peer in range(world_size):
              nvshmem_triton.int_p(request_buffer_ptr + rank, accumulation_command, peer)
        __syncthreads()

        

        if tidx == 0:
          for peer in range(world_size):
              r=next_request_id-1
              while r != next_request_id:
                nvshmem_triton.quiet()
                r=nvshmem_triton.int_g(response_buffer_ptr + rank, peer)
        __syncthreads()

@triton.jit(do_not_specialize=["next_request_id", "accumulation_command"])
def pre(
    input_tensor_ptr,
    output_tensor_ptr,
    request_buffer_ptr,
    response_buffer_ptr,
    elem_per_rank,
    size_per_elem,
    rank,
    world_size,
    chunk_size,
    next_request_id,
    accumulation_command,
):
    pid = tl.program_id(axis=0)
    tidx = tid(axis=0)
    peer = (pid + rank + 1) % world_size
    # chunk_size = elem_per_rank // num_chunks
    num_chunks = elem_per_rank // chunk_size

    if tidx == 0:
      nvshmem_triton.int_p(request_buffer_ptr + rank, -1, peer)
    nvshmem_triton.quiet()
    __syncthreads()

    if tidx == 0:
      r=next_request_id-1
      while r != next_request_id:
        nvshmem_triton.quiet()
        r=nvshmem_triton.int_g(response_buffer_ptr + rank, peer)
    __syncthreads()
    
@triton.jit(do_not_specialize=["next_request_id", "accumulation_command"])
def push(
    input_tensor_ptr,
    output_tensor_ptr,
    request_buffer_ptr,
    response_buffer_ptr,
    elem_per_rank,
    size_per_elem,
    rank,
    world_size,
    chunk_size,
    next_request_id,
    accumulation_command,
):
    pid = tl.program_id(axis=0)
    tidx = tid(axis=0)
    peer = (pid + rank + 1) % world_size
    num_chunks = elem_per_rank // chunk_size
    for chunk in range(num_chunks):
        nvshmem_triton.putmem_block(
            output_tensor_ptr + (chunk * chunk_size),
            input_tensor_ptr + peer * elem_per_rank + (chunk * chunk_size),
            chunk_size * size_per_elem,
            peer,
        )
    # nvshmem_triton.quiet()
    # nvshmem_triton.putmem_nbi_block(
    #     input_tensor_ptr + peer * elem_per_rank,
    #     output_tensor_ptr,
    #     elem_per_rank * size_per_elem,
    #     peer,
    # )
    # nvshmem_triton.quiet()

@triton.jit(do_not_specialize=["next_request_id", "accumulation_command"])
def post(
    input_tensor_ptr,
    output_tensor_ptr,
    request_buffer_ptr,
    response_buffer_ptr,
    elem_per_rank,
    size_per_elem,
    rank,
    world_size,
    chunk_size,
    next_request_id,
    accumulation_command,
):
    pid = tl.program_id(axis=0)
    tidx = tid(axis=0)
    peer = (pid + rank + 1) % world_size
    # chunk_size = elem_per_rank // num_chunks
    num_chunks = elem_per_rank // chunk_size
    if tidx == 0:
      nvshmem_triton.int_p(request_buffer_ptr + rank, accumulation_command, peer)
    nvshmem_triton.quiet()
    __syncthreads()

    next_request_id += 1

    if tidx == 0:
      r=next_request_id-1
      while r != next_request_id:
        nvshmem_triton.quiet()
        r=nvshmem_triton.int_g(response_buffer_ptr + rank, peer)
    __syncthreads()



@dataclass
class ClientContext:
    request_buffer: torch.Tensor
    response_buffer: torch.Tensor
    next_request_id: int

client_context = None
cmd_buffer = {}
def remote_request(client_context, server_rank, command):
    assert get_local_world_size() == torch.distributed.get_world_size()
    if cmd_buffer.get(command, None) is None:
        cmd_buffer[command] = torch.tensor([command], device="cuda", dtype=client_context.request_buffer.dtype)
    # SymmBufferRegistry.get_instance().get_peer_tensors(client_context.request_buffer)[server_rank][torch.distributed.get_rank():torch.distributed.get_rank()+1].copy_(cmd_buffer[command])
    SymmBufferRegistry.get_instance().get_peer_tensors(client_context.request_buffer)[server_rank][torch.distributed.get_rank()]=(cmd_buffer[command][0])
    with torch.cuda.nvtx.range(f"remote_request {server_rank} cmd {command}"):
        nvshmem_request_wait_kernel[(1, )](
            client_context.request_buffer,
            client_context.response_buffer,
            client_rank=torch.distributed.get_rank(),
            server_rank=server_rank,
            command=command,
            request_id=client_context.next_request_id[server_rank])
    client_context.next_request_id[server_rank] += 1

@dataclass
class ServerContext:
    request_buffer: torch.Tensor
    response_buffer: torch.Tensor
    next_request_id: list[int]
    accumulation_start: Mapping[Tuple[int, int], int]

def ack(server_context, client_rank):
    server_context.request_buffer[client_rank] = 0
    server_context.response_buffer[client_rank] = server_context.next_request_id[client_rank]
    server_context.next_request_id[client_rank] += 1
    if server_context.next_request_id[client_rank] > MAX_REQUEST_COUNT:
        server_context.next_request_id[client_rank] = 1

def server_loop(server_context, dispatch_func, exit_predicate, client_mask=set()):
    request_buffer_cpu = torch.empty_like(server_context.request_buffer, device="cpu").pin_memory()
    while True:
        request_buffer_cpu.copy_(server_context.request_buffer)
        nonzeros = torch.nonzero(request_buffer_cpu, as_tuple=False).squeeze(1).tolist()
        time.sleep(1/10000)
        for client_rank in nonzeros:
            if len(client_mask) > 0 and client_rank not in client_mask:
                continue
            command = request_buffer_cpu[client_rank].item()
            assert isinstance(client_rank, int)
            assert isinstance(command, int)
            acked = dispatch_func(client_rank, command)
            if not acked:
                with torch.cuda.nvtx.range(f"ack {client_rank} cmd {command}"):
                    ack(server_context, client_rank)
        if exit_predicate():
            break

def test_request_response():
    request_buffer = nvshmem_create_tensor(1024, torch.int32)
    response_buffer = nvshmem_create_tensor(1024, torch.int32)
    request_buffer_cpu = torch.empty_like(request_buffer, device="cpu").pin_memory()
    torch.distributed.barrier()
    torch.cuda.synchronize()
    num_requests = 10000
    if torch.distributed.get_rank() != 0:
      client_context = ClientContext(request_buffer, response_buffer, 1)
      for i in range(num_requests):
        remote_request(client_context, 0, i % 2 + 1)
    else:
      server_context = ServerContext(request_buffer, response_buffer, [1] * 1024, defaultdict(lambda: 0))
      server_loop(server_context, lambda client_rank, command: False, lambda: min(server_context.next_request_id[1:torch.distributed.get_world_size()]) == num_requests + 1)
      # done_counts = torch.zeros_like(request_buffer, device="cpu")
      # while True:
      #   request_buffer_cpu.copy_(request_buffer)
      #   nonzeros = torch.nonzero(request_buffer_cpu, as_tuple=False).squeeze(1).tolist()
      #   for client_rank in nonzeros:
      #     command = request_buffer_cpu[client_rank]
      #     print(f"Rank {torch.distributed.get_rank()} received command {command} from client {client_rank}")
      #     done_counts[client_rank] += 1
      #     request_buffer[client_rank] = 0
      #     response_buffer[client_rank] = done_counts[client_rank]
      #   if done_counts[1:8].min() == 10000:
      #       break
    torch.distributed.barrier()
    torch.cuda.synchronize()

class DistLock:
    def __init__(self, world_size):
        self.world_size = world_size
        self.request_buffer = SymmBufferRegistry.get_instance().allocate_symm_buffer('request_buffer', (self.world_size,), torch.int32)
        self.response_buffer = SymmBufferRegistry.get_instance().allocate_symm_buffer('response_buffer', (self.world_size,), torch.int32)
        self.request_buffer.fill_(0)
        self.response_buffer.fill_(0)
        self.client_context = ClientContext(self.request_buffer, self.response_buffer, 1)
        

    def lock(self, target_rank):
        assert target_rank < self.world_size
        remote_request(self.client_context, target_rank, -1)
    
    def notify_data(self, target_rank, buffer_id, accumulation_id):
        assert accumulation_id > 0
        assert buffer_id < 2** 10
        assert accumulation_id < 2** 10
        command = (buffer_id << 16) | accumulation_id
        remote_request(self.client_context, target_rank, command)

class ReductionWatcher:
    def __init__(self, world_size, accumulations: List[torch.Tensor], buffers: List[torch.Tensor], request_buffer: torch.Tensor, response_buffer: torch.Tensor):
        self.accumulations = accumulations
        self.buffers = buffers
        self.request_buffer = request_buffer
        self.response_buffer = response_buffer
        self.world_size = world_size
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
        rank = os.environ['RANK']
        def dispatch_func(client_rank, command):
            if command == -1:
                # client_mask.add(client_rank)
                return False
            else:
                buffer_id = command >> 16
                accumulation_id = command & 0xFFFF

                acc = self.accumulations[accumulation_id - 1][0]
                buf = self.buffers[buffer_id][client_rank]
                start = self.server_context.accumulation_start[(buffer_id, client_rank)]
                size = min(buf.numel(), acc.numel() - start)
                with torch.cuda.nvtx.range(f"add client {client_rank} buffer {buffer_id} accumulation {accumulation_id}"):
                    acc[start:start+size].add_(buf[:size])
                if start + size >= acc.numel():
                    assert start + size == acc.numel()
                    self.server_context.accumulation_start[(buffer_id, client_rank)] = 0
                else:
                    self.server_context.accumulation_start[(buffer_id, client_rank)] += size
                torch.cuda.current_stream().synchronize()
                self.task_count += 1
                # client_mask.remove(client_rank)
                return False
        def exit_predicate():
            return not self.running
        self.server_context = ServerContext(self.request_buffer, self.response_buffer, [1] * self.world_size, defaultdict(lambda: 0))
        client_mask = set()
        server_loop(self.server_context, dispatch_func, exit_predicate, client_mask)

def tensor_from_handle(handle, size, dtype):
    from tensor_ipc import reconstruct_tensor
    return reconstruct_tensor(handle, (size,), dtype)

def reduction_watcher_function(device_id, world_size, accumulations, buffers, request_buffer, response_buffer, cmd_queue, response_queue):
    torch.cuda.set_device(device_id)
    import sys
    # torch.cuda.cudart().cudaProfilerStart()
    buffers = [tensor_from_handle(*buffer) for buffer in buffers]
    accumulations = [tensor_from_handle(*acc) for acc in accumulations]
    request_buffer = tensor_from_handle(*request_buffer)
    response_buffer = tensor_from_handle(*response_buffer)

    watcher = ReductionWatcher(world_size, accumulations, buffers, request_buffer, response_buffer)

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

def start_reduction_watcher(accumulations, buffers, request_buffer, response_buffer):
    from torch.multiprocessing import Process

    original_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if original_visible_devices is not None:
        del os.environ['CUDA_VISIBLE_DEVICES']
    
    ctx = torch.multiprocessing.get_context("spawn")
    cmd_queue = ctx.Queue()
    response_queue = ctx.Queue()
    device_id = torch.distributed.get_rank() % get_local_world_size()
    world_size = torch.distributed.get_world_size()
    process = ctx.Process(target=reduction_watcher_function,
                       args=(device_id, world_size, accumulations, buffers, request_buffer, response_buffer, cmd_queue, response_queue))
    process.start()
    if original_visible_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = original_visible_devices
    elif 'CUDA_VISIBLE_DEVICES' in os.environ:
        del os.environ['CUDA_VISIBLE_DEVICES']
    return cmd_queue, response_queue

def call_watcher(watcher_handle, cmd, *args):
    cmd_queue, response_queue = watcher_handle
    cmd_queue.put((cmd, *args))
    return response_queue.get()

def get_nvshmem_handle(tensor):
    from tensor_ipc import get_ipc_handle
    print(f"Rank {torch.distributed.get_rank()} get_nvshmem_handle {tensor.data_ptr()} with shape {tensor.shape} and dtype {tensor.dtype}")
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
        self.input_buffer = {}
        self.dispatched_tasks = 0
        self.accumulation_dtype = accumulation_dtype
        self.rank_streams = defaultdict(lambda: torch.cuda.Stream())
        self.buffer_shape_cache = {}
        self.buffer_splitter = BufferSplitter()

    def register(self, key, output_tensor_shape, grad_dtype,reduction_dtype):
        if self.reduction_watcher is None:
            self.lock = DistLock(torch.distributed.get_world_size())
            request_buffer_handle = get_nvshmem_handle(self.lock.request_buffer)
            response_buffer_handle = get_nvshmem_handle(self.lock.response_buffer)

            # Make sure changes are visible to all reduction watchers
            torch.distributed.barrier()
            torch.cuda.synchronize()

            self.reduction_watcher = start_reduction_watcher([], [], request_buffer_handle, response_buffer_handle)

        buffer_key = f'rs_buffer_{key}'
        accumulation_key = f'rs_accumulation_{key}'
        assert len(output_tensor_shape) == 1
        registry = SymmBufferRegistry.get_instance()
        assert not registry.has_key(accumulation_key)
        # assert self.reduction_watcher is None, "Reduction watcher is already running"
        
        def create_and_register_accumulation(key, shape, dtype, add_func):
            buffer = registry.allocate_symm_buffer(key, shape, dtype)
            call_watcher(self.reduction_watcher, add_func, [get_nvshmem_handle(buffer)])
            return buffer
        
        def create_and_register_buffer(key, shape, dtype, add_func):
            buffers = []
            for rank in range(torch.distributed.get_world_size()):
                buffer = registry.allocate_symm_buffer(f'{key}_rank_{rank}', shape, dtype)
                buffers.append(buffer)
            call_watcher(self.reduction_watcher, add_func, [get_nvshmem_handle(b) for b in buffers])
            return buffers
            
           
        # acc = registry.allocate_symm_buffer(accumulation_key, output_tensor_shape, reduction_dtype)
        acc = create_and_register_accumulation(accumulation_key, output_tensor_shape, reduction_dtype, 'add_accumulation')
        self.accumulation_indices[key] = len(self.accumulations)
        self.accumulations.append(acc)
        
        world_size = torch.distributed.get_world_size()
        if os.getenv('ODC_RS_DISABLE_SPLIT_TRANS_BUFFER', '0') != '1':
            buffer_size = self.buffer_splitter.get_local_buffer_size(output_tensor_shape, world_size)
            buffer_shape = (buffer_size,)
        else:
            buffer_shape = output_tensor_shape

        shared_buffer_key = (grad_dtype, buffer_shape)
        if shared_buffer_key not in self.shared_buffer:
            output_size = reduce(lambda x, y: x * y, output_tensor_shape)
            buffer_size = reduce(lambda x, y: x * y, buffer_shape)
            print(f"Rank {torch.distributed.get_rank()} create buffer: output_size: {output_size} num_sub_buffers: {math.ceil(output_size / buffer_size)} buffer_size: {buffer_size}")
            cnt = len(self.shared_buffer)
            buffers = create_and_register_buffer(f'shared_buffer_{cnt}', buffer_shape, grad_dtype, 'add_buffer')
            self.shared_buffer[shared_buffer_key] = (cnt, buffers)
            
            self.buffers.append(buffers)
        self.buffer_indices[key] = self.shared_buffer[shared_buffer_key][0]

        # Make sure changes are visible to all reduction watchers
        torch.distributed.barrier()
        torch.cuda.synchronize()

    def clear_accumulations(self):
        for acc in self.accumulations:
            acc.fill_(0)
    
    def infer_output_shape(self, input_tensor, pg: dist.ProcessGroup):
        assert len(input_tensor.shape) == 1
        assert input_tensor.shape[0] % dist.get_world_size(pg) == 0
        return (input_tensor.shape[0] // dist.get_world_size(pg),)

    def reduce_scatter_accumulation(self, key, input_tensor, pg: dist.ProcessGroup):
        output_tensor_shape = self.infer_output_shape(input_tensor, pg)
        accum_dtype = self.accumulation_dtype if self.accumulation_dtype is not None else input_tensor.dtype
        if key not in self.accumulation_indices:
            self.register(key, output_tensor_shape, input_tensor.dtype, accum_dtype)
        
        world_size = torch.distributed.get_world_size(pg)
        local_buf_size = self.buffer_splitter.get_local_buffer_size(output_tensor_shape, world_size)
        output_size = reduce(lambda x, y: x * y, output_tensor_shape)
        assert local_buf_size <= output_size

        input_buf_size = local_buf_size * world_size
        input_tensor_symm_shape = (input_buf_size,)
        if (input_tensor_symm_shape, input_tensor.dtype) not in self.input_buffer:
            self.input_buffer[(input_tensor_symm_shape, input_tensor.dtype)] = SymmBufferRegistry.get_instance().allocate_symm_buffer(
                f'rs_buffer_{input_tensor_symm_shape}_{input_tensor.dtype}', input_tensor_symm_shape, input_tensor.dtype)
        input_tensor_symm = self.input_buffer[(input_tensor_symm_shape, input_tensor.dtype)]

        acc = self.accumulations[self.accumulation_indices[key]]
        buffer = self.buffers[self.buffer_indices[key]][torch.distributed.get_rank(pg)]
        buffer_id = self.buffer_indices[key]
        accumulation_id = self.accumulation_indices[key] + 1

        accumulation_command = (buffer_id << 16) | accumulation_id
        assert (buffer.numel() * buffer.element_size()) % (2**6) == 0, 'better align to 64 for efficiency'
        chunk_size = (2**20 // buffer.element_size())
        # chunk_size = buffer.numel()
        # assert buffer.shape[0] % chunk_size == 0

        get_comm_stream().wait_stream(torch.cuda.current_stream())

        split_trans_buffer = buffer.numel() < output_size
        with torch.cuda.stream(get_comm_stream()):
            input_tensor_split = input_tensor.view(-1).view(world_size, -1)
            signal_ptr = torch.empty(1, dtype=torch.int32, device="cuda")
            for start in range(0, output_size, local_buf_size):
                size = min(local_buf_size, output_size - start)
                input_size = size * world_size
                input_tensor_symm_split = input_tensor_symm[:input_size].view(world_size, -1)
                assert local_buf_size <= buffer.numel()
                if local_buf_size < output_size:
                    for r in range(world_size):
                        input_tensor_symm_split[r, :].copy_(input_tensor_split[r, start:start+size])
                else:
                    input_tensor_symm.copy_(input_tensor.view(-1))
                if split_trans_buffer:
                    buf = buffer[:size]
                else:
                    buf = buffer[start:start+size]
                signal_ptr.fill_(0)
                need_accumulation = split_trans_buffer or start + size == output_size
                assert world_size % 8 == 0 or world_size < 8
                grid_size = 8 if world_size == 32 else world_size
                nvshmem_reduce_scatter_kernel[(grid_size, )](
                    input_tensor_ptr = input_tensor_symm_split.view(-1),
                    output_tensor_ptr = buf,
                    request_buffer_ptr = self.lock.request_buffer,
                    response_buffer_ptr = self.lock.response_buffer,
                    elem_per_rank = size,
                    size_per_elem = buf.element_size(),
                    rank = torch.distributed.get_rank(pg),
                    world_size = world_size,
                    chunk_size = chunk_size,
                    next_request_id = self.lock.client_context.next_request_id,
                    accumulation_command = accumulation_command,
                    signal_ptr = signal_ptr,
                    need_accumulation = need_accumulation,
                    num_warps=32,
                    extern_libs=NVSHMEM_EXTERN_LIBS,
                )
                if need_accumulation:
                    self.lock.client_context.next_request_id += 1
                    if self.lock.client_context.next_request_id > MAX_REQUEST_COUNT:
                        self.lock.client_context.next_request_id = 1
                    self.dispatched_tasks += 1
        torch.cuda.current_stream().wait_stream(get_comm_stream())
        # nvshmem.core.quiet(stream=torch.cuda.current_stream())
        # pre[(world_size, )](
        #   input_tensor_ptr = input_tensor_symm,
        #   output_tensor_ptr = buffer,
        #   request_buffer_ptr = self.lock.request_buffer,
        #   response_buffer_ptr = self.lock.response_buffer,
        #   elem_per_rank = buffer.numel(),
        #   size_per_elem = buffer.element_size(),
        #   rank = torch.distributed.get_rank(pg),
        #   world_size = world_size,
        #   chunk_size = chunk_size,
        #   next_request_id = self.lock.client_context.next_request_id,
        #   accumulation_command = accumulation_command,
        # )
        # push[(world_size, )](
        #   input_tensor_ptr = input_tensor_symm,
        #   output_tensor_ptr = buffer,
        #   request_buffer_ptr = self.lock.request_buffer,
        #   response_buffer_ptr = self.lock.response_buffer,
        #   elem_per_rank = buffer.numel(),
        #   size_per_elem = buffer.element_size(),
        #   rank = torch.distributed.get_rank(pg),
        #   world_size = world_size,
        #   chunk_size = chunk_size,
        #   next_request_id = self.lock.client_context.next_request_id,
        #   accumulation_command = accumulation_command,
        #   num_warps=32,
        # )
        # post[(world_size, )](
        #   input_tensor_ptr = input_tensor_symm,
        #   output_tensor_ptr = buffer,
        #   request_buffer_ptr = self.lock.request_buffer,
        #   response_buffer_ptr = self.lock.response_buffer,
        #   elem_per_rank = buffer.numel(),
        #   size_per_elem = buffer.element_size(),
        #   rank = torch.distributed.get_rank(pg),
        #   world_size = world_size,
        #   chunk_size = chunk_size,
        #   next_request_id = self.lock.client_context.next_request_id,
        #   accumulation_command = accumulation_command,
        # )
        
    def get_accumulation(self, key):
        acc = self.accumulations[self.accumulation_indices[key]]
        return acc
    
    def sync(self, pg: dist.ProcessGroup):
        # TODO: This actually only syncs CPU of reduction workers, it's possible that the last reduction is on-the-fly.
        dispatched_task_list = [None for _ in range(dist.get_world_size(pg))]
        torch.distributed.all_gather_object(dispatched_task_list, self.dispatched_tasks, group=pg)
        torch.cuda.synchronize()

        target = sum(dispatched_task_list)
        call_watcher(self.reduction_watcher, 'wait_and_reset_task_count', target)
        self.dispatched_tasks = 0

    def stop(self):
        if self.reduction_watcher is not None:
            call_watcher(self.reduction_watcher, 'stop')
            torch.distributed.barrier()
            torch.cuda.synchronize()



if __name__ == "__main__":
    import os
    torch.cuda.cudart().cudaProfilerStart()
    try:
      device_id = int(os.environ['RANK']) % torch.cuda.device_count()
      device = torch.device(f"cuda:{device_id}")
      torch.cuda.set_device(f"cuda:{int(os.environ['RANK']) % torch.cuda.device_count()}")
      torch.distributed.init_process_group("nccl", device_id=device)
      # torch.distributed.barrier()
      init_nvshmem()
      # test_request_response()
      world_size = torch.distributed.get_world_size()
      rank = torch.distributed.get_rank()

      accum_dtype = torch.float32
      grad_dtype = torch.float32

      reduction_service = ReductionService(accumulation_dtype=accum_dtype)
      cnt = 1
      times = 10
      size = 128 * (1024 ** 2) + 1024
      # cnt = 1
      # times = 2
      # size = 16 * (1000 ** 0)
      # comp_sizes = torch.rand(cnt).tolist()
      comp_sizes = [2]

      
      
      

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

      def some_compute(x):
        return x
        with torch.no_grad():
          x = x @ compute_param
          x = x @ compute_param
          q = x.reshape(1, x.shape[0], 64, 128)
          k = x.reshape(1,x.shape[0], 64, 128)
          v = x.reshape(1,x.shape[0], 64, 128)
          from flash_attn.flash_attn_interface import flash_attn_func
          x = flash_attn_func(q, k, v, causal=True)
          x = x.reshape(-1, 8192)
          for i in range(10):
            x=x * 2
            x=x * 1.5
            x=x + 1
            x=x - 1
            x=x + 0.5
            x=x - 0.5
            x=x + 0.25
            x=x - 0.25
            x=x + 0.125
            x=x - 0.125
          x = x @ compute_param
          return x

      nccl_accumulations = [torch.zeros(size // group_size, dtype=accum_dtype, device="cuda") for _ in range(cnt)]
      def reduce_scatter_accumulation_nccl(src_tensor, dest_idx, pg: dist.ProcessGroup):
        output = torch.empty((src_tensor.numel() // dist.get_world_size(pg),), dtype=src_tensor.dtype, device="cuda")
        torch.distributed.reduce_scatter_tensor(output, src_tensor, op=torch.distributed.ReduceOp.SUM, group=pg)
        nccl_accumulations[dest_idx].add_(output)

      def reduce_scatter_accumulation(src_tensor, dest_idx, pg: dist.ProcessGroup):
        reduction_service.reduce_scatter_accumulation(dest_idx, src_tensor, pg)

      def reduce_scatter_accumulation_nccl_comm(src_tensor, dest_idx, pg: dist.ProcessGroup):
        reduction_service.reduce_scatter_accumulation_nccl_comm(dest_idx, src_tensor, pg)

      dist.barrier()
      torch.cuda.synchronize()
      comp_stream = torch.cuda.Stream()
      for reduce_scatter_func in [reduce_scatter_accumulation_nccl, reduce_scatter_accumulation]:
        reduction_service.clear_accumulations()
        
        with torch.cuda.nvtx.range(reduce_scatter_func.__name__):
          start_events = [torch.cuda.Event(enable_timing=True) for _ in range(cnt * times)]
          comm_events = [torch.cuda.Event(enable_timing=True) for _ in range(cnt * times)]
          compute_events = [torch.cuda.Event(enable_timing=True) for _ in range(cnt * times)]
          start = torch.cuda.Event(enable_timing=True)

          for i in range(cnt * times):
            dst_idx = i % cnt
            torch.distributed.barrier(group)
            
            if i == cnt:
              start.record()
            # dst_arr = [
            #   dst[r * size:(r + 1) * size]
            #   for r in range(world_size)
            # ]
            start_events[i].record()
            comp_stream.wait_stream(torch.cuda.current_stream())
            
            reduce_scatter_func(data[i], dst_idx, group)
            with torch.cuda.stream(comp_stream):
                some_compute(compute_buffer[0])
            torch.cuda.current_stream().wait_stream(comp_stream)
            comm_events[i].record()
            # compute_buffer[dst_idx] @ compute_param
            compute_events[i].record()
            
            # print(dst)
          end = torch.cuda.Event(enable_timing=True)
          end.record()
          
          
          
          if reduce_scatter_func == reduce_scatter_accumulation or reduce_scatter_func == reduce_scatter_accumulation_nccl_comm:
            reduction_service.sync(group)
            for i in range(cnt):
              pass
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
          print(f"Rank {rank} {reduce_scatter_func.__name__} bw: {reduce_scatter_payload / 1024 ** 2 * (cnt * (times - 1)) / start.elapsed_time(end)}")
          print(f"Rank {rank} Total time: {start.elapsed_time(end)}")
          # print(f"Rank {rank} dst: {dst}")
        # torch.cuda.current_stream().wait_stream(comp_stream)

      reduction_service.stop()

    except Exception as e:
      print(e)
      import traceback
      traceback.print_exc()
    finally:
      SymmBufferRegistry.get_instance().finalize()
      torch.distributed.destroy_process_group()
    torch.cuda.cudart().cudaProfilerStop()
