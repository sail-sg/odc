import os
import time
from collections import defaultdict
from functools import reduce
from typing import List

import nvshmem.core
import torch
import torch.distributed as dist
import triton
import triton.language as tl

from odc.primitives import (
    NVSHMEM_EXTERN_LIBS,
    __syncthreads,
    get_ipc_handle,
    int_atomic_compare_swap,
    reconstruct_tensor,
    tid,
)
from odc.primitives.utils import (
    SymmBufferRegistry,
    get_local_world_pg,
    get_local_world_size,
    get_same_local_rank_pg,
    nvshmem_create_tensor,
    nvshmem_free_tensor_sync,
)


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
            r = int_atomic_compare_swap(lock_buffer_ptr + lock_id, 0, -1, target_rank)
    __syncthreads()


@triton.jit(do_not_specialize=["lock_id"])
def reset_lock_kernel(lock_buffer_ptr, lock_id):
    pid = tl.program_id(axis=0)
    tidx = tid(axis=0)
    if pid == 0 and tidx == 0:
        tl.atomic_xchg(lock_buffer_ptr + lock_id, 0)
    __syncthreads()


cmd_buffer = {}


class DistLock:
    def __init__(self, num_locks):
        self.num_locks = num_locks
        self.lock_buffers = nvshmem_create_tensor(self.num_locks, torch.int32)
        self.lock_buffers.fill_(0)
        self.cpu_lock_buffers = torch.empty_like(self.lock_buffers, device="cpu").pin_memory()

    def lock(self, target_rank, buffer_id):
        assert buffer_id < self.num_locks
        # TODO: This is a hack as currently nvshmem doesn't work cross node. So we init nvshmem only within node.
        nvshmem_poll_lock_kernel[(1,)](
            self.lock_buffers,
            target_rank % get_local_world_size(),
            buffer_id,
            num_warps=1,
            extern_libs=NVSHMEM_EXTERN_LIBS,
        )

    def notify_data(self, target_rank, buffer_id, accumulation_id):
        assert buffer_id < self.num_locks
        assert accumulation_id > 0
        if cmd_buffer.get(accumulation_id, None) is None:
            cmd_buffer[accumulation_id] = torch.tensor(
                accumulation_id, device="cuda", dtype=self.lock_buffers.dtype
            )
        peer_tensor = nvshmem.core.get_peer_tensor(
            self.lock_buffers, target_rank % get_local_world_size()
        )
        # print(f"Rank {torch.distributed.get_rank()} notify_data {target_rank} {buffer_id} {accumulation_id} {peer_tensors}")
        # peer_tensor = peer_tensors[target_rank % get_local_world_size()]
        # print(f"Rank {torch.distributed.get_rank()} notify_data {target_rank} {buffer_id} {accumulation_id} {peer_tensor}")
        # peer_tensor[buffer_id] = cmd_buffer[accumulation_id]
        peer_tensor[buffer_id].copy_(cmd_buffer[accumulation_id], non_blocking=True)
        # nvshmem_set_kernel[(1, )](self.lock_buffers, target_rank % get_local_world_size(), buffer_id, accumulation_id)


class ReductionWatcher:
    def __init__(
        self,
        accumulations: List[torch.Tensor],
        buffers: List[torch.Tensor],
        lock_buffers: torch.Tensor,
    ):
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
            # print(f"Rank {torch.cuda.current_device()} waiting for task count {self.task_count} < {expected}")
        self.task_count = 0

    def add_buffer(self, buffers):
        # print(f"Rank {dist.get_rank()} adding buffer {accumulation} {buffer}")
        self.buffers.append([tensor_from_handle(*buffer) for buffer in buffers])

    def add_accumulation(self, accumulations):
        self.accumulations.append([tensor_from_handle(*acc) for acc in accumulations])

    def run(self):
        while self.running:
            self.cpu_lock_buffers.fill_(0)
            time.sleep(1 / 10000)
            # reduction_watcher_kernel[(1, )](self.lock_buffers, self.num_locks, BLOCK_SIZE=block_size)

            self.cpu_lock_buffers.copy_(self.lock_buffers, non_blocking=True)
            torch.cuda.current_stream().synchronize()
            if not self.running:
                break

            nonzeros = torch.nonzero(self.cpu_lock_buffers, as_tuple=False).squeeze(1).tolist()

            for buf_id in nonzeros:
                acc_id = self.cpu_lock_buffers[buf_id]
                if acc_id > 0:
                    # print(f"Rank {torch.cuda.current_device()} adding buffer {buf_id} -> {acc_id - 1}")
                    for acc, buf in zip(self.accumulations[acc_id - 1], self.buffers[buf_id]):
                        acc.add_(buf)
                    self.task_count += 1

                    # print(f"adding buffer {idx} {self.accumulations[idx]} {self.buffers[idx]}")
                    reset_lock_kernel[(1,)](self.lock_buffers, buf_id, num_warps=1)


def tensor_from_handle(handle, size, dtype):
    return reconstruct_tensor(handle, (size,), dtype)


def reduction_watcher_function(
    device_id, accumulations, buffers, lock_buffers, cmd_queue, response_queue
):
    torch.cuda.set_device(device_id)

    # torch.cuda.cudart().cudaProfilerStart()
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
            if cmd == "stop":
                break

    cmd_thread = Thread(target=cmd_thread)
    cmd_thread.start()
    watcher.run()
    cmd_thread.join()


def start_reduction_watcher(accumulations, buffers, lock_buffers):
    # original_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    # if original_visible_devices is not None:
    #     del os.environ['CUDA_VISIBLE_DEVICES']

    ctx = torch.multiprocessing.get_context("spawn")
    cmd_queue = ctx.Queue()
    response_queue = ctx.Queue()
    device_id = torch.distributed.get_rank() % get_local_world_size()
    process = ctx.Process(
        target=reduction_watcher_function,
        args=(device_id, accumulations, buffers, lock_buffers, cmd_queue, response_queue),
    )
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
    handle = get_ipc_handle(tensor)
    return handle, tensor.numel(), tensor.dtype


class ReductionIntraNodeService:
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
        self.rank_streams = defaultdict(torch.cuda.Stream)

    def pre_register(self, key, input_tensor, pg: dist.ProcessGroup):
        assert key not in self.accumulation_indices
        accum_dtype = (
            self.accumulation_dtype if self.accumulation_dtype is not None else input_tensor.dtype
        )
        self.register(
            key, self.infer_output_shape(input_tensor, pg), input_tensor.dtype, accum_dtype, pg
        )

    def register(
        self, key, output_tensor_shape, grad_dtype, reduction_dtype, pg: dist.ProcessGroup
    ):
        if self.reduction_watcher is None:
            self.lock = DistLock(128)
            lock_buffers_handle = get_nvshmem_handle(self.lock.lock_buffers)

            # Make sure changes are visible to all reduction watchers
            torch.distributed.barrier()
            torch.cuda.synchronize()

            self.reduction_watcher = start_reduction_watcher([], [], lock_buffers_handle)

        buffer_key = f"rs_buffer_{key}"
        accumulation_key = f"rs_accumulation_{key}"
        assert len(output_tensor_shape) == 1
        registry = SymmBufferRegistry.get_instance()
        assert not registry.has_key(accumulation_key)
        # assert self.reduction_watcher is None, "Reduction watcher is already running"

        def create_and_register(key, shape, dtype, add_func):
            buffer = registry.allocate_symm_buffer(key, shape, dtype, pg)
            local_handles = []
            for local_tensor in registry.get_local_peer_tensors(buffer):
                local_tensor.fill_(0)
                local_handles.append(get_nvshmem_handle(local_tensor))
            call_watcher(self.reduction_watcher, add_func, local_handles)
            return buffer

        # acc = registry.allocate_symm_buffer(accumulation_key, output_tensor_shape, reduction_dtype)
        acc = create_and_register(
            accumulation_key, output_tensor_shape, reduction_dtype, "add_accumulation"
        )
        self.accumulation_indices[key] = len(self.accumulations)
        self.accumulations.append(acc)

        if (
            os.environ.get("ODC_SINGLE_BUFFER", "0") != "1"
            and os.environ.get("ODC_NUM_BUFFERS", "0") == "0"
        ):
            buffer = create_and_register(buffer_key, output_tensor_shape, grad_dtype, "add_buffer")
            self.buffer_indices[key] = len(self.buffers)
            self.buffers.append(buffer)
        elif os.environ.get("ODC_NUM_BUFFERS", "0") != "0":
            num_buffers = int(os.environ.get("ODC_NUM_BUFFERS", "0"))
            assert num_buffers > 0, f"Invalid ODC_NUM_BUFFERS {num_buffers}"
            fixed_buffer_key = (grad_dtype, output_tensor_shape)
            if fixed_buffer_key not in self.fixed_buffers:
                bufs = []
                for i in range(num_buffers):
                    buffer_key = self.get_fixed_buffer_key(output_tensor_shape, grad_dtype, i)
                    buffer = create_and_register(
                        buffer_key, output_tensor_shape, grad_dtype, "add_buffer"
                    )
                    bufs.append(buffer)
                    self.buffer_indices[buffer_key] = len(self.buffers)
                    self.buffers.append(buffer)
                self.fixed_buffers[fixed_buffer_key] = bufs
        else:
            shared_buffer_key = (grad_dtype, output_tensor_shape)
            if shared_buffer_key not in self.shared_buffer:
                cnt = len(self.shared_buffer)
                buffer = create_and_register(
                    f"shared_buffer_{cnt}", output_tensor_shape, grad_dtype, "add_buffer"
                )
                self.shared_buffer[shared_buffer_key] = (cnt, buffer)

                self.buffers.append(buffer)
            buffer = self.shared_buffer[shared_buffer_key][1]
            self.buffer_indices[key] = self.shared_buffer[shared_buffer_key][0]

        # Make sure changes are visible to all reduction watchers
        torch.distributed.barrier()
        torch.cuda.synchronize()

    def get_fixed_buffer_key(self, output_tensor_shape, grad_dtype, index):
        numel = reduce(lambda x, y: x * y, output_tensor_shape)
        return f"rs_fixed_buffer_{numel}_{grad_dtype}_{index}"

    def rank_to_fixed_buffer_index(self, rank):
        assert "ODC_NUM_BUFFERS" in os.environ, "ODC_NUM_BUFFERS must be set"
        num_buffers = int(os.environ.get("ODC_NUM_BUFFERS", "0"))
        return rank % num_buffers

    def clear_accumulations(self):
        for acc in self.accumulations:
            acc.fill_(0)

    def infer_output_shape(self, input_tensor, pg: dist.ProcessGroup):
        assert len(input_tensor.shape) == 1
        assert input_tensor.shape[0] % dist.get_world_size(pg) == 0
        return (input_tensor.shape[0] // dist.get_world_size(pg),)

    def reduce_scatter_accumulation_nccl_comm(self, key, input_tensor, pg: dist.ProcessGroup):
        output_shape = self.infer_output_shape(input_tensor, pg)
        if key not in self.accumulation_indices:
            accum_dtype = (
                self.accumulation_dtype
                if self.accumulation_dtype is not None
                else input_tensor.dtype
            )
            self.register(key, output_shape, input_tensor.dtype, accum_dtype, pg)

        acc = self.accumulations[self.accumulation_indices[key]]
        # buffer = self.buffers[self.buffer_indices[key]]
        if os.environ.get("ODC_NUM_BUFFERS", "0") != "0":
            index = self.rank_to_fixed_buffer_index(torch.distributed.get_rank())
            buffer_key = self.get_fixed_buffer_key(output_shape, input_tensor.dtype, index)
            assert (
                buffer_key in self.buffer_indices
            ), f"Buffer key {buffer_key} not found in buffer_indices {self.buffer_indices} fixed_buffers {self.fixed_buffers}"
            buffer_id = self.buffer_indices[buffer_key]
        else:
            buffer_id = self.buffer_indices[key]
        buffer = self.buffers[buffer_id]
        assert buffer.numel() == reduce(lambda x, y: x * y, output_shape)
        buffer = buffer.view(*output_shape)

        local_peer_accs = SymmBufferRegistry.get_instance().get_local_peer_tensors(acc)

        size = self.infer_output_shape(input_tensor, pg)[0]

        local_world_pg = get_local_world_pg(pg)
        local_world_size = torch.distributed.get_world_size(group=local_world_pg)
        assert len(local_peer_accs) * local_world_size == torch.distributed.get_world_size(group=pg)
        for i in range(0, torch.distributed.get_world_size(group=pg), local_world_size):
            src_tensor = input_tensor[i * size : (i + local_world_size) * size]
            torch.distributed.reduce_scatter_tensor(buffer, src_tensor, group=local_world_pg)
            local_peer_accs[i // local_world_size].add_(buffer)

    def reduce_scatter_accumulation(self, key, input_tensor, pg: dist.ProcessGroup):
        if key not in self.accumulation_indices:
            accum_dtype = (
                self.accumulation_dtype
                if self.accumulation_dtype is not None
                else input_tensor.dtype
            )
            self.register(
                key, self.infer_output_shape(input_tensor, pg), input_tensor.dtype, accum_dtype, pg
            )

        # acc = self.accumulations[self.accumulation_indices[key]]

        if os.environ.get("ODC_NUM_BUFFERS", "0") != "0":
            index = self.rank_to_fixed_buffer_index(torch.distributed.get_rank())
            buffer_key = self.get_fixed_buffer_key(
                self.infer_output_shape(input_tensor, pg), input_tensor.dtype, index
            )
            assert (
                buffer_key in self.buffer_indices
            ), f"Buffer key {buffer_key} not found in buffer_indices {self.buffer_indices} fixed_buffers {self.fixed_buffers}"
            buffer_id = self.buffer_indices[buffer_key]
        else:
            buffer_id = self.buffer_indices[key]
        buffer = self.buffers[buffer_id]

        peer_buffers = SymmBufferRegistry.get_instance().get_peer_tensors(buffer)

        group_size = dist.get_world_size(pg)
        group_ranks = dist.get_process_group_ranks(pg)
        accumulation_id = self.accumulation_indices[key] + 1

        size = buffer.numel()
        assert input_tensor.numel() == size * group_size

        local_world_size = get_local_world_size()

        events = []

        rank_info = [
            (r, group_ranks[r], group_ranks[r] % local_world_size) for r in range(group_size)
        ]

        for local_r_offset in range(0, local_world_size):
            dst_local_rank = (torch.distributed.get_rank() + local_r_offset) % local_world_size
            dst_rank_infos = [r for r in rank_info if r[2] == dst_local_rank]
            matching_rank_in_local_world = (
                torch.distributed.get_rank() // local_world_size * local_world_size + dst_local_rank
            )

            if len(dst_rank_infos) == 0:
                continue

            stream = self.rank_streams[dst_local_rank]
            event = torch.cuda.Event()
            events.append(event)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self.lock.lock(target_rank=matching_rank_in_local_world, buffer_id=buffer_id)

                for dst_group_idx, _dst_rank, dst_local_rank in dst_rank_infos:
                    dst_buffer = peer_buffers[dst_group_idx]
                    dst_buffer.copy_(
                        input_tensor[dst_group_idx * size : (dst_group_idx + 1) * size],
                        non_blocking=True,
                    )
                self.lock.notify_data(
                    target_rank=matching_rank_in_local_world,
                    buffer_id=buffer_id,
                    accumulation_id=accumulation_id,
                )
                event.record()

            # self.lock.lock(target_rank=matching_rank_in_local_world, buffer_id=buffer_id)
            # for (dst_group_idx, dst_rank, dst_local_rank) in dst_rank_infos:
            #     dst_buffer = peer_buffers[dst_group_idx]
            #     dst_buffer.copy_(input_tensor[dst_group_idx * size:(dst_group_idx + 1) * size], non_blocking=True)
            # self.lock.notify_data(target_rank=matching_rank_in_local_world, buffer_id=buffer_id, accumulation_id=accumulation_id)
        self.dispatched_tasks += 1
        for event in events:
            torch.cuda.current_stream().wait_event(event)

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
            assert (
                torch.distributed.get_world_size(pg) == torch.distributed.get_world_size()
            ), "Cached AG only supports pure data parallelism"
            local_world_start = torch.distributed.get_rank() // local_world_size * local_world_size
            local_world_end = local_world_start + local_world_size
            target = sum(dispatched_task_list[local_world_start:local_world_end])

        call_watcher(self.reduction_watcher, "wait_and_reset_task_count", target)
        self.dispatched_tasks = 0

        for acc in self.accumulations:
            self.reduce_scatter_sync_cache(acc, pg)

    def reduce_scatter_sync_cache(self, accumulation, pg: dist.ProcessGroup):
        local_world_size = get_local_world_size()
        if local_world_size == torch.distributed.get_world_size():
            return

        same_local_rank_pg = get_same_local_rank_pg(pg)

        registry = SymmBufferRegistry.get_instance()
        local_peer_tensors = registry.get_local_peer_tensors(accumulation)
        local_tensor = accumulation
        assert (
            local_tensor is local_peer_tensors[torch.distributed.get_rank(group=same_local_rank_pg)]
        )
        torch.distributed.reduce_scatter(local_tensor, local_peer_tensors, group=same_local_rank_pg)
        for tensor in local_peer_tensors:
            if tensor is not accumulation:
                tensor.fill_(0)

    def stop(self):
        if self.reduction_watcher is not None:
            call_watcher(self.reduction_watcher, "stop")
            torch.distributed.barrier()
            torch.cuda.synchronize()
            self.lock.lock_buffers.fill_(2)
            nvshmem_free_tensor_sync(self.lock.lock_buffers)
