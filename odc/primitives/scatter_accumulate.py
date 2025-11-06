import logging
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from threading import Thread
from typing import List, Mapping, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing
import triton
import triton.language as tl

from odc.primitives import (
    NVSHMEM_EXTERN_LIBS,
    __syncthreads,
    get_ipc_handle,
    int_g,
    int_p,
    putmem_nbi_block,
    quiet,
    reconstruct_tensor,
    tid,
)
from odc.primitives.utils import (
    PROCESS_GROUP_RANKS_TENSORS,
    BufferSplitter,
    SymmBufferRegistry,
    get_comm_stream,
    get_local_world_size,
    sync_cta,
)

logger = logging.getLogger(__name__)


MAX_REQUEST_COUNT = 2 * 100000


@triton.jit(do_not_specialize=[])
def nvshmem_scatter_kernel(
    input_tensor_ptr,
    rank_input_size,
    input_segment_start,
    chunk_buffer,
    output_tensor_ptr,
    elem_per_rank,
    size_per_elem,
    group_rank,
    num_ranks_per_node,
    group_world_size: tl.constexpr,
    pg_ranks_ptr,
    chunk_size: tl.constexpr,
    signal_next_expected,
    signal_ptr,
):
    pid = tl.program_id(axis=0)
    tidx = tid(axis=0)
    # np = tl.num_programs(axis=0)
    assert num_ranks_per_node == tl.num_programs(axis=0)
    np = num_ranks_per_node
    assert group_world_size % np == 0
    num_nodes = group_world_size // np
    expected = signal_next_expected
    chunk_buffer_seg = chunk_buffer + pid * chunk_size

    # Use different kernel for the ranks in the same node.
    for i in range(1, num_nodes):
        peer_node = (i + group_rank // np) % num_nodes
        peer = (pid + peer_node * np) % group_world_size

        num_chunks = tl.cdiv(elem_per_rank, chunk_size)
        for chunk in range(num_chunks):
            this_chunk_size = chunk_size
            if chunk == num_chunks - 1:
                this_chunk_size = elem_per_rank - chunk * chunk_size
            chunk_offsets = tl.arange(0, chunk_size)
            input_start = peer * rank_input_size + input_segment_start + (chunk * chunk_size)
            mask = chunk_offsets < this_chunk_size
            input_chunk_data = tl.load(input_tensor_ptr + input_start + chunk_offsets, mask=mask)
            tl.store(chunk_buffer_seg + chunk_offsets, input_chunk_data, mask=mask)
            # As we initialize NVSHMEM on global process group,
            # we need to use the global rank to access the peer tensor.
            global_peer = tl.load(pg_ranks_ptr + peer)
            putmem_nbi_block(
                output_tensor_ptr + (chunk * chunk_size),
                chunk_buffer_seg,
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

    expected += np
    sync_cta(signal_ptr, expected)

    if pid == 0:
        if tidx == 0:
            quiet()
        __syncthreads()

    return expected


@triton.jit(do_not_specialize=["rank", "peer", "next_request_id"])
def nvshmem_cross_node_scatter(
    input_tensor_ptr,
    rank_input_size,
    chunk_buffer,
    trans_buffer,
    size_per_elem,
    group_rank,
    num_ranks_per_node,
    group_world_size: tl.constexpr,
    pg_ranks_ptr,
    output_size,
    local_buf_size,
    chunk_size: tl.constexpr,
    signal_ptr,
    # client request
    request_buffer_ptr,
    response_buffer_ptr,
    rank_start_same_node,
    rank_end_same_node,
    accumulation_command,
    next_request_id,
):
    signal_next_expected = 0
    for start in range(0, output_size, local_buf_size):
        size = min(local_buf_size, output_size - start)
        signal_next_expected = nvshmem_scatter_kernel(
            input_tensor_ptr=input_tensor_ptr,
            rank_input_size=rank_input_size,
            input_segment_start=start,
            chunk_buffer=chunk_buffer,
            output_tensor_ptr=trans_buffer,
            elem_per_rank=size,
            size_per_elem=size_per_elem,
            group_rank=group_rank,
            num_ranks_per_node=num_ranks_per_node,
            group_world_size=group_world_size,
            pg_ranks_ptr=pg_ranks_ptr,
            chunk_size=chunk_size,
            signal_next_expected=signal_next_expected,
            signal_ptr=signal_ptr,
        )

        nvshmem_request_accumulation_remote_node_kernel(
            request_buffer_ptr=request_buffer_ptr,
            group_rank=group_rank,
            rank_start_same_node=rank_start_same_node,
            rank_end_same_node=rank_end_same_node,
            group_world_size=group_world_size,
            accumulation_command=accumulation_command,
        )
        nvshmem_wait_accumulation_remote_node_kernel(
            response_buffer_ptr=response_buffer_ptr,
            group_rank=group_rank,
            rank_start_same_node=rank_start_same_node,
            rank_end_same_node=rank_end_same_node,
            group_world_size=group_world_size,
            next_request_id=next_request_id,
        )
        next_request_id += 1

        # All CTAs need to wait for the first CTA to finish the accumulation request.
        if start + local_buf_size < output_size:
            signal_next_expected += num_ranks_per_node
            sync_cta(signal_ptr, signal_next_expected)


@triton.jit(do_not_specialize=["rank", "peer", "accumulation_command"])
def nvshmem_request_accumulation_same_node_kernel(
    request_buffer_ptr,
    rank,
    peer,
    accumulation_command,
):
    pid = tl.program_id(axis=0)
    tidx = tid(axis=0)
    if pid == 0:
        if tidx == 0:
            int_p(request_buffer_ptr + rank, accumulation_command, peer)
        __syncthreads()


@triton.jit(do_not_specialize=["rank", "peer", "next_request_id"])
def nvshmem_wait_accumulation_same_node_kernel(
    response_buffer_ptr,
    rank,
    peer,
    next_request_id,
):
    pid = tl.program_id(axis=0)
    tidx = tid(axis=0)
    if pid == 0:
        if tidx == 0:
            r = next_request_id - 1
            while r != next_request_id:
                quiet()
                r = int_g(response_buffer_ptr + rank, peer)
        __syncthreads()


@triton.jit(
    do_not_specialize=[
        "group_rank",
        "rank_start_same_node",
        "rank_end_same_node",
        "group_world_size",
        "accumulation_command",
    ]
)
def nvshmem_request_accumulation_remote_node_kernel(
    request_buffer_ptr,
    group_rank,
    rank_start_same_node,
    rank_end_same_node,
    group_world_size,
    accumulation_command,
):
    pid = tl.program_id(axis=0)
    tidx = tid(axis=0)
    if pid == 0:
        if tidx == 0:
            for peer in range(group_world_size):
                if peer < rank_start_same_node or peer >= rank_end_same_node:
                    int_p(request_buffer_ptr + group_rank, accumulation_command, peer)
        __syncthreads()


@triton.jit(
    do_not_specialize=[
        "group_rank",
        "rank_start_same_node",
        "rank_end_same_node",
        "group_world_size",
        "next_request_id",
    ]
)
def nvshmem_wait_accumulation_remote_node_kernel(
    response_buffer_ptr,
    group_rank,
    rank_start_same_node,
    rank_end_same_node,
    group_world_size,
    next_request_id,
):
    pid = tl.program_id(axis=0)
    tidx = tid(axis=0)
    if pid == 0:
        if tidx == 0:
            for peer in range(group_world_size):
                if peer < rank_start_same_node or peer >= rank_end_same_node:
                    r = next_request_id - 1
                    while r != next_request_id:
                        quiet()
                        r = int_g(response_buffer_ptr + group_rank, peer)
        __syncthreads()


@dataclass
class ClientContext:
    request_buffer: torch.Tensor
    response_buffer: torch.Tensor
    next_request_id: int
    local_next_request_id: int


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


def server_loop(server_context, dispatch_func, exit_predicate, client_mask=None):
    if client_mask is None:
        client_mask = set()
    request_buffer_cpu = torch.empty_like(server_context.request_buffer, device="cpu").pin_memory()
    while True:
        request_buffer_cpu.copy_(server_context.request_buffer)
        nonzeros = torch.nonzero(request_buffer_cpu, as_tuple=False).squeeze(1).tolist()
        time.sleep(1 / 10000)
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


class DistLock:
    def __init__(self, pg: dist.ProcessGroup):
        group_rank = torch.distributed.get_rank(group=pg)
        self.world_size = torch.distributed.get_world_size(pg)
        self.request_buffer = SymmBufferRegistry.get_instance().allocate_symm_buffer(
            "request_buffer", (self.world_size,), torch.int32, group_rank
        )
        self.response_buffer = SymmBufferRegistry.get_instance().allocate_symm_buffer(
            "response_buffer", (self.world_size,), torch.int32, group_rank
        )
        self.request_buffer.fill_(0)
        self.response_buffer.fill_(0)
        self.client_context = ClientContext(self.request_buffer, self.response_buffer, 1, 1)


class ReductionWatcher:
    def __init__(
        self,
        world_size,
        accumulations: List[torch.Tensor],
        buffers: List[torch.Tensor],
        request_buffer: torch.Tensor,
        response_buffer: torch.Tensor,
    ):
        self.accumulations = accumulations
        self.buffers = buffers
        self.request_buffer = request_buffer
        self.response_buffer = response_buffer
        self.world_size = world_size
        self.running = True
        self.task_count = 0
        self.server_context = ServerContext(
            self.request_buffer, self.response_buffer, [1] * self.world_size, defaultdict(lambda: 0)
        )

    def stop(self):
        self.running = False

    def wait_and_reset_task_count(self, expected):
        while self.task_count < expected:
            time.sleep(0)
        self.task_count = 0

    def add_buffer(self, buffers):
        self.buffers.append([tensor_from_handle(*buffer) for buffer in buffers])

    def add_accumulation(self, accumulations):
        self.accumulations.append([tensor_from_handle(*acc) for acc in accumulations])

    def run(self):
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
                with torch.cuda.nvtx.range(
                    f"add client {client_rank} buffer {buffer_id} accumulation {accumulation_id}"
                ):
                    acc[start : start + size].add_(buf[:size])
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

        client_mask = set()
        server_loop(self.server_context, dispatch_func, exit_predicate, client_mask)


def tensor_from_handle(handle, size, dtype):
    return reconstruct_tensor(handle, (size,), dtype)


def reduction_watcher_function(
    device_id,
    world_size,
    accumulations,
    buffers,
    request_buffer,
    response_buffer,
    cmd_queue,
    response_queue,
):
    torch.cuda.set_device(device_id)

    # torch.cuda.cudart().cudaProfilerStart()
    buffers = [tensor_from_handle(*buffer) for buffer in buffers]
    accumulations = [tensor_from_handle(*acc) for acc in accumulations]
    request_buffer = tensor_from_handle(*request_buffer)
    response_buffer = tensor_from_handle(*response_buffer)

    watcher = ReductionWatcher(world_size, accumulations, buffers, request_buffer, response_buffer)

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


def start_reduction_watcher(accumulations, buffers, request_buffer, response_buffer):
    original_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if original_visible_devices is not None:
        del os.environ["CUDA_VISIBLE_DEVICES"]

    ctx = torch.multiprocessing.get_context("spawn")
    cmd_queue = ctx.Queue()
    response_queue = ctx.Queue()
    device_id = torch.distributed.get_rank() % get_local_world_size()
    world_size = torch.distributed.get_world_size()
    process = ctx.Process(
        target=reduction_watcher_function,
        args=(
            device_id,
            world_size,
            accumulations,
            buffers,
            request_buffer,
            response_buffer,
            cmd_queue,
            response_queue,
        ),
    )
    process.start()
    if original_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = original_visible_devices
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]
    return cmd_queue, response_queue


def call_watcher(watcher_handle, cmd, *args):
    cmd_queue, response_queue = watcher_handle
    cmd_queue.put((cmd, *args))
    return response_queue.get()


def get_nvshmem_handle(tensor):
    logger.info(
        f"Rank {torch.distributed.get_rank()} get_nvshmem_handle {tensor.data_ptr()} with shape {tensor.shape} and dtype {tensor.dtype}",
    )
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
        self.buffer_splitter = BufferSplitter()
        self.rank_streams = defaultdict(torch.cuda.Stream)
        self.chunk_size_bytes = 2**20

    def get_chunk_size(self, buffer_dtype):
        return self.chunk_size_bytes // buffer_dtype.itemsize

    def register(
        self, key, output_tensor_shape, grad_dtype, reduction_dtype, pg: dist.ProcessGroup
    ):
        if self.reduction_watcher is None:
            self.lock = DistLock(pg)
            request_buffer_handle = get_nvshmem_handle(self.lock.request_buffer)
            response_buffer_handle = get_nvshmem_handle(self.lock.response_buffer)

            # Make sure changes are visible to all reduction watchers
            torch.distributed.barrier()
            torch.cuda.synchronize()

            self.reduction_watcher = start_reduction_watcher(
                [], [], request_buffer_handle, response_buffer_handle
            )

        accumulation_key = f"rs_accumulation_{key}"
        assert len(output_tensor_shape) == 1
        registry = SymmBufferRegistry.get_instance()
        assert not registry.has_key(accumulation_key)
        # assert self.reduction_watcher is None, "Reduction watcher is already running"

        group_rank = torch.distributed.get_rank(group=pg)

        def create_and_register_accumulation(key, shape, dtype, add_func):
            buffer = registry.allocate_symm_buffer(key, shape, dtype, group_rank)
            call_watcher(self.reduction_watcher, add_func, [get_nvshmem_handle(buffer)])
            return buffer

        def create_and_register_buffer(key, shape, dtype, add_func):
            buffers = []
            for rank in range(torch.distributed.get_world_size()):
                buffer = registry.allocate_symm_buffer(
                    f"{key}_rank_{rank}", shape, dtype, group_rank
                )
                buffers.append(buffer)
            call_watcher(self.reduction_watcher, add_func, [get_nvshmem_handle(b) for b in buffers])
            return buffers

        # acc = registry.allocate_symm_buffer(accumulation_key, output_tensor_shape, reduction_dtype)
        acc = create_and_register_accumulation(
            accumulation_key, output_tensor_shape, reduction_dtype, "add_accumulation"
        )
        self.accumulation_indices[key] = len(self.accumulations)
        self.accumulations.append(acc)

        world_size = torch.distributed.get_world_size()
        buffer_size = self.buffer_splitter.get_local_buffer_size(output_tensor_shape, world_size)
        output_size = reduce(lambda x, y: x * y, output_tensor_shape)
        logger.info(
            f"buffer_size: {buffer_size} output_size: {output_size} num_split: {math.ceil(output_size / buffer_size)}"
        )
        buffer_shape = (buffer_size,)

        shared_buffer_key = (grad_dtype, buffer_shape)
        if shared_buffer_key not in self.shared_buffer:
            output_size = reduce(lambda x, y: x * y, output_tensor_shape)
            logger.info(
                f"Rank {torch.distributed.get_rank()} create buffer: output_size: {output_size} num_sub_buffers: {math.ceil(output_size / buffer_size)} buffer_size: {buffer_size}",
            )
            cnt = len(self.shared_buffer)
            buffers = create_and_register_buffer(
                f"shared_buffer_{cnt}", buffer_shape, grad_dtype, "add_buffer"
            )
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

    def scatter_accumulate(self, key, input_tensor, pg: dist.ProcessGroup):
        output_tensor_shape = self.infer_output_shape(input_tensor, pg)
        accum_dtype = (
            self.accumulation_dtype if self.accumulation_dtype is not None else input_tensor.dtype
        )
        if key not in self.accumulation_indices:
            self.register(key, output_tensor_shape, input_tensor.dtype, accum_dtype, pg)

        world_size = torch.distributed.get_world_size(pg)
        local_buf_size = self.buffer_splitter.get_local_buffer_size(output_tensor_shape, world_size)
        output_size = reduce(lambda x, y: x * y, output_tensor_shape)

        chunk_size = self.get_chunk_size(input_tensor.dtype)
        grid_size = get_local_world_size()
        input_tensor_symm_shape = (chunk_size * grid_size,)
        rank = torch.distributed.get_rank(pg)
        if (input_tensor_symm_shape, input_tensor.dtype) not in self.input_buffer:
            self.input_buffer[
                (input_tensor_symm_shape, input_tensor.dtype)
            ] = SymmBufferRegistry.get_instance().allocate_symm_buffer(
                f"rs_buffer_{input_tensor_symm_shape}_{input_tensor.dtype}",
                input_tensor_symm_shape,
                input_tensor.dtype,
                rank,
            )
        input_tensor_symm = self.input_buffer[(input_tensor_symm_shape, input_tensor.dtype)]

        buffer_id = self.buffer_indices[key]
        buffer = self.buffers[buffer_id][rank]
        accumulation_id = self.accumulation_indices[key] + 1

        accumulation_command = (buffer_id << 16) | accumulation_id
        assert (
            buffer.nbytes % (2**6) == 0
        ), f"better align to 64 for efficiency. Found {buffer.nbytes} bytes"

        get_comm_stream().wait_stream(torch.cuda.current_stream())
        rank_start_same_node = rank - rank % get_local_world_size()
        rank_end_same_node = rank_start_same_node + get_local_world_size()
        for local_peer in range(rank_start_same_node, rank_end_same_node):
            self.rank_streams[local_peer].wait_stream(torch.cuda.current_stream())

        pg_ranks_tensor = PROCESS_GROUP_RANKS_TENSORS.get_pg_ranks_tensor(pg)

        with torch.cuda.stream(get_comm_stream()):
            signal_ptr = torch.empty(1, dtype=torch.int32, device="cuda")
        num_requests = math.ceil(output_size / local_buf_size)
        assert world_size % 8 == 0 or world_size < 8
        assert world_size % grid_size == 0
        _, rank_input_size = input_tensor.view(-1).view(world_size, -1).shape

        for start in range(0, output_size, local_buf_size):
            size = min(local_buf_size, output_size - start)
            assert local_buf_size == buffer.numel()
            buf = buffer[:size]
            # Use mem-copy for the ranks in the same node.
            same_node_tensors = SymmBufferRegistry.get_instance().get_peer_tensors(buffer)
            for i in range(get_local_world_size()):
                peer_idx = (rank % get_local_world_size() + i) % get_local_world_size()
                local_peer = rank_start_same_node + peer_idx
                rank_input_start = local_peer * rank_input_size + start
                same_node_peer_buffer = same_node_tensors[peer_idx]
                peer_buf = same_node_peer_buffer[:size]
                stream = self.rank_streams[local_peer]
                with torch.cuda.stream(stream):
                    peer_buf.copy_(
                        input_tensor[rank_input_start : rank_input_start + size],
                        non_blocking=True,
                    )

            for local_peer in range(rank_start_same_node, rank_end_same_node):
                with torch.cuda.stream(self.rank_streams[local_peer]):
                    nvshmem_request_accumulation_same_node_kernel[(1,)](
                        request_buffer_ptr=self.lock.request_buffer,
                        rank=rank,
                        peer=local_peer,
                        accumulation_command=accumulation_command,
                        num_warps=1,
                        extern_libs=NVSHMEM_EXTERN_LIBS,
                    )
                    nvshmem_wait_accumulation_same_node_kernel[(1,)](
                        response_buffer_ptr=self.lock.response_buffer,
                        rank=rank,
                        peer=local_peer,
                        next_request_id=self.lock.client_context.local_next_request_id,
                        num_warps=1,
                        extern_libs=NVSHMEM_EXTERN_LIBS,
                    )
            self.lock.client_context.local_next_request_id += 1
            if self.lock.client_context.local_next_request_id > MAX_REQUEST_COUNT:
                self.lock.client_context.local_next_request_id = 1
        with torch.cuda.stream(get_comm_stream()):
            signal_ptr.fill_(0)
            nvshmem_cross_node_scatter[(grid_size,)](
                input_tensor_ptr=input_tensor,
                rank_input_size=rank_input_size,
                chunk_buffer=input_tensor_symm,
                trans_buffer=buf,
                size_per_elem=buf.element_size(),
                group_rank=rank,
                num_ranks_per_node=grid_size,
                group_world_size=world_size,
                pg_ranks_ptr=pg_ranks_tensor,
                output_size=output_size,
                local_buf_size=local_buf_size,
                chunk_size=chunk_size,
                signal_ptr=signal_ptr,
                # client request
                request_buffer_ptr=self.lock.request_buffer,
                response_buffer_ptr=self.lock.response_buffer,
                rank_start_same_node=rank_start_same_node,
                rank_end_same_node=rank_end_same_node,
                accumulation_command=accumulation_command,
                next_request_id=self.lock.client_context.next_request_id,
                num_warps=32,
                extern_libs=NVSHMEM_EXTERN_LIBS,
            )
        self.lock.client_context.next_request_id += num_requests
        if self.lock.client_context.next_request_id > MAX_REQUEST_COUNT:
            self.lock.client_context.next_request_id = 1
        self.dispatched_tasks += num_requests

        for local_peer in range(rank_start_same_node, rank_end_same_node):
            torch.cuda.current_stream().wait_stream(self.rank_streams[local_peer])
        torch.cuda.current_stream().wait_stream(get_comm_stream())
        # nvshmem.core.quiet(stream=torch.cuda.current_stream())

    def get_accumulation(self, key):
        acc = self.accumulations[self.accumulation_indices[key]]
        return acc

    def sync(self, pg: dist.ProcessGroup):
        # TODO: This actually only syncs CPU of reduction workers, it's possible that the last reduction is on-the-fly.
        dispatched_task_list = [None for _ in range(dist.get_world_size(pg))]
        torch.distributed.all_gather_object(dispatched_task_list, self.dispatched_tasks, group=pg)
        torch.cuda.synchronize()

        target = sum(dispatched_task_list)
        call_watcher(self.reduction_watcher, "wait_and_reset_task_count", target)
        self.dispatched_tasks = 0

    def stop(self):
        if self.reduction_watcher is not None:
            call_watcher(self.reduction_watcher, "stop")
            torch.distributed.barrier()
            torch.cuda.synchronize()
