import logging
import os
import sys

import torch
import torch.distributed as dist
from loguru import logger
from torch import Tensor

from odc.primitives.gather import GatherService
from odc.primitives.utils import SymmBufferRegistry, init_nvshmem


class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())


logger.remove()
logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

LOG_ERR_LEVELS = ["ERROR", "CRITICAL"]

logger.add(
    sys.stdout,
    level="INFO",
    filter=lambda record: record["level"].name not in LOG_ERR_LEVELS,
)

logger.add(
    sys.stderr,
    level="ERROR",
    filter=lambda record: record["level"].name in LOG_ERR_LEVELS,
)


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
        comp_sizes = [2]
        dtype = torch.int64

        group_ranks = list(range(0, world_size, 1))
        group = torch.distributed.new_group(ranks=group_ranks, backend="nccl")
        assert rank in group_ranks
        group_size = len(group_ranks)
        print(f"Rank {rank} group: {group_ranks}")

        torch.cuda.synchronize()
        mem_allocated = torch.cuda.memory_allocated() / (1024**2)
        mem_reserved = torch.cuda.memory_reserved() / (1024**2)
        print(
            f"[Rank {rank}] CUDA memory allocated: {mem_allocated:.2f} MB, reserved: {mem_reserved:.2f} MB"
        )
        compute_buffer = [
            torch.empty(int(x * 16384), 8192, dtype=torch.bfloat16, device="cuda")
            for x in comp_sizes
        ]
        compute_param = torch.empty(8192, 8192, dtype=torch.bfloat16, device="cuda")

        def some_compute(x):
            return x

        src_tensors = [torch.empty(size, dtype=dtype, device="cuda") for _ in range(cnt)]
        group_rank = torch.distributed.get_rank(group=group)
        for i in range(cnt):
            src_tensors[i].fill_(i + rank * 2)
            src_tensors[i] = registry.update_symm_buffer(i, src_tensors[i], group_rank)

        gather_service = GatherService()

        comp_stream = torch.cuda.Stream()
        for all_gather_func in [all_gather_into_tensor_nccl, gather_service.gather_into_tensor]:
            with torch.cuda.nvtx.range(all_gather_func.__name__):
                start_events = [torch.cuda.Event(enable_timing=True) for _ in range(cnt)]
                comm_events = [torch.cuda.Event(enable_timing=True) for _ in range(cnt)]
                compute_events = [torch.cuda.Event(enable_timing=True) for _ in range(cnt)]
                start = torch.cuda.Event(enable_timing=True)

                for i in range(cnt):
                    if i == 1:
                        torch.distributed.barrier(group)
                        start.record()
                    dst = torch.empty(size * group_size, dtype=dtype, device="cuda")
                    # dst_arr = [
                    #   dst[r * size:(r + 1) * size]
                    #   for r in range(world_size)
                    # ]
                    start_events[i].record()
                    comp_stream.wait_stream(torch.cuda.current_stream())
                    all_gather_func(dst, src_tensors[i], group)
                    with torch.cuda.stream(comp_stream):
                        some_compute(compute_buffer[0])
                    torch.cuda.current_stream().wait_stream(comp_stream)
                    comm_events[i].record()
                    # compute_buffer[i] @ compute_param
                    compute_events[i].record()

                    # print(dst)
                    for r in range(group_size):
                        expected = group_ranks[r] * 2 + i
                        assert torch.eq(
                            dst[r * size : (r + 1) * size], expected
                        ).all(), f"Rank {rank} cnt {i} r {r} dst: {dst[r * size:(r + 1) * size]}, expected: {expected} group_ranks: {group_ranks}"
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                dist.barrier()
                torch.cuda.synchronize()
                # print(f"Rank {rank} comm time: {[start_events[i].elapsed_time(comm_events[i]) for i in range(cnt)]}, compute time: {[comm_events[i].elapsed_time(compute_events[i]) for i in range(cnt)]}")
                all_gather_payload = (
                    size * (group_size - 1) * dtype.itemsize  # pylint: disable=no-member
                )
                print(
                    f"Rank {rank} {all_gather_func.__name__} bw: {all_gather_payload / 1024 ** 2 * (cnt - 1) / start.elapsed_time(end)}"
                )
                print(f"Total time: {start.elapsed_time(end)}")
                # print(f"Rank {rank} dst: {dst}")

    except Exception as e:
        print(e)
        import traceback

        traceback.print_exc()
    finally:
        registry.finalize()
        torch.distributed.destroy_process_group()
    torch.cuda.cudart().cudaProfilerStop()

# for t in local_tensors:
#   nvshmem.core.free_tensor(t)
