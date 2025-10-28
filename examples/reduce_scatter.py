import logging
import os
import sys

import torch
import torch.distributed as dist
from loguru import logger

from odc.primitives.scatter_accumulate import ReductionService
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


def main():
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
        # reduction_service = ReductionIntraNodeService(accumulation_dtype=accum_dtype)
        cnt = 1
        times = 10
        size = 128 * (1024**2) + 1024
        # cnt = 1
        # times = 2
        # size = 16
        # comp_sizes = torch.rand(cnt).tolist()
        comp_sizes = [2]

        group_ranks = list(range(0, world_size, 1))
        group = torch.distributed.new_group(ranks=group_ranks, backend="nccl")
        assert rank in group_ranks
        group_size = len(group_ranks)
        print(f"Rank {rank} group: {group_ranks}")

        data = [torch.rand(size, dtype=grad_dtype, device="cuda") for _ in range(cnt * times)]
        # data = torch.arange(cnt * times * size, dtype=grad_dtype, device="cuda").reshape(cnt * times, size) / group_size / times
        # print(f"Rank {rank} data: {data}")
        # data = torch.ones(cnt * times, size, dtype=grad_dtype, device="cuda") * rank

        # for i in range(cnt):
        #   reduction_service.register(i, (size // group_size,), grad_dtype, accum_dtype)

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
        # compute_param = torch.empty(8192, 8192, dtype=torch.bfloat16, device="cuda")

        def some_compute(x):
            return x

        nccl_accumulations = [
            torch.zeros(size // group_size, dtype=accum_dtype, device="cuda") for _ in range(cnt)
        ]

        def reduce_scatter_accumulation_nccl(src_tensor, dest_idx, pg: dist.ProcessGroup):
            output = torch.empty(
                (src_tensor.numel() // dist.get_world_size(pg),),
                dtype=src_tensor.dtype,
                device="cuda",
            )
            torch.distributed.reduce_scatter_tensor(
                output, src_tensor, op=torch.distributed.ReduceOp.SUM, group=pg
            )
            nccl_accumulations[dest_idx].add_(output)

        def scatter_accumulation(src_tensor, dest_idx, pg: dist.ProcessGroup):
            reduction_service.scatter_accumulate(dest_idx, src_tensor, pg)

        dist.barrier()
        torch.cuda.synchronize()
        comp_stream = torch.cuda.Stream()

        sync_inputs = torch.zeros(world_size, dtype=torch.int32, device="cuda")

        reduction_service.clear_accumulations()
        with torch.cuda.nvtx.range("scatter_accumulation"):
            for i in range(cnt * times):
                dst_idx = i % cnt
                torch.distributed.all_reduce(sync_inputs, group=group)
                scatter_accumulation(data[i], dst_idx, group)
            reduction_service.sync(group)
            dist.barrier()
            torch.cuda.synchronize()

        for reduce_scatter_func in [
            reduce_scatter_accumulation_nccl,
            scatter_accumulation,
        ]:
            reduction_service.clear_accumulations()

            with torch.cuda.nvtx.range(reduce_scatter_func.__name__):
                start_events = [torch.cuda.Event(enable_timing=True) for _ in range(cnt * times)]
                comm_events = [torch.cuda.Event(enable_timing=True) for _ in range(cnt * times)]
                compute_events = [torch.cuda.Event(enable_timing=True) for _ in range(cnt * times)]
                start = torch.cuda.Event(enable_timing=True)

                for i in range(cnt * times):
                    dst_idx = i % cnt
                    torch.distributed.all_reduce(sync_inputs, group=group)

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

                if reduce_scatter_func == scatter_accumulation:
                    reduction_service.sync(group)
                    for i in range(cnt):
                        # print(f"Rank {rank} nccl_accumulations: {nccl_accumulations[i]} reduction_service: {reduction_service.accumulations[i]}")
                        torch.testing.assert_close(
                            nccl_accumulations[i],
                            reduction_service.accumulations[i],
                            rtol=5e-3,
                            atol=5e-3,
                        )
                    # print(f"Rank {rank} reduction_service: {reduction_service.buffers[0][0]}")
                else:
                    pass
                    # print(f"Rank {rank} nccl_accumulations: {nccl_accumulations[0]}")
                dist.barrier()
                torch.cuda.synchronize()
                # print(f"Rank {rank} comm time: {[start_events[i].elapsed_time(comm_events[i]) for i in range(cnt * times)]}, compute time: {[comm_events[i].elapsed_time(compute_events[i]) for i in range(cnt * times)]}")
                reduce_scatter_payload = (
                    size // group_size * (group_size - 1) * data[0].dtype.itemsize
                )
                print(
                    f"Rank {rank} {reduce_scatter_func.__name__} bw: {reduce_scatter_payload / 1024 ** 2 * (cnt * (times - 1)) / start.elapsed_time(end)}"
                )
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


if __name__ == "__main__":
    main()
