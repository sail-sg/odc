import logging
from itertools import chain
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp._fully_shard._fsdp_api import AllGather, ReduceScatter, _ReduceOp
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    AllGatherResult,
    DefaultAllocMixin,
    ProcessGroupAllocMixin,
    _get_all_gather_input_metadatas,
    _get_param_all_gather_inputs,
)
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam

from odc.primitives.gather import GatherService
from odc.primitives.scatter_accumulate import ReductionService
from odc.primitives.utils import SymmBufferRegistry

logger = logging.getLogger(__name__)


class ODCAllGather(DefaultAllocMixin, AllGather):
    _odc_gather_instance = None

    @classmethod
    def get_odc_gather(cls) -> GatherService:
        if cls._odc_gather_instance is None:
            cls._odc_gather_instance = GatherService()
        return cls._odc_gather_instance

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False,
    ) -> Optional[dist.Work]:
        gather = self.get_odc_gather()
        torch.distributed.barrier(group)  # TODO: remove this
        gather.gather_into_tensor(output_tensor, input_tensor, group)
        if async_op:
            event = torch.cuda.Event()
            event.record()
            return event
        return None
        # return dist.all_gather_into_tensor(
        #     output_tensor,
        #     input_tensor,
        #     group=group,
        #     async_op=async_op,
        # )


class ODCReduceScatter(ProcessGroupAllocMixin, ReduceScatter):
    _odc_reduction_instance = None

    @classmethod
    def get_odc_reduction(cls) -> ReductionService:
        if cls._odc_reduction_instance is None:
            cls._odc_reduction_instance = ReductionService()
        return cls._odc_reduction_instance

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op: _ReduceOp,
        async_op: bool = False,
    ) -> dist.Work:
        # reduction = self.get_odc_reduction()
        # torch.distributed.barrier(group)  # TODO: remove this
        # reduction.scatter_accumulate(output_tensor, input_tensor, group)
        # reduction.sync(group)
        # if async_op:
        #     event = torch.cuda.Event()
        #     event.record()
        #     return event
        # return None
        return dist.reduce_scatter_tensor(
            output=output_tensor,
            input=input_tensor,
            group=group,
            op=op,
            async_op=async_op,
        )


class FSDPParamsGatherBuffers:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = FSDPParamsGatherBuffers()
        return cls._instance

    def __init__(self):
        self.buffers = {}

    def get_buffer(
        self, fsdp_params: list[FSDPParam], input_tensor: torch.Tensor, rank: int
    ) -> torch.Tensor:
        key = self._get_fsdp_params_key(fsdp_params)
        if key not in self.buffers:
            registry = SymmBufferRegistry.get_instance()
            self.buffers[key] = registry.allocate_symm_buffer(
                key, input_tensor.shape, input_tensor.dtype, rank
            )
        else:
            assert self.buffers[key].shape == input_tensor.shape
            assert self.buffers[key].dtype == input_tensor.dtype
        return self.buffers[key]

    def _get_fsdp_params_key(self, fsdp_params: list[FSDPParam]) -> str:
        return "_".join([str(id(param)) for param in fsdp_params])


@torch.no_grad()
def custom_foreach_all_gather(
    fsdp_params: list[FSDPParam],
    group: dist.ProcessGroup,
    async_op: bool,
    all_gather_copy_in_stream: torch.Stream,
    all_gather_stream: torch.Stream,
    device: torch.device,
    all_gather_comm: AllGather,
) -> Optional[AllGatherResult]:
    world_size, rank = group.size(), group.rank()
    device_handle = _get_device_handle(device.type)
    with device_handle.stream(all_gather_copy_in_stream):
        param_all_gather_inputs = _get_param_all_gather_inputs(fsdp_params)
        (
            param_all_gather_input_dtypes,
            param_all_gather_input_numels,
            dtype,
        ) = _get_all_gather_input_metadatas(param_all_gather_inputs)
        if dtype == torch.uint8:
            all_gather_inputs = [t.view(torch.uint8) for ts in param_all_gather_inputs for t in ts]
        else:
            all_gather_inputs = [*chain.from_iterable(param_all_gather_inputs)]
        inp_split_sizes = [t.numel() for t in all_gather_inputs]
        all_gather_input_numel = sum(inp_split_sizes)
        all_gather_output = all_gather_comm.allocate(
            (all_gather_input_numel * world_size,), dtype=dtype, device=device
        )
        all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in(
            all_gather_inputs,
            all_gather_output,
            inp_split_sizes,
            all_gather_input_numel,
            rank,
        )
        del param_all_gather_inputs
        symm_buffer = FSDPParamsGatherBuffers.get_instance().get_buffer(
            fsdp_params, all_gather_input, rank
        )
        symm_buffer.copy_(all_gather_input, non_blocking=True)
    all_gather_stream.wait_stream(all_gather_copy_in_stream)
    with device_handle.stream(all_gather_stream):
        all_gather_work = all_gather_comm(
            output_tensor=all_gather_output,
            # input_tensor=all_gather_input,
            input_tensor=symm_buffer,
            group=group,
            async_op=async_op,
        )
        all_gather_event = all_gather_stream.record_event()
        return AllGatherResult(
            all_gather_output,
            all_gather_event,
            all_gather_work,
            param_all_gather_input_dtypes,
            param_all_gather_input_numels,
            inp_split_sizes,
        )


def set_custom_all_gather(fsdp_model: FSDPModule, comm: AllGather) -> None:
    """
    Overrides the default ``all_gather`` communication behavior,
    to have better control over the communication and memory usage.
    See `Comm` and `ReduceScatter` for details.

    Args:
        comm (AllGather): Custom all-gather communication.
    """
    state = fsdp_model._get_fsdp_state()
    if (fsdp_param_group := state._fsdp_param_group) is not None:
        fsdp_param_group._all_gather_comm = comm


def set_custom_reduce_scatter(fsdp_model: FSDPModule, comm: ReduceScatter) -> None:
    """
    Overrides the default ``reduce_scatter`` communication behavior,
    to have better control over the communication and memory usage.
    See `Comm` and `ReduceScatter` for details.

    Args:
        comm (ReduceScatter): Custom reduce_scatter communication.
    """
    state = fsdp_model._get_fsdp_state()
    if (fsdp_param_group := state._fsdp_param_group) is not None:
        fsdp_param_group._reduce_scatter_comm = comm


def patch_fsdp2(fsdp_model: FSDPModule) -> None:
    from torch.distributed.fsdp._fully_shard import _fsdp_collectives, _fsdp_param_group

    _fsdp_collectives.foreach_all_gather = custom_foreach_all_gather
    _fsdp_param_group.foreach_all_gather = custom_foreach_all_gather

    set_custom_all_gather(fsdp_model, ODCAllGather())


# def pre_optimizer_step(fsdp_module):

#     assert isinstance(fsdp_module, _FSDPState)
#     get_reduction_service().sync(fsdp_module.process_group)

#     # time.sleep(1)
#     for acc in get_reduction_service().accumulations:
#         if hasattr(fsdp_module, "_inter_node_pg"):
#             dist.all_reduce(acc, group=fsdp_module._inter_node_pg)
#         _div_if_needed(acc, fsdp_module._gradient_postdivide_factor)
#     # print(f"Model parameters: {[p.numel()/ 1e6 for p in fsdp_module.parameters()]}")
#     for handle in fsdp_module._all_handles:
#         # print(f"Rank {dist.get_rank()}: cast_grad_to_param_dtype_if_needed shape: {handle.flat_param.shape}")
#         handle.flat_param.grad = (
#             get_reduction_service()
#             .get_accumulation(id(handle.flat_param))
#             .to(handle.flat_param.dtype)
#         )


def stop():
    # get_reduction_service().stop()
    SymmBufferRegistry.get_instance().finalize()
