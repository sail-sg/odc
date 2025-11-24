import dataclasses
import logging
import operator
from functools import reduce
from itertools import chain
from typing import Any, Callable, Optional, Union, cast

import torch
import torch.distributed as dist
from torch import nn
from torch._logging import warning_once
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_api import AllGather, ReduceScatter
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    AllGatherResult,
    DefaultAllocMixin,
    _div_if_needed,
    _get_all_gather_input_metadatas,
    _get_gradient_divide_factors,
    foreach_reduce_scatter_copy_in,
)
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    FSDPMeshInfo,
    HSDPMeshInfo,
    TrainingState,
    _get_dim0_padded_size,
    _raise_assert_with_print,
    _to_dtype_if_needed,
    compiled_autograd_enabled,
)
from torch.distributed.fsdp._fully_shard._fsdp_param import (
    FSDPParam,
    ShardedState,
    set_requires_grad_if_needed,
)
from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
    AllReduceState,
    FSDPParamGroup,
    ReduceScatterState,
)
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.profiler import record_function

from odc.primitives.gather import GatherService
from odc.primitives.scatter_accumulate import ReductionService
from odc.primitives.utils import SymmBufferRegistry

logger = logging.getLogger(__name__)


class nvtx_record_function(record_function):
    def __enter__(self):
        torch.cuda.nvtx.range_push(self.name)
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        torch.cuda.nvtx.range_pop()


reduction_service = None
gather_service = None


def get_reduction_service():
    return reduction_service


def get_gather_service():
    return gather_service


class ODCAllGather(DefaultAllocMixin, AllGather):
    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False,
    ) -> Optional[dist.Work]:
        gather = get_gather_service()
        gather.gather_into_tensor(output_tensor, input_tensor, group)
        if async_op:
            event = torch.cuda.Event()
            event.record()
            return event
        return None


def get_fsdp_params_key(fsdp_params: list[FSDPParam]) -> str:
    ids = "_".join([str(id(param)) for param in fsdp_params])
    return f"fsdp_params_gather_{ids}"


def get_hsdp_params_key(fsdp_param_key: str) -> str:
    return f"hsdp_{fsdp_param_key}"


@torch.no_grad()
def replace_sharded_param_with_symm_buffer(
    fsdp_model,
) -> torch.Tensor:
    state = fsdp_model._get_fsdp_state()
    fsdp_param_group = state._fsdp_param_group
    if fsdp_param_group is None:
        return

    dtype = fsdp_param_group.fsdp_params[0]._sharded_param_data.dtype
    for fsdp_param in fsdp_param_group.fsdp_params:
        if fsdp_param._sharded_param_data.dtype != dtype:
            raise ValueError(
                f"All FSDP parameters must have the same dtype: {fsdp_param._sharded_param_data.dtype=} {dtype=}"
            )

    total_size = sum(
        fsdp_param._sharded_param_data.numel() for fsdp_param in fsdp_param_group.fsdp_params
    )
    # This key needs to be the same as the one used in `foreach_all_gather`
    key = get_fsdp_params_key(fsdp_param_group.fsdp_params)
    symm_buffer = SymmBufferRegistry.get_instance().get_or_create_symm_buffer(
        key, (total_size,), dtype
    )

    # Pre-allocate the symmetric buffer for HSDP.
    hsdp_sharded_param_total_size = 0
    num_fsdp_groups = 1
    hsdp_symm_buffer = None
    post_forward_mesh_info = fsdp_param_group.post_forward_mesh_info
    if fsdp_param_group._is_hsdp:
        shard_world_size = post_forward_mesh_info.shard_mesh_size
        world_size = fsdp_param_group._all_gather_process_group.size()
        assert world_size % shard_world_size == 0, f"{world_size=} {shard_world_size=}"
        num_fsdp_groups = world_size // shard_world_size
        hsdp_sharded_param_total_size = total_size * num_fsdp_groups
        hsdp_key = get_hsdp_params_key(key)
        hsdp_symm_buffer = SymmBufferRegistry.get_instance().get_or_create_symm_buffer(
            hsdp_key, (hsdp_sharded_param_total_size,), dtype
        )

    offset = 0
    hsdp_offset = 0
    # Refer to the codes in `FSDPParam._init_sharded_param`
    for fsdp_param in fsdp_param_group.fsdp_params:
        param_size = fsdp_param._sharded_param_data.numel()
        numel = reduce(operator.mul, fsdp_param.padded_sharded_param_size, 1)
        assert numel == param_size, f"{fsdp_param.padded_sharded_param_size=} {param_size=}"
        symm_buffer[offset : offset + param_size].copy_(fsdp_param._sharded_param_data)
        padded_sharded_param = symm_buffer[offset : offset + param_size]
        padded_sharded_param = padded_sharded_param.view(fsdp_param.padded_sharded_param_size)
        offset += param_size

        fsdp_param._sharded_param_data = padded_sharded_param.view(-1)
        shard_dim = fsdp_param.fsdp_placement.dim
        old_sharded_param = fsdp_param.sharded_param._local_tensor
        length = old_sharded_param.size(shard_dim) if old_sharded_param.numel() > 0 else 0
        sharded_param = padded_sharded_param.narrow(dim=shard_dim, start=0, length=length)
        fsdp_param.sharded_param._local_tensor = sharded_param

        if fsdp_param_group._is_hsdp:
            hsdp_param_size = param_size * num_fsdp_groups
            fsdp_param._sharded_post_forward_param_data = hsdp_symm_buffer[
                hsdp_offset : hsdp_offset + hsdp_param_size
            ]
            hsdp_offset += hsdp_param_size


@torch.no_grad()
def _get_param_all_gather_inputs(
    fsdp_params: list[FSDPParam],
) -> list[list[torch.Tensor]]:
    if compiled_autograd_enabled():
        return [fsdp_param.all_gather_inputs for fsdp_param in fsdp_params]

    # Intentionally try to run a fast-path that bypasses abstractions for the
    # common FSDP case of bf16/fp32 mixed precision in order to use foreach
    # copy for lower CPU overhead and more efficient copying in eager
    def use_foreach_copy(fsdp_param: FSDPParam) -> bool:
        return (
            fsdp_param.param_dtype is not None
            and not fsdp_param.offload_to_cpu
            and not hasattr(fsdp_param._sharded_local_tensor, "fsdp_pre_all_gather")
        )

    param_all_gather_inputs: list[list[torch.Tensor]] = [[] for _ in fsdp_params]
    foreach_copy_indices: list[int] = []
    # foreach_copy_inputs: list[torch.Tensor] = []
    # foreach_copy_input_numels: list[int] = []

    # 1st pass: for foreach-copy parameters, get inputs and metadata for the
    # foreach copy, and for the others, actually get their all-gather inputs
    for i, fsdp_param in enumerate(fsdp_params):
        assert not hasattr(
            fsdp_param._sharded_local_tensor, "fsdp_pre_all_gather"
        ), "fsdp_pre_all_gather not supported in ODC"
        if use_foreach_copy(fsdp_param):
            foreach_copy_indices.append(i)
            all_gather_input = (
                fsdp_param._sharded_param_data
                if fsdp_param.sharded_state == ShardedState.SHARDED
                else cast(torch.Tensor, fsdp_param._sharded_post_forward_param_data)
            )
            # foreach_copy_inputs.append(all_gather_input)
            # foreach_copy_input_numels.append(all_gather_input.numel())
            param_all_gather_inputs[i] = [all_gather_input]
        else:
            if (
                hasattr(fsdp_param, "_sharded_post_forward_param_data")
                and fsdp_param.param_dtype is not None
            ):
                # avoid converting symmetric buffer to a new tensor with param dtype for HSDP
                assert (
                    fsdp_param._sharded_post_forward_param_data.dtype == fsdp_param.param_dtype
                ), f"{fsdp_param._sharded_post_forward_param_data.dtype=} != {fsdp_param.param_dtype=}"
            assert fsdp_param.sharded_state in (
                ShardedState.SHARDED,
                ShardedState.SHARDED_POST_FORWARD,
            ), f"Unexpected sharded state: {fsdp_param.sharded_state=}"
            param_all_gather_inputs[i] = fsdp_param.all_gather_inputs

    # We have to use the original symmetric buffer for odc gather.
    # Just return the slices of the original symmetric buffer.
    # The original codes below are not needed for ODC.

    # # 2nd pass: use foreach copy to compute the remaining all-gather inputs
    # if foreach_copy_inputs:
    #     fsdp_param_0 = fsdp_params[foreach_copy_indices[0]]
    #     param_dtype, device = fsdp_param_0.param_dtype, fsdp_param_0.device
    #     flat_foreach_copy_input = torch.empty(
    #         (sum(foreach_copy_input_numels),), device=device, dtype=param_dtype
    #     )
    #     splits = torch.split(flat_foreach_copy_input, foreach_copy_input_numels)
    #     torch._foreach_copy_(splits, foreach_copy_inputs)
    #     for i, split in zip(foreach_copy_indices, splits):
    #         param_all_gather_inputs[i] = [split]

    return param_all_gather_inputs


@torch.no_grad()
def foreach_all_gather(
    fsdp_params: list[FSDPParam],
    group: dist.ProcessGroup,
    async_op: bool,
    all_gather_copy_in_stream: torch.Stream,
    all_gather_stream: torch.Stream,
    device: torch.device,
    all_gather_comm: AllGather,
) -> Optional[AllGatherResult]:
    world_size, _rank = group.size(), group.rank()
    device_handle = _get_device_handle(device.type)
    # Override the all-gather comm with ODCAllGather
    all_gather_comm = ODCAllGather()
    with device_handle.stream(all_gather_copy_in_stream):
        key = get_fsdp_params_key(fsdp_params)
        # if isinstance(fsdp_params[0].post_forward_mesh_info, HSDPMeshInfo):
        print(f"foreach_all_gather: fsdp_params[0].sharded_state={fsdp_params[0].sharded_state}")
        if fsdp_params[0].sharded_state == ShardedState.SHARDED_POST_FORWARD:
            key = get_hsdp_params_key(key)
        assert SymmBufferRegistry.get_instance().has_key(
            key
        ), f"{key=} not found. The fsdp param group has not been replaced with symm buffer yet."
        all_gather_input = SymmBufferRegistry.get_instance().get_symm_buffer(key)
        all_gather_output = all_gather_comm.allocate(
            (all_gather_input.numel() * world_size,), dtype=all_gather_input.dtype, device=device
        )
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
        # all_gather_input_numel = sum(inp_split_sizes)
        # all_gather_output = all_gather_comm.allocate(
        #     (all_gather_input_numel * world_size,), dtype=dtype, device=device
        # )
        # all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in(
        #     all_gather_inputs,
        #     all_gather_output,
        #     inp_split_sizes,
        #     all_gather_input_numel,
        #     rank,
        # )
        # del param_all_gather_inputs
    all_gather_stream.wait_stream(all_gather_copy_in_stream)
    with device_handle.stream(all_gather_stream):
        print(f"gather group: {group.size()} {group}")
        all_gather_work = all_gather_comm(
            output_tensor=all_gather_output,
            input_tensor=all_gather_input,
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


def pre_minibatch_start():
    get_reduction_service().clear_accumulations()

    # Make sure optimizer updates are visible to all ranks
    dist.barrier()


def is_bw() -> bool:
    return torch._C._current_graph_task_id() != -1


# Old version does not skip resharding after the recomputation in backward,
# resulting in duplicated all-gather in backward.
def post_forward(self, _module: nn.Module, _input: Any, output: Any):
    if not compiled_autograd_enabled():
        logger.debug("%s", self._with_fqn("FSDP::post_forward"))
    with record_function(self._with_fqn("FSDP::post_forward")):
        if not compiled_autograd_enabled():
            # for AC(fully_shard(model)), AC runs fsdp's _pre_forward
            # it shouldn't change post_forward_order
            # print(f"post_forward {is_bw()=} {self._training_state=} {self._reshard_after_forward=}")
            if not is_bw():
                print(
                    f"post_forward: _training_state={self._training_state}, "
                    f"_reshard_after_forward={self._reshard_after_forward}, "
                    f"_use_post_forward_mesh={self._use_post_forward_mesh}, "
                    f"mesh_info={type(self.mesh_info).__name__}, "
                    f"post_forward_mesh_info={type(self.post_forward_mesh_info).__name__ if self.post_forward_mesh_info else None}"
                )
                self.reshard()
                self._record_post_forward()
        else:
            self.reshard()
            self._record_post_forward()
        self._training_state = TrainingState.IDLE
        return output


def post_backward(self, *_unused: Any):
    # This method should be idempotent and safe to call even when this
    # FSDP parameter group was not used in backward (should be a no-op)
    if not compiled_autograd_enabled():
        logger.debug("%s", self._with_fqn("FSDP::post_backward"))
    self._training_state = TrainingState.POST_BACKWARD
    with record_function(self._with_fqn("FSDP::post_backward_accumulate")):
        for fsdp_param in self.fsdp_params:
            fsdp_param.accumulate_unsharded_grad_if_needed()
    with record_function(self._with_fqn("FSDP::post_backward_reshard")):
        if not self.reduce_grads:
            if self.reshard_after_backward:
                self.reshard()
            for fsdp_param in self.fsdp_params:
                fsdp_param.to_accumulated_grad_if_needed()
            return
        # Save the autograd-computed gradients before resharding to only
        # access the unsharded parameters when their data is present
        fsdp_params_with_grad: list[FSDPParam] = []
        unsharded_grads: list[torch.Tensor] = []
        for fsdp_param in self.fsdp_params:
            if not hasattr(fsdp_param, "_unsharded_param"):
                continue
            # May have an accumulated gradient of the reduce dtype if the
            # previous backward did not reduce-scatter
            if fsdp_param.unsharded_accumulated_grad is not None:
                fsdp_params_with_grad.append(fsdp_param)
                unsharded_grads.append(fsdp_param.unsharded_accumulated_grad_data)
                fsdp_param.unsharded_accumulated_grad = None
            elif fsdp_param.unsharded_param.grad is not None:
                fsdp_params_with_grad.append(fsdp_param)
                unsharded_grads.append(fsdp_param.unsharded_grad_data)
                fsdp_param.unsharded_param.grad = None
        if self.reshard_after_backward:
            self.reshard()
    if len(fsdp_params_with_grad) == 0:
        return
    with record_function(self._with_fqn("FSDP::post_backward_reduce")):
        if (
            self.comm_ctx.reduce_scatter_state is not None
            and self.comm_ctx.reduce_scatter_state.event is not None
        ):
            self.device_handle.current_stream().wait_event(self.comm_ctx.reduce_scatter_state.event)
        self.comm_ctx.reduce_scatter_state = None
        all_reduce_pg = self._all_reduce_process_group if self._is_hsdp else None
        all_reduce_stream: torch.cuda.Stream
        if all_reduce_pg is None and self._all_reduce_hook_stream is not None:
            # this means the native HSDP is not enabled,
            # but user may want to have a custom HSDP setup
            assert (
                self._all_reduce_hook is not None
            ), "all reduce hook stream is specified but hook itself is missing."
            all_reduce_stream = self._all_reduce_hook_stream
        else:
            all_reduce_stream = self.comm_ctx.all_reduce_stream

        self._wait_for_post_backward()
        (
            reduce_scatter_input,
            reduce_scatter_event,
            self._post_reduce_event,
            all_reduce_input,
            all_reduce_event,
            self._partial_reduce_output,
        ) = foreach_reduce(
            self,
            fsdp_params_with_grad,
            unsharded_grads,
            self._reduce_scatter_process_group,
            self.comm_ctx.reduce_scatter_stream,
            self._reduce_scatter_comm,
            self._orig_dtype,
            self._reduce_dtype,
            self.device,
            self.gradient_divide_factor,
            self._all_reduce_process_group if self._is_hsdp else None,
            all_reduce_stream,
            self.all_reduce_grads,
            self._partial_reduce_output,
            self._all_reduce_hook,
            self.force_sum_reduction_for_comms,
        )
        self.comm_ctx.reduce_scatter_state = ReduceScatterState(
            reduce_scatter_input, reduce_scatter_event
        )
        if all_reduce_input is not None:
            if self.device.type != "cpu":
                assert all_reduce_event is not None
            self._all_reduce_state = AllReduceState(all_reduce_input, all_reduce_event)


@dataclasses.dataclass
class ReduceScatterContext:
    # arguments
    fsdp_params: list[FSDPParam]
    unsharded_grads: list[torch.Tensor]
    reduce_scatter_group: dist.ProcessGroup
    reduce_scatter_stream: torch.Stream
    orig_dtype: Optional[torch.dtype]
    reduce_dtype: Optional[torch.dtype]
    device: torch.device
    gradient_divide_factor: Optional[float]
    all_reduce_group: Optional[dist.ProcessGroup]  # not `None` iff HSDP
    all_reduce_stream: torch.Stream
    all_reduce_grads: bool
    partial_reduce_output: Optional[torch.Tensor]  # only used for HSDP
    all_reduce_hook: Optional[Callable[[torch.Tensor], None]]
    force_sum_reduction_for_comms: bool
    # others
    padded_unsharded_sizes: tuple
    grad_dtype: torch.dtype


@torch.no_grad()
def foreach_reduce(
    fsdp_param_group: FSDPParamGroup,
    fsdp_params: list[FSDPParam],
    unsharded_grads: list[torch.Tensor],
    reduce_scatter_group: dist.ProcessGroup,
    reduce_scatter_stream: torch.Stream,
    reduce_scatter_comm: ReduceScatter,
    orig_dtype: Optional[torch.dtype],
    reduce_dtype: Optional[torch.dtype],
    device: torch.device,
    gradient_divide_factor: Optional[float],
    all_reduce_group: Optional[dist.ProcessGroup],  # not `None` iff HSDP
    all_reduce_stream: torch.Stream,
    all_reduce_grads: bool,
    partial_reduce_output: Optional[torch.Tensor],  # only used for HSDP
    all_reduce_hook: Optional[Callable[[torch.Tensor], None]],
    force_sum_reduction_for_comms: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Event,
    torch.Event,
    Optional[torch.Tensor],
    Optional[torch.Event],
    Optional[torch.Tensor],
]:
    """
    ``unsharded_grads`` owns the references to the gradients computed by
    autograd, so clearing the list frees the gradients.
    """

    grad_dtypes = {grad.dtype for grad in unsharded_grads}
    if len(grad_dtypes) != 1:
        # Check this at runtime since it could be a real runtime error if e.g.
        # fp8 weights do not produce the correct higher precision gradients
        _raise_assert_with_print(
            f"FSDP reduce-scatter expects uniform gradient dtype but got {grad_dtypes}"
        )
    grad_dtype = unsharded_grads[0].dtype
    reduce_dtype = reduce_dtype or grad_dtype
    (predivide_factor, _postdivide_factor, _reduce_scatter_op, _all_reduce_op) = (
        _get_gradient_divide_factors(
            reduce_scatter_group,
            all_reduce_group,
            reduce_dtype,
            device.type,
            gradient_divide_factor,
            force_sum_reduction_for_comms,
        )
    )
    world_size = reduce_scatter_group.size()
    device_handle = _get_device_handle(device.type)
    current_stream = device_handle.current_stream()

    if world_size > 1:
        for i, (fsdp_param, unsharded_grad) in enumerate(zip(fsdp_params, unsharded_grads)):
            if (shard_dim := fsdp_param.fsdp_placement.dim) == 0:
                continue
            assert (
                unsharded_grad.size(shard_dim) % world_size == 0
            ), f"Shard({shard_dim}) requires even sharding: {unsharded_grad.size()=} {world_size=}"
            chunks = torch.chunk(unsharded_grad, world_size, dim=shard_dim)
            unsharded_grads[i] = torch.cat(chunks, dim=0)

    padded_unsharded_sizes = tuple(
        _get_dim0_padded_size(grad.size(), world_size) for grad in unsharded_grads
    )
    reduce_scatter_input_numel = sum(s.numel() for s in padded_unsharded_sizes)
    # reduce_scatter_output_numel = reduce_scatter_input_numel // world_size
    reduce_scatter_input = reduce_scatter_comm.allocate(
        (reduce_scatter_input_numel,),
        dtype=reduce_dtype,
        device=device,
    )

    foreach_reduce_scatter_copy_in(unsharded_grads, reduce_scatter_input, world_size)

    # Only after the copy-in finishes can we free the gradients
    unsharded_grads.clear()
    reduce_scatter_stream.wait_stream(current_stream)
    all_reduce_input = None
    all_reduce_event = None

    with device_handle.stream(reduce_scatter_stream):
        # with torch.cuda.nvtx.range("reduce_scatter_output_allocate"):
        #     reduce_output = reduce_scatter_comm.allocate(
        #         (reduce_scatter_output_numel,),
        #         dtype=reduce_dtype,
        #         device=device,
        #     )
        _div_if_needed(reduce_scatter_input, predivide_factor)
        # if world_size > 1:
        #     reduce_scatter_comm(
        #         output_tensor=reduce_output,
        #         input_tensor=reduce_scatter_input,
        #         group=reduce_scatter_group,
        #         op=reduce_scatter_op,
        #     )
        # else:
        #     # For single GPU, just copy the input to output (no actual reduce-scatter needed)
        #     reduce_output.copy_(reduce_scatter_input)
        key = id(fsdp_param_group)
        scatter = get_reduction_service()
        print(
            f"scatter_accumulate: key={key}, reduce_scatter_input.sum()={reduce_scatter_input.sum().item():.6f}, reduce_scatter_input.shape={reduce_scatter_input.shape} {reduce_scatter_group.size()=}"
        )
        scatter.scatter_accumulate(key, reduce_scatter_input, reduce_scatter_group)

        post_reduce_stream = reduce_scatter_stream
        reduce_scatter_event = reduce_scatter_stream.record_event()

        # Save the context for the next update_gradients call
        fsdp_param_group.__odc_reduce_scatter_context = ReduceScatterContext(
            fsdp_params=fsdp_params,
            unsharded_grads=unsharded_grads,
            reduce_scatter_group=reduce_scatter_group,
            reduce_scatter_stream=reduce_scatter_stream,
            orig_dtype=orig_dtype,
            reduce_dtype=reduce_dtype,
            device=device,
            gradient_divide_factor=gradient_divide_factor,
            all_reduce_group=all_reduce_group,
            all_reduce_stream=all_reduce_stream,
            all_reduce_grads=all_reduce_grads,
            partial_reduce_output=partial_reduce_output,
            all_reduce_hook=all_reduce_hook,
            force_sum_reduction_for_comms=force_sum_reduction_for_comms,
            padded_unsharded_sizes=padded_unsharded_sizes,
            grad_dtype=grad_dtype,
        )

        return (
            reduce_scatter_input,
            reduce_scatter_event,
            post_reduce_stream.record_event(),
            all_reduce_input,
            all_reduce_event,
            partial_reduce_output,
        )


# @torch.no_grad()
def update_gradients(fsdp_param_group: FSDPParamGroup):
    reduce_scatter_context = fsdp_param_group.__odc_reduce_scatter_context
    del fsdp_param_group.__odc_reduce_scatter_context
    fsdp_params = reduce_scatter_context.fsdp_params
    reduce_scatter_group = reduce_scatter_context.reduce_scatter_group
    reduce_scatter_stream = reduce_scatter_context.reduce_scatter_stream
    orig_dtype = reduce_scatter_context.orig_dtype
    reduce_dtype = reduce_scatter_context.reduce_dtype
    device = reduce_scatter_context.device
    gradient_divide_factor = reduce_scatter_context.gradient_divide_factor
    all_reduce_group = reduce_scatter_context.all_reduce_group
    all_reduce_stream = reduce_scatter_context.all_reduce_stream
    # all_reduce_grads = reduce_scatter_context.all_reduce_grads
    # partial_reduce_output = reduce_scatter_context.partial_reduce_output
    all_reduce_hook = reduce_scatter_context.all_reduce_hook
    force_sum_reduction_for_comms = reduce_scatter_context.force_sum_reduction_for_comms
    padded_unsharded_sizes = reduce_scatter_context.padded_unsharded_sizes
    grad_dtype = reduce_scatter_context.grad_dtype

    device_handle = _get_device_handle(device.type)

    reduce_dtype = reduce_dtype or grad_dtype
    (_predivide_factor, postdivide_factor, reduce_scatter_op, all_reduce_op) = (
        _get_gradient_divide_factors(
            reduce_scatter_group,
            all_reduce_group,
            reduce_dtype,
            device.type,
            gradient_divide_factor,
            force_sum_reduction_for_comms,
        )
    )
    world_size = reduce_scatter_group.size()
    device_handle = _get_device_handle(device.type)
    current_stream = device_handle.current_stream()

    print(f"reduce-scatter {reduce_scatter_op=} all-reduce {all_reduce_op=}")

    with device_handle.stream(reduce_scatter_stream):
        post_reduce_stream = all_reduce_stream

        scatter = get_reduction_service()
        key = id(fsdp_param_group)
        reduce_output = scatter.get_accumulation(key)
        print(
            f"get_accumulation: key={key}, reduce_output.sum()={reduce_output.sum().item():.6f}, reduce_output.shape={reduce_output.shape}"
        )
        assert reduce_scatter_op in [
            torch.distributed.ReduceOp.SUM,
            torch.distributed.ReduceOp.AVG,
        ], f"reduce_scatter_op {reduce_scatter_op} is not supported"
        print(
            f"Before division: reduce_output.sum()={reduce_output.sum().item():.6f}, world_size={world_size}"
        )
        if reduce_scatter_op == torch.distributed.ReduceOp.AVG:
            reduce_output /= world_size
            print(f"After division: reduce_output.sum()={reduce_output.sum().item():.6f}")

        if all_reduce_group is not None:  # HSDP
            # ODC: all_reduce_grads is set to False during gradient accumulation in HSDP
            # to defer all-reduce until the last microbatch.
            # But this has been implemented in scatter-accumulate
            # so we can remove this part from the original implementation.
            #
            # Accumulations must run in the reduce-scatter stream
            # if not all_reduce_grads:
            #     if partial_reduce_output is not None:
            #         # partial_reduce_output += reduce_output
            #         pass
            #     else:
            #         partial_reduce_output = reduce_output
            #     return (
            #         reduce_scatter_input,
            #         reduce_scatter_event,
            #         post_reduce_stream.record_event(),
            #         all_reduce_input,
            #         all_reduce_event,
            #         partial_reduce_output,
            #     )
            # if partial_reduce_output is not None:
            #     reduce_output += partial_reduce_output
            post_reduce_stream = all_reduce_stream
            if world_size >= 1:
                all_reduce_stream.wait_stream(reduce_scatter_stream)
            else:
                all_reduce_stream.wait_stream(current_stream)
            print(f"{all_reduce_group.size()=}")
            with device_handle.stream(all_reduce_stream):
                print(
                    f"Before all-reduce: reduce_output.sum()={reduce_output.sum().item():.6f}, all_reduce_group.size()={all_reduce_group.size()}"
                )
                dist.all_reduce(
                    reduce_output,
                    group=all_reduce_group,
                    op=all_reduce_op,
                )
                print(f"After all-reduce: reduce_output.sum()={reduce_output.sum().item():.6f}")
                # all_reduce_input = reduce_output
                # all_reduce_event = all_reduce_stream.record_event()
    # -- END: ops in reduce_scatter stream

    if all_reduce_hook is not None:
        # Execute user-specified all reduce hook.
        # If native HSDP is used, this is executed after the HSDP all reduce.
        # If 1-d FSDP is used, this is executed post reduce-scatter.
        post_reduce_stream = all_reduce_stream
        all_reduce_stream.wait_stream(reduce_scatter_stream)
        with device_handle.stream(all_reduce_stream):
            all_reduce_hook(reduce_output)
    # -- END: ops post reduce_scatter

    with device_handle.stream(post_reduce_stream):
        print(f"{postdivide_factor=}")
        _div_if_needed(reduce_output, postdivide_factor)
        reduce_output = _to_dtype_if_needed(reduce_output, orig_dtype)
        # View out and accumulate sharded gradients
        flat_grad_offset = 0  # [0, reduce_scatter_output_numel - 1]
        for padded_unsharded_size, fsdp_param in zip(padded_unsharded_sizes, fsdp_params):
            # Assume even sharding for Shard(i), i > 0; otherwise would require
            # copy-out for contiguous strides
            new_sharded_grad = torch.as_strided(
                reduce_output,
                size=fsdp_param.sharded_size,
                stride=fsdp_param.contiguous_sharded_stride,
                storage_offset=flat_grad_offset,
            )
            to_accumulate_grad = fsdp_param.sharded_param.grad is not None
            if fsdp_param.offload_to_cpu:
                # Only overlap the D2H copy (copying to pinned memory) if not
                # accumulating gradients since the CPU add kernel depends on
                # the copy result and we cannot run the add as a callback
                non_blocking = fsdp_param.pin_memory and not to_accumulate_grad
                # Since the GPU sharded gradient is allocated in the RS stream,
                # we can free it here by not keeping a ref without waiting for
                # the D2H copy since future RS-stream ops run after the copy
                new_sharded_grad = new_sharded_grad.to(
                    torch.device("cpu"), non_blocking=non_blocking
                )
                if non_blocking:
                    # Record an event on which to block the CPU thread to
                    # ensure that the D2H copy finishes before the optimizer
                    fsdp_param.grad_offload_event = post_reduce_stream.record_event()
            if to_accumulate_grad:
                assert isinstance(fsdp_param.sharded_param.grad, DTensor)
                fsdp_param.sharded_param.grad._local_tensor += new_sharded_grad
            else:
                new_sharded_dtensor_grad = fsdp_param.to_sharded_dtensor(new_sharded_grad)
                fsdp_param.sharded_param.grad = new_sharded_dtensor_grad
            if not compiled_autograd_enabled():
                for hook in (
                    getattr(fsdp_param.sharded_param, "_post_accumulate_grad_hooks", {}) or {}
                ).values():
                    hook(fsdp_param.sharded_param)
            padded_sharded_numel = padded_unsharded_size.numel() // world_size
            flat_grad_offset += padded_sharded_numel
        post_reduce_event = post_reduce_stream.record_event()
    # The RS output is allocated in the RS stream and used in the default
    # stream (for optimizer). To ensure its memory is not reused for later
    # RSs, we do not need extra synchronization since the sharded parameters
    # hold refs through the end of backward.
    # return (
    #     reduce_scatter_input,
    #     reduce_scatter_event,
    #     post_reduce_event,
    #     all_reduce_input,
    #     all_reduce_event,
    #     None,
    # )

    # Synchronize the default stream with post_reduce_stream to ensure
    # gradient writes are visible before optimizer step
    current_stream = device_handle.current_stream()
    current_stream.wait_event(post_reduce_event)


# def hsdp_sharded_param_key()


# FSDPParam
def to_sharded_post_forward(self) -> None:
    if self.is_dtensor:
        raise NotImplementedError("Resharding to smaller mesh with TP is not supported yet")
    self._assert_in_states(ShardedState.UNSHARDED)
    assert self.post_forward_mesh_info is not None  # mypy
    assert len(self.all_gather_outputs) == 1
    shard_world_size = self.post_forward_mesh_info.shard_mesh_size
    if (numel := self.all_gather_outputs[0].numel()) % shard_world_size != 0:
        _raise_assert_with_print(
            f"All-gather output size ({numel}) must be divisible by the shard "
            f"world size ({shard_world_size})"
        )
    shard_rank = self.post_forward_mesh_info.shard_mesh_rank
    # pyrefly: ignore  # unbound-name
    sharded_numel = numel // shard_world_size
    # self._sharded_post_forward_param_data = (
    #     self.all_gather_outputs[0].narrow(
    #         0, sharded_numel * shard_rank, sharded_numel
    #     )
    # ).clone()  # clone to be able to free all-gather output
    self._sharded_post_forward_param_data.copy_(
        self.all_gather_outputs[0].narrow(0, sharded_numel * shard_rank, sharded_numel)
    )
    sharded_post_forward_tensor = torch.as_strided(
        self._sharded_post_forward_param_data,
        size=self.sharded_post_forward_size,
        stride=self.contiguous_sharded_post_forward_stride,
        storage_offset=0,
    )
    self._sharded_post_forward_param = nn.Parameter(
        self.to_sharded_post_forward_dtensor(sharded_post_forward_tensor)
    )
    self._setattr_on_modules(self._sharded_post_forward_param)
    self.free_unsharded_param()
    self.sharded_state = ShardedState.SHARDED_POST_FORWARD


# FSDPParam
def to_unsharded(self) -> None:
    # Assume that the data has been allocated and all-gathered
    set_requires_grad_if_needed(self.sharded_param, self._unsharded_param)
    self._setattr_on_modules(self._unsharded_param)
    if self.sharded_state == ShardedState.SHARDED_POST_FORWARD:
        # The data is allocated in the default stream via the post-forward
        # reshard and must be kept alive for the next all-gather copy-in.
        # Since we call this method after the copy-out, the data's lifetime
        # is ensured without further synchronization.
        self._sharded_post_forward_param = None
        # Do not free the symmetric buffer for HSDP.
        # self._sharded_post_forward_param_data = None  # free
    self.sharded_state = ShardedState.UNSHARDED


def _get_post_forward_mesh_info(
    reshard_after_forward: Union[bool, int], mesh_info: FSDPMeshInfo
) -> Optional[FSDPMeshInfo]:
    shard_mesh_size = mesh_info.shard_mesh_size
    is_hsdp = isinstance(mesh_info, HSDPMeshInfo)

    if not isinstance(reshard_after_forward, (bool, int)):
        raise ValueError(
            "reshard_after_forward should be a bool or an int representing the "
            f"group size to reshard to, not {reshard_after_forward}"
        )

    # NOTE: `isinstance(False, int)` returns `True`.
    if not isinstance(reshard_after_forward, bool) and isinstance(reshard_after_forward, int):
        if (
            reshard_after_forward < 1
            or reshard_after_forward > shard_mesh_size
            or shard_mesh_size % reshard_after_forward != 0
        ):
            raise ValueError(
                "If passing reshard_after_forward as an int, it should be a "
                f"factor of {shard_mesh_size}, not {reshard_after_forward}"
            )
        if reshard_after_forward == 1:
            msg = (
                "reshard_after_forward=1 (int) means resharding parameters to world size 1, "
                "instead of reshard_after_forward=True (bool)"
            )
            warning_once(logger, msg, stacklevel=2)
            reshard_after_forward = False
        # Seems like a bug in FSDP2: reshard_after_forward == shard_mesh_size is not handled correctly for HSDP
        elif reshard_after_forward == shard_mesh_size and not is_hsdp:
            reshard_after_forward = True

    print(f"{reshard_after_forward=} {is_hsdp=}")

    post_forward_mesh_info = None
    if reshard_after_forward is True:
        post_forward_mesh_info = mesh_info
    elif reshard_after_forward is not False:  # int case
        print(f"enable HSDP {reshard_after_forward=}")
        # For HSDP, we can flatten the two replicate dims into the 0th dim
        post_forward_mesh_tensor = mesh_info.mesh.mesh.view(-1, reshard_after_forward)
        post_forward_mesh = DeviceMesh(mesh_info.mesh.device_type, post_forward_mesh_tensor)
        post_forward_mesh_info = HSDPMeshInfo(
            post_forward_mesh, shard_mesh_dim=1, replicate_mesh_dim=0
        )
    return post_forward_mesh_info


def pre_optimizer_step(fsdp_module):
    scatter = get_reduction_service()

    root_state = fully_shard.state(fsdp_module)
    all_fsdp_states = root_state._state_ctx.all_states
    all_fsdp_param_groups = [
        state._fsdp_param_group for state in all_fsdp_states if state._fsdp_param_group is not None
    ]
    for i, fsdp_param_group in enumerate(all_fsdp_param_groups):
        if i == 0:
            mesh_info = (
                cast(FSDPMeshInfo, fsdp_param_group.post_forward_mesh_info)
                if fsdp_param_group.is_sharded_post_forward
                else fsdp_param_group.mesh_info
            )
            assert isinstance(mesh_info, FSDPMeshInfo)
            if fsdp_param_group._is_hsdp:
                reduce_scatter_group = fsdp_param_group.post_forward_mesh_info.shard_process_group
            else:
                reduce_scatter_group = fsdp_param_group.mesh_info.shard_process_group
            with torch.cuda.nvtx.range("scatter_accumulate_sync"):
                scatter.sync(reduce_scatter_group)

        with torch.cuda.nvtx.range(f"update_gradients:{fsdp_param_group._module_fqn}"):
            update_gradients(fsdp_param_group)


def patch_hsdp_fix():
    from torch.distributed.fsdp._fully_shard import _fsdp_init, _fully_shard

    _fsdp_init._get_post_forward_mesh_info = _get_post_forward_mesh_info
    _fully_shard._get_post_forward_mesh_info = _get_post_forward_mesh_info


def patch_fsdp2() -> None:
    from torch.distributed.fsdp._fully_shard import _fsdp_collectives, _fsdp_param_group

    _fsdp_collectives.foreach_all_gather = foreach_all_gather
    _fsdp_param_group.foreach_all_gather = foreach_all_gather
    FSDPParamGroup.post_backward = post_backward
    FSDPParamGroup.post_forward = post_forward
    FSDPParam.to_sharded_post_forward = to_sharded_post_forward
    FSDPParam.to_unsharded = to_unsharded
    _fsdp_param_group.record_function = nvtx_record_function
    torch.profiler.record_function = nvtx_record_function
    torch.autograd.profiler.record_function = nvtx_record_function

    patch_hsdp_fix()

    global reduction_service
    global gather_service
    reduction_service = ReductionService()
    gather_service = GatherService()


def stop():
    get_reduction_service().stop()
    SymmBufferRegistry.get_instance().finalize()
