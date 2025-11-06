import logging
from contextlib import contextmanager
from itertools import chain
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp._fully_shard._fsdp_api import AllGather, ReduceScatter, _ReduceOp
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    AllGatherResult,
    DefaultAllocMixin,
    ProcessGroupAllocMixin,
    _div_if_needed,
    _get_all_gather_input_metadatas,
    _get_gradient_divide_factors,
    _get_param_all_gather_inputs,
    foreach_reduce_scatter_copy_in,
)
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    _get_dim0_padded_size,
    _raise_assert_with_print,
    _to_dtype_if_needed,
    compiled_autograd_enabled,
)
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
from torch.distributed.tensor import DTensor

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

    @classmethod
    def get_fsdp_params_key(cls, fsdp_params: list[FSDPParam]) -> str:
        ids = "_".join([str(id(param)) for param in fsdp_params])
        return f"fsdp_params_reduce_{ids}"

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op: _ReduceOp,
        async_op: bool = False,
    ) -> dist.Work:
        raise NotImplementedError("ODCReduceScatter is not implemented")


class FSDPParamsGatherBuffers:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = FSDPParamsGatherBuffers()
        return cls._instance

    def get_buffer(
        self, fsdp_params: list[FSDPParam], input_tensor: torch.Tensor, rank: int
    ) -> torch.Tensor:
        key = self._get_fsdp_params_key(fsdp_params)
        buf = SymmBufferRegistry.get_instance().get_or_create_symm_buffer(
            key, input_tensor.shape, input_tensor.dtype, rank
        )
        assert buf.shape == input_tensor.shape
        assert buf.dtype == input_tensor.dtype
        return buf

    def _get_fsdp_params_key(self, fsdp_params: list[FSDPParam]) -> str:
        ids = "_".join([str(id(param)) for param in fsdp_params])
        return f"fsdp_params_gather_{ids}"


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


_is_last_microbatch = False


@contextmanager
def last_microbatch_context():
    global _is_last_microbatch
    _is_last_microbatch = True
    yield
    _is_last_microbatch = False


def pre_minibatch_start():
    ODCReduceScatter.get_odc_reduction().clear_accumulations()

    # Make sure optimizer updates are visible to all ranks
    dist.barrier()


@torch.no_grad()
def custom_foreach_reduce(
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
    (predivide_factor, postdivide_factor, reduce_scatter_op, all_reduce_op) = (
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
    reduce_scatter_output_numel = reduce_scatter_input_numel // world_size
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
        reduce_output = reduce_scatter_comm.allocate(
            (reduce_scatter_output_numel,),
            dtype=reduce_dtype,
            device=device,
        )
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
        key = ODCReduceScatter.get_fsdp_params_key(fsdp_params)
        scatter = ODCReduceScatter.get_odc_reduction()
        scatter.scatter_accumulate(key, reduce_scatter_input, reduce_scatter_group)

        reduce_scatter_event = reduce_scatter_stream.record_event()
        post_reduce_stream = reduce_scatter_stream
        if not _is_last_microbatch:
            return (
                reduce_scatter_input,
                reduce_scatter_event,
                post_reduce_stream.record_event(),
                all_reduce_input,
                all_reduce_event,
                partial_reduce_output,
            )

        scatter.sync(reduce_scatter_group)
        reduce_output = scatter.get_accumulation(key)
        assert reduce_scatter_op in [
            torch.distributed.ReduceOp.SUM,
            torch.distributed.ReduceOp.AVG,
        ], f"reduce_scatter_op {reduce_scatter_op} is not supported"
        if reduce_scatter_op == torch.distributed.ReduceOp.AVG:
            reduce_output /= world_size

        if all_reduce_group is not None:  # HSDP
            # Accumulations must run in the reduce-scatter stream
            assert (
                all_reduce_grads
            ), "set_is_last_microbatch(True) needs set_requires_all_reduce() for the last microbatch"
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
            with device_handle.stream(all_reduce_stream):
                dist.all_reduce(
                    reduce_output,
                    group=all_reduce_group,
                    op=all_reduce_op,
                )
                all_reduce_input = reduce_output
                all_reduce_event = all_reduce_stream.record_event()
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
    return (
        reduce_scatter_input,
        reduce_scatter_event,
        post_reduce_event,
        all_reduce_input,
        all_reduce_event,
        None,
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
    _fsdp_collectives.foreach_reduce = custom_foreach_reduce
    _fsdp_param_group.foreach_reduce = custom_foreach_reduce

    set_custom_all_gather(fsdp_model, ODCAllGather())


def stop():
    ODCReduceScatter.get_odc_reduction().stop()
    SymmBufferRegistry.get_instance().finalize()
