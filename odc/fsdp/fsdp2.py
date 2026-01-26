import dataclasses
import gc
import logging
import operator
import types
from functools import reduce
from itertools import chain
from typing import Any, Callable, Optional, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_api import AllGather, ReduceScatter
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    AllGatherResult,
    DefaultAllocMixin,
    _div_if_needed,
    _get_all_gather_input_metadatas,
    _get_gradient_divide_factors,
    _get_param_all_gather_inputs,
    foreach_reduce_scatter_copy_in,
)
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    FSDPMeshInfo,
    TrainingState,
    _get_dim0_padded_size,
    _raise_assert_with_print,
    _to_dtype_if_needed,
    compiled_autograd_enabled,
)
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam, ShardedState
from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
    AllReduceState,
    FSDPParamGroup,
    ReduceScatterState,
)
from torch.distributed.tensor import DTensor
from torch.profiler import record_function

from odc.primitives.gather import GatherService
from odc.primitives.scatter_accumulate import ReductionService
from odc.primitives.utils import SymmBufferRegistry, finalize_distributed, log_total_gpu_memory, reset_peak_memory_stats

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


@torch.no_grad()
def patch_lazy_init(fsdp_model):
    state = fsdp_model._get_fsdp_state()
    group = state._fsdp_param_group
    prev_lazy_init = group.lazy_init

    def patched_lazy_init(_self):
        # Debug: Check dtype BEFORE original lazy_init
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and len(_self.fsdp_params) > 0:
            print(f"[DEBUG ODC=1 Rank {rank}] BEFORE lazy_init: _sharded_param_data.dtype = {_self.fsdp_params[0]._sharded_param_data.dtype}")
        
        prev_lazy_init()
        
        # Debug: Check dtype AFTER original lazy_init, BEFORE replace_sharded_param_with_symm_buffer
        if rank == 0 and len(_self.fsdp_params) > 0:
            print(f"[DEBUG ODC=1 Rank {rank}] AFTER lazy_init, BEFORE replace: _sharded_param_data.dtype = {_self.fsdp_params[0]._sharded_param_data.dtype}")
        
        replace_sharded_param_with_symm_buffer(fsdp_model)

    group.lazy_init = types.MethodType(patched_lazy_init, group)


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
    print(f"### param dtype: {dtype}")

    total_size = sum(
        fsdp_param._sharded_param_data.numel() for fsdp_param in fsdp_param_group.fsdp_params
    )
    # This key needs to be the same as the one used in `foreach_all_gather`
    key = get_fsdp_params_key(fsdp_param_group.fsdp_params)
    symm_buffer = SymmBufferRegistry.get_instance().get_or_create_symm_buffer(
        key, (total_size,), dtype
    )
    print(f"[DEBUG ODC=1] symm_buffer total_size {total_size} dtype: {symm_buffer.dtype} memory {total_size * symm_buffer.element_size() / 1e9:.2f} GB")

    offset = 0
    # Refer to the codes in `FSDPParam._init_sharded_param`
    for fsdp_param in fsdp_param_group.fsdp_params:
        param_size = fsdp_param._sharded_param_data.numel()
        numel = reduce(operator.mul, fsdp_param.padded_sharded_param_size, 1)
        assert numel == param_size, f"{fsdp_param.padded_sharded_param_size=} {param_size=}"
        symm_buffer[offset : offset + param_size].copy_(fsdp_param._sharded_param_data)
        padded_sharded_param = symm_buffer[offset : offset + param_size]
        padded_sharded_param = padded_sharded_param.view(fsdp_param.padded_sharded_param_size)
        offset += param_size

        # Store reference to old tensor to ensure we can track/free it
        old_sharded_param_data = fsdp_param._sharded_param_data
        
        fsdp_param._sharded_param_data = padded_sharded_param.view(-1)
        shard_dim = fsdp_param.fsdp_placement.dim
        old_sharded_param = fsdp_param.sharded_param._local_tensor
        length = old_sharded_param.size(shard_dim) if old_sharded_param.numel() > 0 else 0
        sharded_param = padded_sharded_param.narrow(dim=shard_dim, start=0, length=length)
        fsdp_param.sharded_param._local_tensor = sharded_param
        
        # Explicitly delete references to old tensors to help GC
        del old_sharded_param_data
        del old_sharded_param

    # Force garbage collection and clear CUDA cache to free original param memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Log memory after replacement for debugging
    log_total_gpu_memory(f"After replace_sharded_param ({key[:50]}...)")


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
        # print(f"foreach_all_gather: fsdp_params[0].sharded_state={fsdp_params[0].sharded_state}")
        for fsdp_param in fsdp_params:
            assert fsdp_param.sharded_state != ShardedState.SHARDED_POST_FORWARD
        assert SymmBufferRegistry.get_instance().has_key(
            key
        ), f"{key=} not found. The fsdp param group has not been replaced with symm buffer yet."
        all_gather_input = SymmBufferRegistry.get_instance().get_symm_buffer(key)
        all_gather_output = all_gather_comm.allocate(
            (all_gather_input.numel() * world_size,), dtype=all_gather_input.dtype, device=device
        )
        # _get_param_all_gather_inputs allocates some tensors
        # But we don't use them here. Just get the metadata.
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
        input_size_sum = sum(inp_split_sizes)
        assert (
            input_size_sum == all_gather_input.numel()
        ), f"{input_size_sum=} != {all_gather_input.numel()}"
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
        all_gather_work = all_gather_comm(
            output_tensor=all_gather_output,
            input_tensor=all_gather_input,
            group=group,
            async_op=async_op,
        )
        all_gather_event = all_gather_stream.record_event()
        # torch.cuda.default_stream().wait_event(all_gather_event)
        return AllGatherResult(
            all_gather_output,
            all_gather_event,
            all_gather_work,
            param_all_gather_input_dtypes,
            param_all_gather_input_numels,
            inp_split_sizes,
        )


@torch.no_grad()
def foreach_all_gather_copy_out(
    all_gather_result: AllGatherResult,
    fsdp_params: list[FSDPParam],
    group: dist.ProcessGroup,
) -> None:
    (
        all_gather_output,
        all_gather_event,
        all_gather_work,
        param_all_gather_input_dtypes,
        param_all_gather_input_numels,
        all_gather_input_split_sizes,
    ) = all_gather_result
    _dtype, device = all_gather_output.dtype, all_gather_output.device
    device_handle = _get_device_handle(device.type)

    if all_gather_event is not None:  # sync op
        device_handle.current_stream().wait_event(all_gather_event)
    if isinstance(all_gather_work, dist.distributed_c10d.Work):  # async op
        all_gather_work.wait()
    world_size, device = group.size(), all_gather_output.device

    # In ODC, we use the original dtype for all-gather input and output.
    # Then when the output needs to be used here, we cast it to the param_dtype.
    param_dtype = fsdp_params[0].param_dtype
    if param_dtype is not None:
        all_gather_output = _to_dtype_if_needed(all_gather_output, param_dtype)
        input_dtype = param_all_gather_input_dtypes[0][0]
        for dtypes in param_all_gather_input_dtypes:
            for dtype in dtypes:
                if dtype != input_dtype:
                    raise ValueError(f"Input dtypes are not the same: {dtype} != {input_dtype}")
        param_all_gather_input_dtypes = [
            [param_dtype] * len(dtypes) for dtypes in param_all_gather_input_dtypes
        ]

    split_with_sizes_out: list[torch.Tensor] = []
    shard_i_copy_infos: list[tuple[FSDPParam, list[torch.Tensor]]] = []
    for all_gather_input_numels, all_gather_input_dtypes, fsdp_param in zip(
        param_all_gather_input_numels, param_all_gather_input_dtypes, fsdp_params
    ):
        # NOTE: Under compile, make sure we always recreate all_gather_outputs
        # per AllGather. See [Note: Invariants for torch.compile Traceable FSDP2].
        force_recreate = compiled_autograd_enabled()
        fsdp_param.init_all_gather_outputs(
            all_gather_input_numels,
            all_gather_input_dtypes,
            world_size,
            device,
            force_recreate=force_recreate,
        )
        if not force_recreate:
            fsdp_param.alloc_all_gather_outputs()
        param_all_gather_outputs = fsdp_param.all_gather_outputs
        if fsdp_param.fsdp_placement.dim != 0:
            # Copy to a temporary and then chunk-cat into the final all-gather
            # output tensors
            param_all_gather_outputs = [torch.empty_like(t) for t in param_all_gather_outputs]
            shard_i_copy_infos.append((fsdp_param, param_all_gather_outputs))
        split_with_sizes_out.extend(param_all_gather_outputs)

    all_gather_output = all_gather_output.view(world_size, -1)
    if all_gather_output.dtype == torch.uint8:
        out = [t.view(world_size, -1).view(torch.uint8) for t in split_with_sizes_out]
    else:
        out = [t.view(world_size, -1) for t in split_with_sizes_out]

    # only avoid VC bump if we are not in inference mode
    if torch._dynamo.is_compiling():
        # For torch.compile, we turn off inference_mode for fake tensor
        # propagation, and therefore graph break on is_inference. For `compile`,
        # we don't care about VCs, so just skip the optimization.
        non_inference_outs = []
    else:
        non_inference_outs = [o for o in out if not o.is_inference()]

    if len(non_inference_outs) > 0:
        with torch.autograd._unsafe_preserve_version_counter(tuple(non_inference_outs)):
            torch.ops.fsdp.split_with_sizes_copy(
                all_gather_output, all_gather_input_split_sizes, dim=1, out=out
            )
    else:
        torch.ops.fsdp.split_with_sizes_copy(
            all_gather_output, all_gather_input_split_sizes, dim=1, out=out
        )

    for fsdp_param, param_all_gather_outputs in shard_i_copy_infos:
        # Chunk-cat from the temporary to the final all-gather output tensors
        shard_dim = fsdp_param.fsdp_placement.dim

        with torch.autograd._unsafe_preserve_version_counter(tuple(fsdp_param.all_gather_outputs)):
            for param_all_gather_output, target_all_gather_output in zip(
                param_all_gather_outputs, fsdp_param.all_gather_outputs
            ):
                padded_sharded_size = (
                    fsdp_param.padded_sharded_param_size
                    if fsdp_param.sharded_state == ShardedState.SHARDED
                    else cast(torch.Tensor, fsdp_param._sharded_post_forward_param_data).size()
                )
                pre_param_size = list(padded_sharded_size)
                pre_param_size[0] *= world_size
                chunks = torch.chunk(
                    param_all_gather_output.view(pre_param_size), world_size, dim=0
                )
                post_param_size = list(padded_sharded_size)
                post_param_size[shard_dim] *= world_size
                cat_out = target_all_gather_output.view(post_param_size)
                torch.cat(chunks, dim=shard_dim, out=cat_out)


_minibatch_count = 0

def pre_minibatch_start(fsdp_module):
    global _minibatch_count
    _minibatch_count += 1
    
    # Log memory at start of first minibatch and reset peak stats
    if _minibatch_count == 1:
        log_total_gpu_memory("Before first minibatch")
        reset_peak_memory_stats()  # Reset so we can track peak during training
    
    get_reduction_service().clear_accumulations()

    # Make sure optimizer updates are visible to all ranks
    dist.barrier()

    check_fsdp_module(fsdp_module)


_microbatch_count = 0
_forward_count = 0
_epoch_count = 0

def log_after_forward(microbatch_id: int):
    """Call this after model forward pass (captures peak activation memory)"""
    global _forward_count
    _forward_count += 1
    log_total_gpu_memory(f"After forward {microbatch_id}")


def log_after_microbatch(microbatch_id: int):
    """Call this after each microbatch (after loss.backward())"""
    global _microbatch_count
    _microbatch_count += 1
    log_total_gpu_memory(f"After backward {microbatch_id}")


def log_after_optimizer_step():
    """Call this after optimizer.step()"""
    global _epoch_count
    _epoch_count += 1
    log_total_gpu_memory(f"After optimizer step {_epoch_count}")


def log_after_optimizer_zero_grad():
    """Call this after optimizer.zero_grad()"""
    global _epoch_count
    log_total_gpu_memory(f"After zero_grad {_epoch_count}")


def is_bw() -> bool:
    return torch._C._current_graph_task_id() != -1


# Old version of pytorch does not skip resharding after the recomputation in backward,
# resulting in duplicated all-gather in backward.
# Patch the new pytorch version here.
def post_forward(self, _module: nn.Module, _input: Any, output: Any):
    if not compiled_autograd_enabled():
        logger.debug("%s", self._with_fqn("FSDP::post_forward"))
    with record_function(self._with_fqn("FSDP::post_forward")):
        if not compiled_autograd_enabled():
            # for AC(fully_shard(model)), AC runs fsdp's _pre_forward
            # it shouldn't change post_forward_order
            if not is_bw():
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


def check_fsdp_module(fsdp_module):
    root_state = fully_shard.state(fsdp_module)
    root_state._lazy_init()
    all_fsdp_states = root_state._state_ctx.all_states
    all_fsdp_param_groups = [
        state._fsdp_param_group for state in all_fsdp_states if state._fsdp_param_group is not None
    ]

    hpz = any(fsdp_param_group._use_post_forward_mesh for fsdp_param_group in all_fsdp_param_groups)
    if hpz:
        raise ValueError("reshard_after_forward=int is not supported yet")


@torch.no_grad()
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

    with device_handle.stream(reduce_scatter_stream):
        post_reduce_stream = all_reduce_stream

        scatter = get_reduction_service()
        key = id(fsdp_param_group)
        reduce_output = scatter.get_accumulation(key)
        assert reduce_scatter_op in [
            torch.distributed.ReduceOp.SUM,
            torch.distributed.ReduceOp.AVG,
        ], f"reduce_scatter_op {reduce_scatter_op} is not supported"
        if reduce_scatter_op == torch.distributed.ReduceOp.AVG:
            reduce_output /= world_size

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
            with device_handle.stream(all_reduce_stream):
                dist.all_reduce(
                    reduce_output,
                    group=all_reduce_group,
                    op=all_reduce_op,
                )
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
                print(f"{orig_dtype=} {fsdp_param.sharded_param.grad._local_tensor.dtype=}")
            else:
                new_sharded_dtensor_grad = fsdp_param.to_sharded_dtensor(new_sharded_grad)
                fsdp_param.sharded_param.grad = new_sharded_dtensor_grad
                print(f"{orig_dtype=} {fsdp_param.sharded_param.grad.dtype=} {new_sharded_grad.dtype=}")
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


_logged_memory_breakdown = False

def pre_optimizer_step(fsdp_module):
    global _logged_memory_breakdown
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
            reduce_scatter_group = fsdp_param_group.mesh_info.shard_process_group
            with torch.cuda.nvtx.range("scatter_accumulate_sync"):
                scatter.sync(reduce_scatter_group)

        with torch.cuda.nvtx.range(f"update_gradients:{fsdp_param_group._module_fqn}"):
            update_gradients(fsdp_param_group)

    # Log memory breakdown once after first backward pass
    if not _logged_memory_breakdown and dist.get_rank() == 0:
        _logged_memory_breakdown = True
        SymmBufferRegistry.get_instance().log_memory_breakdown()
        log_total_gpu_memory("After first backward")


def patch_fsdp2() -> None:
    from torch.distributed.fsdp._fully_shard import _fsdp_collectives, _fsdp_param_group

    _fsdp_collectives.foreach_all_gather = foreach_all_gather
    _fsdp_param_group.foreach_all_gather = foreach_all_gather
    _fsdp_collectives.foreach_all_gather_copy_out = foreach_all_gather_copy_out
    _fsdp_param_group.foreach_all_gather_copy_out = foreach_all_gather_copy_out
    FSDPParamGroup.post_backward = post_backward
    FSDPParamGroup.post_forward = post_forward
    _fsdp_param_group.record_function = nvtx_record_function
    torch.profiler.record_function = nvtx_record_function
    torch.autograd.profiler.record_function = nvtx_record_function

    global reduction_service
    global gather_service
    reduction_service = ReductionService()
    gather_service = GatherService()


def stop():
    """Stop ODC services and log final memory summary."""
    reduction_svc = get_reduction_service()
    if reduction_svc is not None:
        # Log final accumulation memory summary
        reduction_svc.stop()
        
        # Also log the overall symm buffer memory breakdown
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            logger.info("[ODC] === Final Memory Summary at Training End ===")
        SymmBufferRegistry.get_instance().log_memory_breakdown()
    
    SymmBufferRegistry.get_instance().finalize()
    finalize_distributed()


# ============== Debugging utilities for dtype checking ==============
_debug_dtype_printed = False


def patch_fsdp2_debug_dtype() -> None:
    """
    Patch FSDP2 to print dtype information during forward pass.
    This can be used with ODC=0 to debug what dtype _sharded_param_data has.
    
    Usage: Call this function before the first forward pass.
    """
    from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
    from torch.distributed.fsdp._fully_shard._fsdp_collectives import foreach_all_gather as orig_foreach_all_gather
    from torch.distributed.fsdp._fully_shard import _fsdp_collectives, _fsdp_param_group
    
    # Store original lazy_init
    orig_lazy_init = FSDPParamGroup.lazy_init
    
    def debug_lazy_init(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and len(self.fsdp_params) > 0:
            print(f"[DEBUG ODC=0 Rank {rank}] BEFORE lazy_init: _sharded_param_data.dtype = {self.fsdp_params[0]._sharded_param_data.dtype}")
        
        result = orig_lazy_init(self)
        
        if rank == 0 and len(self.fsdp_params) > 0:
            print(f"[DEBUG ODC=0 Rank {rank}] AFTER lazy_init: _sharded_param_data.dtype = {self.fsdp_params[0]._sharded_param_data.dtype}")
            print(f"[DEBUG ODC=0 Rank {rank}] AFTER lazy_init: orig_dtype = {self.fsdp_params[0].orig_dtype}")
            print(f"[DEBUG ODC=0 Rank {rank}] AFTER lazy_init: param_dtype = {self.fsdp_params[0].param_dtype}")
        
        return result
    
    FSDPParamGroup.lazy_init = debug_lazy_init
    
    # Store original all_gather_inputs property
    orig_all_gather_inputs = FSDPParam.all_gather_inputs.fget
    
    @property
    def debug_all_gather_inputs(self) -> list[torch.Tensor]:
        global _debug_dtype_printed
        if not _debug_dtype_printed:
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(f"[DEBUG ODC=0 Rank {rank}] all_gather_inputs: _sharded_param_data.dtype = {self._sharded_param_data.dtype}")
            print(f"[DEBUG ODC=0 Rank {rank}] all_gather_inputs: orig_dtype = {self.orig_dtype}")
            print(f"[DEBUG ODC=0 Rank {rank}] all_gather_inputs: param_dtype = {self.param_dtype}")
            print(f"[DEBUG ODC=0 Rank {rank}] all_gather_inputs: reduce_dtype = {self.reduce_dtype}")
            _debug_dtype_printed = True
        
        # Call original implementation
        return orig_all_gather_inputs(self)
    
    # Patch the property
    FSDPParam.all_gather_inputs = debug_all_gather_inputs
    
    # Also patch foreach_all_gather to print input dtypes
    def debug_foreach_all_gather(
        fsdp_params,
        group,
        async_op,
        all_gather_copy_in_stream,
        all_gather_stream,
        device,
        all_gather_comm,
    ):
        global _debug_dtype_printed
        if not _debug_dtype_printed:
            rank = dist.get_rank() if dist.is_initialized() else 0
            for i, fsdp_param in enumerate(fsdp_params):
                print(f"[DEBUG ODC=0 Rank {rank}] foreach_all_gather fsdp_param[{i}]._sharded_param_data.dtype = {fsdp_param._sharded_param_data.dtype}")
                print(f"[DEBUG ODC=0 Rank {rank}] foreach_all_gather fsdp_param[{i}].orig_dtype = {fsdp_param.orig_dtype}")
                print(f"[DEBUG ODC=0 Rank {rank}] foreach_all_gather fsdp_param[{i}].param_dtype = {fsdp_param.param_dtype}")
                if i >= 2:  # Only print first 3 params
                    print(f"[DEBUG ODC=0 Rank {rank}] ... (truncated, total {len(fsdp_params)} params)")
                    break
            _debug_dtype_printed = True
        
        return orig_foreach_all_gather(
            fsdp_params,
            group,
            async_op,
            all_gather_copy_in_stream,
            all_gather_stream,
            device,
            all_gather_comm,
        )
    
    _fsdp_collectives.foreach_all_gather = debug_foreach_all_gather
    _fsdp_param_group.foreach_all_gather = debug_foreach_all_gather
    
    logger.info("FSDP2 debug dtype patch applied. Will print dtype info during lazy_init and first forward pass.")
