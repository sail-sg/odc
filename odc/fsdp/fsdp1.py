import torch
import torch.distributed as dist
from torch.distributed.fsdp._flat_param import FlatParamHandle, HandleShardingStrategy
from torch.distributed.fsdp._runtime_utils import (
    _div_if_needed,
    _FSDPState,
    _get_reduce_scatter_tensors,
)

import odc

reduction_service = None
gather_service = None


def _reduce_grad(state, handle) -> None:
    """
    For sharded strategies, this runs gradient reduction, sharded gradient
    accumulation if needed, and the post-reduction callback.
    """

    flat_param = handle.flat_param
    uses_hybrid_sharded_strategy = handle._sharding_strategy in (
        HandleShardingStrategy.HYBRID_SHARD,
        HandleShardingStrategy._HYBRID_SHARD_ZERO2,
    )
    # We clear `.grad` to permit multiple backwards. This avoids a race where
    # the second backward pass computation precedes ahead of the first backward
    # pass reduction, which is possible since the reduction is issued in a
    # separate stream and is async and would result in reducing the wrong
    # gradient.
    unsharded_grad = flat_param.grad.data
    flat_param.grad = None
    # print(f"unsharded_grad shape: {unsharded_grad.shape} dtype {unsharded_grad.dtype}")
    padded_unsharded_grad, new_sharded_grad = _get_reduce_scatter_tensors(state, unsharded_grad)
    # print(state._comm_hook)
    if state._comm_hook is None:  # default path
        _div_if_needed(padded_unsharded_grad, state._gradient_predivide_factor)
        pg = handle._fake_process_group if handle._use_fake_reduce else state.process_group
        # dist.reduce_scatter_tensor(
        #     new_sharded_grad,
        #     padded_unsharded_grad,
        #     group=pg,
        # )
        # print(f"Rank {dist.get_rank()}: reduce_scatter_accumulation")

        rs_func = reduction_service.reduce_scatter_accumulation
        rs_func(id(handle.flat_param), padded_unsharded_grad, pg)
        handle.flat_param._saved_grad_shard = reduction_service.get_accumulation(
            id(handle.flat_param)
        )

        assert not uses_hybrid_sharded_strategy, "ODC does not support hybrid sharded strategy"
        # if uses_hybrid_sharded_strategy:
        #     # Don't wait during trace
        #     if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
        #         state._all_reduce_stream.wait_stream(state._post_backward_stream)
        #     with state._device_handle.stream(state._all_reduce_stream):
        #         # Since the new sharded gradient is produced in the post-
        #         # backward stream and consumed in the all-reduce stream,
        #         # inform the caching allocator
        #         _no_dispatch_record_stream(new_sharded_grad, state._all_reduce_stream)
        #         dist.all_reduce(new_sharded_grad, group=state._inter_node_pg)
        #         _div_if_needed(new_sharded_grad, state._gradient_postdivide_factor)
        #         grad_to_offload = _accumulate_sharded_grad(
        #             state, handle, new_sharded_grad
        #         )
        #         _post_reduce_grad_callback(state, handle, grad_to_offload)
        #         return
        # _div_if_needed(new_sharded_grad, state._gradient_postdivide_factor)
    else:
        state._comm_hook(state._comm_hook_state, padded_unsharded_grad, new_sharded_grad)
        # NOTE: HSDP variants do not support communication hook.

    # Already set below: handle.flat_param._saved_grad_shard = xxx
    # grad_to_offload = _accumulate_sharded_grad(state, handle, new_sharded_grad)
    # Not supported by ODC
    assert not handle._offload_params, "ODC does not support offloading"
    assert not handle._use_orig_params, "ODC does not support using original parameters"
    # _post_reduce_grad_callback(state, handle, grad_to_offload)


def prepare_gradient_for_optim(self):
    """Prepare the gradient for optimizer computation by moving the sharded gradient to the ``.grad`` attribute."""
    from torch.distributed.utils import _p_assert

    def cast_grad_to_param_dtype_if_needed(flat_param):
        # TODO (rohan-varma): test for full precision with keep_low_precision_grads
        if not self._force_full_precision and self._keep_low_precision_grads:
            _p_assert(flat_param.grad is not None, "Unexpected None grad!")
            if flat_param.grad.dtype != self._fwd_bwd_param_dtype:
                flat_param.grad.data = flat_param.grad.to(self._fwd_bwd_param_dtype)
                if self._use_orig_params:
                    self._use_sharded_grad_views()

    flat_param = self.flat_param
    # TODO (awgu): We should replace these conditional checks to encode
    # the logical intention more directly.
    if hasattr(flat_param, "_cpu_grad"):
        # NOTE: This branch includes `NO_SHARD`.
        self._check_sharded(flat_param)
        self._check_on_cpu(flat_param)
        flat_param.grad = flat_param._cpu_grad  # type: ignore[attr-defined]
        cast_grad_to_param_dtype_if_needed(flat_param)
    elif hasattr(flat_param, "_saved_grad_shard"):
        self._check_sharded(flat_param)
        self._check_on_compute_device(flat_param)
        if flat_param._saved_grad_shard is not None:
            self._check_on_compute_device(flat_param._saved_grad_shard)  # type: ignore[attr-defined]
        # If no sharded gradient was computed this iteration, then there is
        # no need to forward `_saved_grad_shard` to `grad`
        if flat_param._post_backward_called:  # type: ignore[attr-defined]
            # print(f"Rank {dist.get_rank()}: flat_param shape: {flat_param.shape}")
            flat_param.grad = None  # type: ignore[attr-defined]
            if flat_param.grad is not None:
                cast_grad_to_param_dtype_if_needed(flat_param)
    else:
        _p_assert(
            not self.uses_sharded_strategy or not flat_param._post_backward_called,  # type: ignore[attr-defined]
            "All sharded parameters that received a gradient in the "
            "post-backward should use `_saved_grad_shard`",
        )
    # Delete `_saved_grad_shard` since its existence indicates a previous
    # gradient to accumulate with in the post-backward hook
    if hasattr(flat_param, "_saved_grad_shard"):
        delattr(flat_param, "_saved_grad_shard")


def all_gather_flat_param(self, padded_unsharded_flat_param):
    """
    All-gather the handle's flat parameter to the destination ``padded_unsharded_flat_param``.

    Then switch to use the all-gathered tensor.
    """
    from torch.distributed.fsdp._common_utils import _no_dispatch_record_stream
    from torch.distributed.utils import _p_assert

    _p_assert(
        hasattr(self, "process_group") and hasattr(self, "world_size"),
        "Expects a process group and world size to have been set via `shard()`",
    )
    sharded_flat_param = self.flat_param.data
    # print(f"sharded_flat_param.dtype: {sharded_flat_param.dtype}")
    expected_numel = sharded_flat_param.numel() * self.world_size
    _p_assert(
        padded_unsharded_flat_param.numel() == expected_numel,
        f"Expects {expected_numel} numel but got {padded_unsharded_flat_param.numel()}",
    )

    pg = self._fake_process_group if self._use_fake_all_gather else self.process_group

    # HACK this should be handled by C10D
    if sharded_flat_param.is_cpu:  # type: ignore[attr-defined]
        tensor_list = list(
            torch.chunk(
                padded_unsharded_flat_param,
                dist.get_world_size(pg),  # type: ignore[arg-type]
            )
        )
        dist.all_gather(tensor_list, sharded_flat_param, group=pg)
    else:
        ag_func = gather_service.all_gather_into_tensor
        ag_func(
            padded_unsharded_flat_param,
            sharded_flat_param,
            pg,
        )

    if self._offload_params:
        # In case of offloading, `flat_param.data` (i.e. sharded param) is
        # created on the pre-unshard stream. We need to hand it over to the
        # unshard stream for all-gather
        _no_dispatch_record_stream(
            sharded_flat_param,
            self._device_handle.current_stream(),  # unshard_stream
        )
    return padded_unsharded_flat_param


old_get_shard = FlatParamHandle._get_shard


def custom_get_shard(tensor, rank, world_size):
    from torch.distributed.fsdp._flat_param import FlatParameter

    sharded, padded = old_get_shard(tensor, rank, world_size)
    assert isinstance(tensor, FlatParameter)
    sharded_in_nvshmem = odc.SymmBufferRegistry.get_instance().update_symm_buffer(
        id(tensor), sharded, rank
    )
    return sharded_in_nvshmem, padded


def patch_fsdp1(reduce_dtype=None):
    global reduction_service, gather_service
    reduction_service = odc.ReductionService(accumulation_dtype=reduce_dtype)
    gather_service = odc.GatherService()
    from torch.distributed.fsdp import _runtime_utils

    _runtime_utils._reduce_grad = _reduce_grad

    FlatParamHandle.prepare_gradient_for_optim = prepare_gradient_for_optim
    FlatParamHandle._get_shard = custom_get_shard
    FlatParamHandle._all_gather_flat_param = all_gather_flat_param


def pre_optimizer_step(fsdp_module):

    assert isinstance(fsdp_module, _FSDPState)
    reduction_service.sync(fsdp_module.process_group)

    # time.sleep(1)
    for acc in reduction_service.accumulations:
        if hasattr(fsdp_module, "_inter_node_pg"):
            dist.all_reduce(acc, group=fsdp_module._inter_node_pg)
        _div_if_needed(acc, fsdp_module._gradient_postdivide_factor)
    # print(f"Model parameters: {[p.numel()/ 1e6 for p in fsdp_module.parameters()]}")
    for handle in fsdp_module._all_handles:
        # print(f"Rank {dist.get_rank()}: cast_grad_to_param_dtype_if_needed shape: {handle.flat_param.shape}")
        handle.flat_param.grad = reduction_service.get_accumulation(id(handle.flat_param)).to(
            handle.flat_param.dtype
        )


def pre_minibatch_start(fsdp_module):
    reduction_service.clear_accumulations()
    # for handle in fsdp_module._all_handles:
    #     odc.all_gather_sync_cache(handle.flat_param, fsdp_module.process_group)

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    for fsdp_mod in FSDP.fsdp_modules(fsdp_module):
        gather_service.all_gather_sync_cache(fsdp_mod._flat_param, fsdp_mod.process_group)

    # Make sure optimizer updates are visible to all ranks
    dist.barrier()


def stop():
    reduction_service.stop()
    odc.SymmBufferRegistry.get_instance().finalize()
