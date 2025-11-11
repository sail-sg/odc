import logging
from contextlib import contextmanager
from itertools import chain
from typing import Callable, List, Optional, cast

import torch
import torch.distributed as dist
from torch import nn
from torch._prims_common import make_contiguous_strides_for
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
    foreach_reduce_scatter_copy_in,
)
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    HSDPMeshInfo,
    _chunk_with_empty,
    _get_dim0_padded_size,
    _get_dim_chunked_size,
    _raise_assert_with_print,
    _to_dtype_if_needed,
    compiled_autograd_enabled,
)
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam, ShardedState
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor.device_mesh import _mesh_resources
from torch.distributed.tensor.placement_types import Placement, _StridedShard

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

    def gather(
        self,
        output_tensor: torch.Tensor,
        input_tensors: List[torch.Tensor],
        group: dist.ProcessGroup,
        async_op=False,
    ) -> Optional[dist.Work]:
        gather = self.get_odc_gather()
        gather.gather_multi_into_tensor(output_tensor, input_tensors, group)
        if async_op:
            event = torch.cuda.Event()
            event.record()
            return event
        return None

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
def _init_sharded_param(
    self,
    param: nn.Parameter,
    device: torch.device,
    shard_placement_fn: Optional[Callable],
):
    if param.device != device and param.device.type != "meta":
        raise AssertionError(
            f"Expects the parameter to already be moved to device {device} but got {param.device}"
        )
    if not param.is_contiguous():
        raise NotImplementedError(
            f"FSDP does not support non-contiguous parameters yet: {param.shape=} {param.stride()=}"
        )
    fsdp_placement = shard_placement_fn(param) if shard_placement_fn else None
    if fsdp_placement is None:
        fsdp_placement = Shard(0)
    elif fsdp_placement.dim < 0:
        fsdp_placement = Shard(fsdp_placement.dim + param.ndim)
    assert isinstance(fsdp_placement, Shard), f"{fsdp_placement}"
    self.fsdp_placement = fsdp_placement
    shard_dim = fsdp_placement.dim
    # TODO: Replace the sharded DTensor parameter construction logic with
    # `distribute_tensor` after https://github.com/pytorch/pytorch/issues/116101
    # TODO: Simplify the following sharded parameter padding logic after
    # https://github.com/pytorch/pytorch/issues/113045
    self.is_dtensor = isinstance(param, DTensor)
    if self.is_dtensor:
        self._tp_spec = cast(DTensor, param)._spec
        dp_mesh, tp_mesh = (self.mesh_info.mesh, self._tp_spec.mesh)
        dp_global_mesh = _mesh_resources.get_root_mesh(dp_mesh)
        tp_global_mesh = _mesh_resources.get_root_mesh(tp_mesh)
        if dp_global_mesh != tp_global_mesh or (dp_global_mesh is None or tp_global_mesh is None):
            raise AssertionError(
                "FSDP requires the DP and model parallel TP/EP mesh to have the same parent mesh but got: \n"
                f"DP's global mesh: {dp_global_mesh}\nTP/EP's global mesh: {tp_global_mesh}"
            )
        name_dims_error = "FSDP requires named DeviceMesh dims for ND parallelism"
        assert dp_mesh.mesh_dim_names is not None, name_dims_error
        assert tp_mesh.mesh_dim_names is not None, name_dims_error
        submesh_names = dp_mesh.mesh_dim_names + tp_mesh.mesh_dim_names
        self._spmd_mesh = dp_global_mesh[submesh_names]
        if len(self._tp_spec.placements) > 2:
            raise NotImplementedError(
                f"FSDP only supports 1D TP/EP or 2D EP+TP, not {self._tp_spec.placements}"
            )
        split_factor = self._tp_spec.num_shards_map[shard_dim]
        assert 2 <= self._spmd_mesh.ndim <= 4, (
            "_spmd_mesh.ndim can only be 2 (FSDP+TP/EP), 3 (FSDP+EP+TP, HSDP+TP/EP), "
            f"or 4 (HSDP+EP+TP) but got {self._spmd_mesh.ndim}."
        )
        self._spmd_placements: tuple[Placement, ...]
        dp_shard_tp_placement = (
            (
                _StridedShard(shard_dim, split_factor=split_factor)
                if split_factor > 1
                else fsdp_placement
            ),
            *self._tp_spec.placements,
        )
        if dp_mesh.ndim == 1:  # FSDP
            self._spmd_placements = dp_shard_tp_placement
        else:  # HSDP
            assert self.mesh_info.replicate_mesh_dim == 0
            self._spmd_placements = (Replicate(),) + dp_shard_tp_placement
        self._sharding_spec = DTensorSpec(
            self._spmd_mesh,
            self._spmd_placements,
            tensor_meta=self._tp_spec.tensor_meta,
        )
        param_data = cast(DTensor, param)._local_tensor
    else:
        self._spmd_mesh = self.mesh_info.mesh
        if isinstance(self.mesh_info, HSDPMeshInfo):
            self._spmd_placements = (Replicate(), fsdp_placement)
        else:
            self._spmd_placements = (fsdp_placement,)
        self._sharding_spec = DTensorSpec(
            self._spmd_mesh,
            self._spmd_placements,
            tensor_meta=TensorMeta(param.size(), param.stride(), param.dtype),
        )
        param_data = param
    assert param_data.is_contiguous(), f"{param_data.shape=} {param_data.stride()=}"
    shard_dim = fsdp_placement.dim
    if shard_dim >= param_data.ndim:
        raise AssertionError(
            f"Shard dim {shard_dim} is invalid for {param_data.ndim}D tensor: {param.shape}"
        )
    self._orig_size = param_data.size()
    self._contiguous_orig_stride = make_contiguous_strides_for(self._orig_size)
    shard_rank = self.mesh_info.shard_mesh_rank
    shard_world_size = self.mesh_info.shard_mesh_size
    if shard_dim > 0 and param_data.size(shard_dim) % shard_world_size != 0:
        # If sharding on nonzero dim, require even sharding for now because
        # the uneven sharding (1) requires extra copies before/after FSDP
        # collectives and (2) introduces extra complexity to handle padding
        # and unpadding
        raise NotImplementedError(
            f"FSDP does not support uneven sharding on dim {shard_dim}: "
            f"{param_data.size()} (world size: {shard_world_size})"
        )
    chunks = _chunk_with_empty(param_data, shard_world_size, dim=shard_dim)
    sharded_param = chunks[shard_rank]
    self.sharded_size = _get_dim_chunked_size(sharded_param, param_data.size(), dim=shard_dim)
    self.contiguous_sharded_stride = make_contiguous_strides_for(self.sharded_size)
    padded_sharded_size = chunks[0].size()  # 0th always padded
    self.padded_sharded_param_size = padded_sharded_size
    # Pre-pad the sharded parameter to avoid padding before all-gather
    # padded_sharded_param = param_data.new_zeros(padded_sharded_size)
    padded_sharded_param = SymmBufferRegistry.get_instance().get_or_create_symm_buffer(
        id(self), param_data.shape, param_data.dtype, shard_rank
    )
    if sharded_param.numel() > 0:
        padded_sharded_param.narrow(
            dim=shard_dim, start=0, length=sharded_param.size(shard_dim)
        ).copy_(sharded_param)
    if self.offload_to_cpu and not padded_sharded_param.is_meta:
        padded_sharded_param = padded_sharded_param.cpu()
        if self.pin_memory:
            padded_sharded_param = padded_sharded_param.pin_memory()
    self._sharded_param_data = padded_sharded_param.view(-1)
    length = sharded_param.size(shard_dim) if sharded_param.numel() > 0 else 0
    sharded_param = padded_sharded_param.narrow(dim=shard_dim, start=0, length=length)
    assert sharded_param.is_contiguous(), f"{self.fsdp_placement=}"
    self.sharded_param = nn.Parameter(self.to_sharded_dtensor(sharded_param))
    self.sharded_param.requires_grad_(param.requires_grad)
    # Let `param_data` be freed normally when its ref count reaches 0 when
    # the `fully_shard` call returns to allow provided parameters to alias
    self._setattr_on_modules(self.sharded_param)
    self.sharded_state = ShardedState.SHARDED


def reset_sharded_param(self):
    # For ops like `nn.Module._apply` or `load_state_dict(assign=True)`
    # that change the sharded parameter tensor, we may need to re-pad the
    # sharded local tensor and re-save the reference.
    module_info = self._module_info
    sharded_param_data = self._sharded_param_data
    new_param = getattr(module_info.module, module_info.param_name)
    if new_param is not self.sharded_param:
        if torch.__future__.get_swap_module_params_on_conversion():  # pylint: disable=no-member
            raise AssertionError(
                f"Expects swap_tensors to preserve object but got {new_param} "
                f"instead of {self.sharded_param}"
            )
        self.sharded_param = new_param
    # pyrefly: ignore  # missing-attribute
    local_tensor = new_param._local_tensor
    if local_tensor.is_meta:
        return
    # updated_local_tensor = False
    # [ODC]: Always update the sharded_param._local_tensor
    # because we need to change to use the symm buffer instead of the local tensor
    updated_local_tensor = True
    # local_tensor can be padded twice
    # 1st time in fully_shard(model)
    # 2nd time in model(input) lazy_init
    # 2nd time should be no-op if parameters remain unchanged
    # 2nd time shouldn't be no-op if people call model.load_state_dict(...) before lazy_init
    # this makes it possible for trainer to call `sd = model.state_dict()` before the training loop
    # and use `sd` without calling .state_dict() per iteration
    same_local_tensor = False
    # TODO: need to support tensor subclass
    if type(self._sharded_param_data) is torch.Tensor:  # pylint: disable=unidiomatic-typecheck
        same_local_tensor = (
            # when sharding param with shape (1, ...) over 2 ranks
            # local_tensor on rank 1 can be size 0, data_ptr() can be 0
            self._sharded_param_data.untyped_storage().data_ptr() > 0
            and self._sharded_param_data.untyped_storage().data_ptr()
            == local_tensor.untyped_storage().data_ptr()
        )
        if same_local_tensor:
            assert (
                self._sharded_param_data.device == local_tensor.device
            ), f"{self._sharded_param_data.device=} {local_tensor.device=}"
            assert (
                self._sharded_param_data.dtype == local_tensor.dtype
            ), f"{self._sharded_param_data.dtype=} {local_tensor.dtype=}"
    padded_sharded_size = self.padded_sharded_param_size
    shard_dim = self.fsdp_placement.dim
    length = local_tensor.size(shard_dim) if local_tensor.numel() > 0 else 0
    # if local_tensor.size() != padded_sharded_size and not same_local_tensor:
    if not same_local_tensor:
        assert shard_dim == 0, f"Shard({shard_dim}) requires even sharding: {local_tensor.size()=}"
        # padded_local_tensor = local_tensor.new_zeros(padded_sharded_size)
        assert (
            local_tensor.size() <= padded_sharded_size
        ), f"{local_tensor.size()=} {padded_sharded_size=}"
        if sharded_param_data.is_meta:
            shard_rank = self.mesh_info.shard_mesh_rank
            sharded_param_data = SymmBufferRegistry.get_instance().get_or_create_symm_buffer(
                id(self), sharded_param_data.shape, sharded_param_data.dtype, shard_rank
            )
        assert SymmBufferRegistry.is_nvshmem_tensor(
            sharded_param_data
        ), f"{sharded_param_data._odc_is_nvshmem=}"
        sharded_param_data.fill_(0)
        padded_local_tensor = sharded_param_data.view(local_tensor.shape)
        assert (
            padded_local_tensor.device == local_tensor.device
        ), f"{padded_local_tensor.device=} {local_tensor.device=}"
        assert (
            padded_local_tensor.dtype == local_tensor.dtype
        ), f"{padded_local_tensor.dtype=} {local_tensor.dtype=}"
        padded_local_tensor.narrow(dim=shard_dim, start=0, length=length).copy_(local_tensor)
        local_tensor = padded_local_tensor
        updated_local_tensor = True
    if self.pin_memory and not local_tensor.is_pinned():
        local_tensor = local_tensor.cpu().pin_memory()
        updated_local_tensor = True
    if not same_local_tensor:
        self._sharded_param_data = local_tensor.view(-1)
    assert isinstance(self.sharded_param, DTensor)  # mypy
    if updated_local_tensor:
        # Only change the local tensor object if needed
        self.sharded_param._local_tensor = local_tensor.narrow(
            dim=shard_dim, start=0, length=length
        )
        assert self.sharded_param._local_tensor.is_contiguous()
    self._sharding_spec = self.sharded_param._spec


@torch.no_grad()
def custom_get_param_all_gather_inputs(
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
        # assert use_foreach_copy(fsdp_param), f"{fsdp_param.param_dtype=} {fsdp_param.offload_to_cpu=} {hasattr(fsdp_param._sharded_local_tensor, 'fsdp_pre_all_gather')=}"
        assert not hasattr(fsdp_param._sharded_local_tensor, "fsdp_pre_all_gather")
        if use_foreach_copy(fsdp_param):
            foreach_copy_indices.append(i)
            # reshard_after_forward=int not supported for now
            # TODO: support reshard_after_forward=int
            assert fsdp_param.sharded_state == ShardedState.SHARDED, f"{fsdp_param.sharded_state=}"
            all_gather_input = (
                fsdp_param._sharded_param_data
                if fsdp_param.sharded_state == ShardedState.SHARDED
                else cast(torch.Tensor, fsdp_param._sharded_post_forward_param_data)
            )
            # foreach_copy_inputs.append(all_gather_input)
            # foreach_copy_input_numels.append(all_gather_input.numel())
            param_all_gather_inputs[i] = [all_gather_input]
        else:
            param_all_gather_inputs[i] = fsdp_param.all_gather_inputs

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
def custom_foreach_all_gather(
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
    gather_comm = ODCAllGather()
    with device_handle.stream(all_gather_copy_in_stream):
        param_all_gather_inputs = custom_get_param_all_gather_inputs(fsdp_params)
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
        # all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in(
        #     all_gather_inputs,
        #     all_gather_output,
        #     inp_split_sizes,
        #     all_gather_input_numel,
        #     rank,
        # )
        # del param_all_gather_inputs
        # symm_buffer = FSDPParamsGatherBuffers.get_instance().get_buffer(
        #     fsdp_params, all_gather_input, rank
        # )
        # symm_buffer.copy_(all_gather_input, non_blocking=True)
    all_gather_stream.wait_stream(all_gather_copy_in_stream)
    with device_handle.stream(all_gather_stream):
        all_gather_work = gather_comm.gather(
            output_tensor=all_gather_output,
            input_tensors=all_gather_inputs,
            group=group,
            async_op=async_op,
        )
        # all_gather_work = all_gather_comm(
        #     output_tensor=all_gather_output,
        #     # input_tensor=all_gather_input,
        #     input_tensor=symm_buffer,
        #     group=group,
        #     async_op=async_op,
        # )
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
    # state = fsdp_model._get_fsdp_state()
    # if (fsdp_param_group := state._fsdp_param_group) is not None:
    #     fsdp_param_group._all_gather_comm = comm
    # Get all FSDP states in the module tree (not just the root)
    from torch.distributed.fsdp._traversal_utils import _get_fsdp_states

    all_states = _get_fsdp_states(fsdp_model)

    # Set _all_gather_comm for all FSDP states that have a param group
    for state in all_states:
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
    FSDPParam._init_sharded_param = _init_sharded_param
    FSDPParam.reset_sharded_param = reset_sharded_param

    set_custom_all_gather(fsdp_model, ODCAllGather())


def stop():
    ODCReduceScatter.get_odc_reduction().stop()
    SymmBufferRegistry.get_instance().finalize()
