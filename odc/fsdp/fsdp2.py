from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp._fully_shard._fsdp_api import AllGather, ReduceScatter
from torch.distributed.fsdp._fully_shard._fsdp_collectives import DefaultAllocMixin

from odc.primitives.gather import GatherService


class ODCAllGather(DefaultAllocMixin, AllGather):
    _odc_all_gather_instance = None

    @classmethod
    def get_odc_all_gather(cls) -> GatherService:
        if cls._odc_all_gather_instance is None:
            cls._odc_all_gather_instance = GatherService()
        return cls._odc_all_gather_instance

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False,
    ) -> Optional[dist.Work]:
        if async_op:
            event = torch.cuda.Event()
        gather = self.get_odc_all_gather()
        gather.gather_into_tensor(output_tensor, input_tensor, group)
        if async_op:
            event.record()
            return event
        return None
        # return dist.all_gather_into_tensor(
        #     output_tensor,
        #     input_tensor,
        #     group=group,
        #     async_op=async_op,
        # )


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
