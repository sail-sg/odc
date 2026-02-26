import torch
import torch.nn.functional as F
from model import Transformer
from torch.distributed.fsdp import FSDPModule

# from torch.distributed.tensor import Shard


def inspect_model(model: FSDPModule):
    assert isinstance(model, Transformer)
    assert isinstance(model, FSDPModule)

    if torch.distributed.get_rank() == 0:
        print(model)

    for param in model.parameters():
        # assert param.placements == (Shard(0),)
        assert param.dtype == torch.float32
        # print(param.get_local_tensor())


def inspect_mixed_precision(model: FSDPModule):
    model.unshard()
    for param in model.parameters(recurse=False):
        assert param.dtype == torch.bfloat16
    model.reshard()


def hash_tensor(x: torch.Tensor) -> str:
    if x is None:
        return x

    y = x.detach()
    if y.numel() == 0:
        return f"empty:{y.dtype}:{tuple(y.shape)}"

    u8 = y.contiguous().view(torch.uint8)
    pad_len = (-u8.numel()) % 8
    if pad_len:
        u8 = F.pad(u8, (0, pad_len))

    i64 = u8.view(torch.int64)
    lcg_a = torch.tensor(6364136223846793005, device=y.device, dtype=torch.int64)
    lcg_c = torch.tensor(1442695040888963407, device=y.device, dtype=torch.int64)
    mixed = i64 * lcg_a + lcg_c
    h1 = torch.sum(mixed, dtype=torch.int64)
    mix = torch.tensor(-7046029254386353131, device=y.device, dtype=torch.int64)
    h2 = torch.sum(mixed * mix, dtype=torch.int64)
    h1_u64 = h1.item() & ((1 << 64) - 1)
    h2_u64 = h2.item() & ((1 << 64) - 1)
    return f"{h1_u64:016x}{h2_u64:016x}:{y.dtype}:{tuple(y.shape)}"


def hash_optimizer_grads(optim: torch.optim.Optimizer) -> str:
    grad_hashes = []
    for group_idx, group in enumerate(optim.param_groups):
        for param_idx, param in enumerate(group["params"]):
            grad = param.grad
            if grad is None:
                grad_hashes.append(f"{group_idx}:{param_idx}:none")
                continue
            if grad.is_sparse:
                grad = grad.to_dense()
            grad = grad._local_tensor
            grad_hashes.append(f"{group_idx}:{param_idx}:{hash_tensor(grad)}")
    return "|".join(grad_hashes)
