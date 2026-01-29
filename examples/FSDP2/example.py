import argparse
import contextlib
import os
import sys

import torch
from checkpoint import Checkpointer
from model import ModelArgs, Transformer
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.profiler import ProfilerActivity, profile
from utils import inspect_mixed_precision, inspect_model

from odc import init_nvshmem
from odc.fsdp import fsdp2
from odc.primitives.utils import SymmBufferRegistry, get_local_world_size

enable_decouple = os.environ.get("ODC", "0") == "1"
enable_profiler = os.environ.get("TORCH_PROFILED", "0") == "1"
enable_cuda_profiler = os.environ.get("CUDA_PROFILED", "0") == "1"


def verify_min_gpu_count(min_gpus: int = 2) -> bool:
    """verification that we have at least 2 gpus to run dist examples"""
    has_gpu = torch.accelerator.is_available()
    gpu_count = torch.accelerator.device_count()
    return has_gpu and gpu_count >= min_gpus


def set_modules_to_forward_prefetch(model, num_to_forward_prefetch):
    for i, layer in enumerate(model.layers):
        if i >= len(model.layers) - num_to_forward_prefetch:
            break
        layers_to_prefetch = [model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)


def set_modules_to_backward_prefetch(model, num_to_backward_prefetch):
    for i, layer in enumerate(model.layers):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)


def main(args):
    _min_gpu_count = 2
    if not verify_min_gpu_count(min_gpus=_min_gpu_count):
        print(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")
        sys.exit(1)
    rank = int(os.environ["LOCAL_RANK"])
    if torch.accelerator.is_available():
        device_type = torch.accelerator.current_accelerator()
        device = torch.device(f"{device_type}:{rank}")
        torch.accelerator.set_device_index(rank)
        print(f"Running on rank {rank} on device {device}")
    else:
        device = torch.device("cpu")
        print(f"Running on device {device}")

    backend = torch.distributed.get_default_backend_for_device(device)  # pylint: disable=no-member
    torch.distributed.init_process_group(backend=backend, device_id=device)
    if enable_decouple:
        init_nvshmem()

    torch.manual_seed(0)
    vocab_size = 1024
    batch_size = 32
    seq_len = 64
    model_args = ModelArgs(
        n_layers=1,
        n_heads=4,
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        dim=1024,
        dropout_p=0,
    )
    with torch.device("meta"):
        model = Transformer(model_args)
    fsdp_kwargs = {}
    if args.mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    gpus_per_node = get_local_world_size()
    hpz = os.environ.get("HPZ", "0") == "1"
    hsdp = os.environ.get("HSDP", "0") == "1"
    if hpz:
        print(f"enable Hierarchical Partitioning for ZeRO(HPZ) with {gpus_per_node} GPUs per node")
        fsdp_kwargs["reshard_after_forward"] = gpus_per_node
    elif hsdp:
        world_size = torch.distributed.get_world_size()
        num_nodes = world_size // gpus_per_node
        print(f"enable HSDP with {num_nodes} nodes and {gpus_per_node} GPUs per node")
        assert world_size % gpus_per_node == 0, f"{world_size=} {gpus_per_node=}"
        fsdp_kwargs["mesh"] = init_device_mesh(
            "cuda", (num_nodes, gpus_per_node), mesh_dim_names=("dp_replicate", "dp_shard")
        )

    if enable_decouple:
        fsdp2.patch_fsdp2()
        print("enable ODC")
    else:
        fsdp2.patch_debug()
        print("disable odc and enable debug mode")

    for layer in model.layers:
        fully_shard(layer, **fsdp_kwargs)
    fsdp_model = fully_shard(model, **fsdp_kwargs)

    inspect_model(model)

    if args.explicit_prefetching:
        set_modules_to_forward_prefetch(model, num_to_forward_prefetch=2)
        set_modules_to_backward_prefetch(model, num_to_backward_prefetch=2)

    checkpointer = Checkpointer("checkpoints", dcp_api=args.dcp_api)
    if checkpointer.last_training_time is None:
        model.to_empty(device=device)
        model.reset_parameters()
    else:
        checkpointer.load_model(model)

    if enable_decouple:
        for layer in fsdp_model.layers:
            fsdp2.patch_lazy_init(layer)
        fsdp2.patch_lazy_init(fsdp_model)

    if args.mixed_precision:
        inspect_mixed_precision(model)

    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    if checkpointer.last_training_time is not None:
        checkpointer.load_optim(model, optim)

    prof = create_profiler(rank)
    profiler_context = prof if prof is not None else contextlib.nullcontext()
    cuda_prof = cuda_profiler_context() if enable_cuda_profiler else contextlib.nullcontext()

    with cuda_prof, profiler_context:
        num_microbatches = 2
        for epoch in range(2):
            if enable_decouple:
                fsdp2.pre_minibatch_start(fsdp_model)
            if args.explicit_prefetching:
                model.unshard()

            with torch.cuda.nvtx.range(f"epoch_{epoch}"):
                for mb_idx, mb in enumerate(range(num_microbatches)):
                    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
                    with torch.cuda.nvtx.range(f"forward_{mb}"):
                        loss = model(x).sum()
                    with torch.cuda.nvtx.range(f"backward_{mb}"):
                        loss.backward()
                    if prof is not None:
                        prof.step()
                    torch.cuda.synchronize()
                    print(f"microbatch {mb_idx}")
                if enable_decouple:
                    fsdp2.pre_optimizer_step(model)
                else:
                    fsdp2.original_impl_pre_optimizer_step(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()
                optim.zero_grad()
                print(f"epoch {epoch} loss: {loss.detach().item()}")

    torch_memory_allocated = torch.cuda.max_memory_allocated()
    symm_buffer_memory_allocated = SymmBufferRegistry.get_instance().memory_allocated()
    print(f"Rank {rank} {torch_memory_allocated=} {symm_buffer_memory_allocated=}")

    # checkpointer.save(model, optim)
    if enable_decouple:
        fsdp2.stop()
    torch.distributed.destroy_process_group()


def create_profiler(rank):
    """Create torch profiler for the first 4 training steps"""
    if not enable_profiler:
        return None

    print("Enable torch profiler")
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    profiler = profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
        # Profile first 4 steps: warmup + 3 active steps
        schedule=torch.profiler.schedule(
            wait=0,  # No initial wait
            warmup=1,  # 1 warmup step
            active=1,  # Profile 3 active steps (steps 1-3)
            repeat=1,  # Only run once
        ),
        on_trace_ready=lambda prof: prof.export_chrome_trace(f"fsdp_profile_rank_{rank}.json"),
    )

    return profiler


@contextlib.contextmanager
def cuda_profiler_context():
    torch.cuda.cudart().cudaProfilerStart()
    yield
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch FSDP2 example")
    parser.add_argument("--explicit-prefetching", action="store_true", default=False)
    parser.add_argument("--mixed-precision", action="store_true", default=False)
    parser.add_argument("--dcp-api", action="store_true", default=False)
    arguments = parser.parse_args()

    if enable_decouple:
        print("Running in decoupled mode")
    main(arguments)
