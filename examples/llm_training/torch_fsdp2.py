"""
FSDP2 Hybrid Sharding Training Example with Hugging Face Qwen 2.5 0.5B

This example demonstrates how to use PyTorch's Fully Sharded Data Parallel (FSDP2)
with HYBRID_SHARD strategy: 8 GPUs configured as 2 replicas * 4 FSDP groups.
Includes gradient accumulation and deterministic training for reproducible results.

Key Features:
- Uses real HuggingFace Qwen 2.5 0.5B model (not from config)
- Loads SQuAD dataset for meaningful text training
- Deterministic training with fixed seeds for reproducible results
- Fast convergence settings to see loss decrease quickly
- Comprehensive logging to track training progress

Deterministic Training:
- All random seeds are fixed (Python, NumPy, PyTorch)
- CUDNN deterministic algorithms enabled
- Data loading is deterministic (no shuffle, no workers)
- Results should be identical across runs with same configuration

Usage:
    torchrun --nproc_per_node=8 --nnodes=1 torch_fsdp2.py
"""

import contextlib
import functools
import os
import random
import time

import numpy as np
import packing
import torch
import torch.distributed as dist
import wandb
from args import get_args
from datasets import load_from_disk
from lm_head import FusedLinearForPPOFunction
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.optim import AdamW
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config

from data import BatchedDataset
from odc import init_nvshmem
from odc.fsdp import fsdp2
from odc.primitives.utils import SymmBufferRegistry, get_local_world_size

enable_decouple = os.environ.get("ODC", "0") == "1"
enable_profiler = os.environ.get("TORCH_PROFILED", "0") == "1"


param_dtype = torch.bfloat16
reduce_dtype = torch.float32
buffer_dtype = torch.float32


args = get_args()


def forward_with_fused_linear(
    self,
    input_ids,
    _attention_mask=None,
    position_ids=None,
    _logits_to_keep=0,
    _temperature=None,
    **_loss_kwargs,
):
    base_outputs = self.model(input_ids=input_ids, attention_mask=None, position_ids=position_ids)
    hidden_states = base_outputs.last_hidden_state
    labels = torch.roll(input_ids, shifts=-1, dims=-1)
    vocab_weights = self.lm_head.weight
    logps, _ = FusedLinearForPPOFunction.apply(hidden_states, vocab_weights, labels, 1.0)
    return logps


def create_qwen_model():
    """Create Qwen 2.5 0.5B model with random initialization"""
    # Use the smallest Qwen 2.5 model (0.5B parameters)
    model_name = args.model_name

    # Load the exact configuration from HuggingFace
    config = Qwen2Config.from_pretrained(
        model_name, trust_remote_code=True, attn_implementation="flash_attention_2"
    )

    # Modify config for training
    config.use_cache = False  # Disable cache for training
    # config.torch_dtype = torch.bfloat16
    config.torch_dtype = torch.float32

    # Create model from config (this will use random initialization)

    # model = Qwen2ForCausalLM(config)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=config.torch_dtype,
        # use_cache=False,
    )
    model.__class__.forward = forward_with_fused_linear

    model.gradient_checkpointing_enable()
    model = model.to(config.torch_dtype)
    print(model)
    packing.get_seq_costs_flops = functools.partial(packing._get_seq_costs_flops, model)

    # Load tokenizer (we still need the tokenizer from pretrained)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="right"
    )

    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def set_deterministic_training(seed=42):
    """Set all random seeds and configurations for deterministic training"""
    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for deterministic algorithms
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Enable deterministic algorithms in PyTorch (may affect performance)
    torch.use_deterministic_algorithms(True, warn_only=True)

    if dist.get_rank() == 0:
        print(f"Deterministic training enabled with seed: {seed}")


def setup_distributed():
    """Initialize distributed training"""
    # Initialize process group
    dist.init_process_group(backend="nccl")

    # Set device
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    if enable_decouple:
        init_nvshmem()


def create_fsdp_model(model, _sharding_group, _replication_group):
    """Wrap model with FSDP2 using fully_shard"""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    if args.model_name.startswith("deepseek-ai/DeepSeek-R1-Distill-Qwen"):
        _layer_cls = Qwen2DecoderLayer
    else:
        raise NotImplementedError(f"Model {args.model_name} not supported")

    # FSDP2 configuration with MixedPrecisionPolicy
    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        ),
    }
    if os.environ.get("HSDP", "0") == "1":
        gpus_per_node = get_local_world_size()
        world_size = dist.get_world_size()
        num_nodes = world_size // gpus_per_node
        print(f"enable HSDP with {num_nodes} nodes and {gpus_per_node} GPUs per node")
        assert world_size % gpus_per_node == 0, f"{world_size=} {gpus_per_node=}"
        fsdp_kwargs["mesh"] = init_device_mesh(
            "cuda", (num_nodes, gpus_per_node), mesh_dim_names=("dp_replicate", "dp_shard")
        )

    if enable_decouple:
        fsdp2.patch_fsdp2()

    # Wrap each transformer layer with fully_shard
    for layer in model.model.layers:
        fully_shard(layer, **fsdp_kwargs)

    # Wrap the entire model with fully_shard
    fsdp_model = fully_shard(model, **fsdp_kwargs)

    return fsdp_model


def setup_wandb(rank, _world_size, config):
    """Initialize wandb for experiment tracking (only on rank 0)"""
    if rank == 0:
        # Initialize wandb
        wandb.init(
            project=args.project_name,
            name=args.run_name,
            config=config,
            tags=["fsdp2", "hybrid-sharding", "qwen2.5", "distributed-training"],
        )
        print("Wandb initialized successfully!")


def create_profiler(rank):
    """Create torch profiler for the first 4 training steps"""
    if not enable_profiler:
        return None

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


class Timer:
    def __init__(self, name):
        self.name = name
        self.start_time = 0.0
        self.end_time = 0.0

    def start(self):
        torch.cuda.synchronize()
        self.start_time = time.time()

    def stop(self):
        torch.cuda.synchronize()
        self.end_time = time.time()
        return self.end_time - self.start_time


def train_step(model, batch):
    """Single training step with gradient accumulation"""
    with torch.cuda.nvtx.range("data_transfer"):
        input_ids, position_ids, attention_mask, loss_scale, _num_seqs = (
            batch["input_ids"],
            batch["position_ids"],
            batch["attention_mask"],
            batch["loss_scale"],
            batch["num_seqs"],
        )
        # print(input_ids)
        # print(labels)
        input_ids = input_ids.cuda(non_blocking=True)
        position_ids = position_ids.cuda(non_blocking=True)
        attention_mask = attention_mask.cuda(non_blocking=True)
        loss_scale = loss_scale.cuda(non_blocking=True)

    # Forward pass
    with torch.cuda.nvtx.range("forward_pass"):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logps = model(input_ids=input_ids, attention_mask=None, position_ids=position_ids)
            # NOTE: this is borrowed from verl to avoid allocating full logits on HBM.
            loss = (-logps.squeeze(0) * loss_scale.squeeze(0)).sum()

    # Backward pass
    if not args.forward_only:
        with torch.cuda.nvtx.range("backward_pass"):
            loss.backward()

    return loss.item()


def main():
    setup_distributed()
    if args.forward_only:
        src_tensor = torch.randn(1024, dtype=torch.bfloat16, device="cuda")
        fsdp2.get_reduction_service().scatter_accumulate(0, src_tensor, dist.group.WORLD)

    # Main training function
    # Set deterministic training first
    set_deterministic_training(seed=42)

    # Setup distributed training

    # Create Qwen 2.5 0.5B model
    model, _tokenizer = create_qwen_model()
    # print('original model dtype', model.dtype)

    # Wrap with FSDP2 using fully_shard
    model = create_fsdp_model(model, None, None)
    # print(model.dtype)

    # Patch lazy init if decouple is enabled
    if enable_decouple:
        for layer in model.model.layers:
            fsdp2.patch_lazy_init(layer)
        fsdp2.patch_lazy_init(model)

    # Optimizer with higher learning rate for faster convergence
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    # Create dataset and dataloader
    dataset = BatchedDataset(
        load_from_disk(args.dataset),
        batch_size=args.minibatch_size,
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        packing_method=args.packing_method,
    )

    # Worker init function for deterministic data loading
    def _worker_init_fn(_worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # Training parameters
    num_epochs = 5  # More epochs to see loss decrease
    max_steps = 25  # Limit steps for demo purposes

    # Setup wandb configuration
    wandb_config = {
        "model": args.model_name,
        "dataset": args.dataset,
        "world_size": dist.get_world_size(),
        "sharding_strategy": "HYBRID_SHARD",
        "minibatch_size": args.minibatch_size,
        "micro_batch_size": args.micro_batch_size,
        "packing_method": args.packing_method,
        "learning_rate": 5e-4,
        "weight_decay": 0.01,
        "max_length": 16384,
        "num_epochs": num_epochs,
        "max_steps": max_steps,
        "precision": "bfloat16",
        "grad_clip_norm": 1.0,
        "seed": 42,
    }

    # Initialize wandb (only on rank 0)
    setup_wandb(dist.get_rank(), dist.get_world_size(), wandb_config)

    print(f"Rank {dist.get_rank()}: Starting deterministic training...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Batch size: {args.minibatch_size}, Max steps: {max_steps}")
    if dist.get_rank() == 0:
        print(f"Profiler: {'ENABLED' if enable_profiler else 'DISABLED'}")

    # Create profiler for this rank
    prof = create_profiler(dist.get_rank())

    # Training loop
    model.train()

    # Use context manager only if profiler is enabled
    profiler_context = prof if prof is not None else contextlib.nullcontext()
    global_step = 0

    with profiler_context:
        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            if epoch == num_epochs - 1:
                torch.cuda.cudart().cudaProfilerStart()

            for minibatch in dataset:
                accumulation_steps = len(minibatch)
                minibatch_loss = torch.tensor(0.0).to("cuda")
                minibatch_seq_length = 0
                global_step += 1
                print(f"rank {dist.get_rank()}: accumulation_steps: {accumulation_steps}")

                if enable_decouple:
                    fsdp2.pre_minibatch_start(model)

                minibatch_start_time = time.time()
                for idx, micro_batch in enumerate(minibatch):
                    # Training step with profiling
                    # torch.cuda.synchronize()
                    # start_time = time.time()

                    with torch.cuda.nvtx.range(f"train_step_{global_step}_{idx}"):
                        loss = train_step(
                            model,
                            micro_batch,
                        )
                        # TODO: the gradient scaling should be dynamic.

                    # torch.cuda.current_stream().synchronize()
                    # end_time = time.time()
                    # # rank 4 total time: 6.0466, batch shape: torch.Size([1, 40843])
                    # print(f"rank {dist.get_rank()} total time: {end_time - start_time:.4f}, batch shape: {micro_batch['input_ids'].shape}")

                    minibatch_loss += loss
                    minibatch_seq_length += micro_batch["input_ids"].shape[1]

                    # Step the profiler after each training step (only if enabled)
                    if prof is not None:
                        prof.step()

                sync_time = 0.0
                optimizer_step_start_time = time.time()
                grad_norm = torch.tensor(0.0).to("cuda")
                if not args.forward_only:
                    if enable_decouple:
                        fsdp2.pre_optimizer_step(model)
                    if enable_decouple:
                        timer = Timer("pre_optimizer_step")
                        timer.start()
                        # In FSDP2, gradient reduction happens during backward
                        # No explicit pre_optimizer_step call needed
                        sync_time = timer.stop()
                        sync_time = torch.tensor(sync_time).to("cuda")
                        torch.distributed.all_reduce(sync_time, op=torch.distributed.ReduceOp.SUM)
                        sync_time = sync_time.item() / dist.get_world_size()
                    with torch.cuda.nvtx.range("optimizer_step"):
                        # Calculate and print gradient norm before clipping
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=float("inf")
                        )
                        # Gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        # Optimizer step
                        optimizer.step()
                        optimizer.zero_grad()

                dist.barrier()
                torch.cuda.synchronize()
                minibatch_end_time = time.time()
                print(
                    f"Minibatch time: {minibatch_end_time - minibatch_start_time:.4f}s, Sync time: {sync_time:.4f}s\n"
                )
                torch.distributed.all_reduce(minibatch_loss, op=torch.distributed.ReduceOp.SUM)
                torch_memory = torch.cuda.max_memory_allocated()
                peak_memory = torch_memory
                if enable_decouple:
                    peak_memory += SymmBufferRegistry.get_instance().memory_allocated()
                minibatch_log = {
                    "loss": minibatch_loss.item() / torch.distributed.get_world_size(),
                    "minibatch_time": minibatch_end_time - minibatch_start_time,
                    "sync_time": sync_time,
                    "optimizer_step_time": minibatch_end_time - optimizer_step_start_time,
                    "seq_length": minibatch_seq_length,
                    "grad_norm": grad_norm.item(),
                    "torch_memory": torch_memory / 1024 / 1024 / 1024,
                    "peak_memory": peak_memory / 1024 / 1024 / 1024,
                    "epoch_idx": epoch,
                }
                if dist.get_rank() == 0:
                    wandb.log(minibatch_log, step=global_step)
                if global_step >= max_steps:
                    break

            if epoch == num_epochs - 1:
                torch.cuda.cudart().cudaProfilerStop()

            if dist.get_rank() == 0:
                epoch_end_time = time.time()
                epoch_total_time = epoch_end_time - epoch_start_time
                epoch_log = {
                    "epoch/epoch_time": epoch_total_time,
                }
                wandb.log(epoch_log, step=epoch)
                loss = minibatch_loss.item() / torch.distributed.get_world_size()
                print(f"Epoch {epoch} completed. Loss: {loss}, Total time: {epoch_total_time:.2f}s")

    if dist.get_rank() == 0:
        print("Deterministic training completed successfully!")
        # Finish wandb run
        wandb.finish()
        print("Wandb run finished!")

    torch.distributed.barrier()
    torch.cuda.synchronize()
    print(f"Rank {dist.get_rank()}: Training completed!")

    # Cleanup
    if enable_decouple:
        fsdp2.stop()
    dist.destroy_process_group()

    completion_file = f"logs/{args.project_name}/{args.run_name}.done"
    os.makedirs(os.path.dirname(completion_file), exist_ok=True)
    with open(completion_file, "w", encoding="utf-8") as f:
        f.write("done")


if __name__ == "__main__":
    main()
