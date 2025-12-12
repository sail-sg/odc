# On-demand Communication (ODC)

ODC is a high-performance communication library that adapts Parameter Server (PS) into Fully Sharded Data Parallel (FSDP) by replacing collective all-gather and
reduce-scatter with on-demand point-to-point communication.

![Original-FSDP](./docs/readme/FSDP-ODC.jpg)

## Usage

### Prerequisites

- PyTorch (with CUDA support)
- CUDA 12.x
- Python >= 3.8

We highly recommand using a pytorch container like
- `nvcr.io/nvidia/pytorch:25.06-py3` [Full List](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags)
- `2.9.0-cuda12.8-cudnn9-devel` [Full List](https://hub.docker.com/r/pytorch/pytorch/tags)

### Install ODC
```
pip install --no-build-isolation -e .
```
It **requires some time** to compile the CUDA extension `tensor_ipc`.

## Quick Start

A complete example is provided in `examples/llm_training/`:
```shell
bash examples/llm_training/run.sh
```

### Basic Usage with FSDP1

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import odc
from odc.fsdp import fsdp1 # import patch_fsdp1, pre_minibatch_start, pre_optimizer_step, stop


fsdp1.patch_fsdp1()

dist.init_process_group(backend="nccl")
odc.init_nvshmem()


fsdp_model = FSDP(
    model,
    # ...
)

for minibatch in dataset:
    fsdp1.pre_minibatch_start()
    loss = train_step(model, ...)
    fsdp1.pre_optimizer_step(model)
    optimizer.step()
    optimizer.zero_grad()

fsdp1.stop()
```

## Development

### Running Linter
```
make lint
```

### Running Tests
```bash
make test
```
