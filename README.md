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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import odc
from odc.fsdp import fsdp1


fsdp1.patch_fsdp1()

torch.distributed.init_process_group(backend="nccl", device_id=device)
odc.init_nvshmem()


fsdp_model = FSDP(
    model,
    # ...
)

for epoch in range(10):
    fsdp1.pre_minibatch_start(fsdp_model)
    for minibatch in dataset:
        loss = loss_fn(model)
        fsdp1.pre_optimizer_step(model)
        optimizer.step()
        optimizer.zero_grad()

fsdp1.stop()
```

### Basic Usage with FSDP2

```python
import torch
import odc
from odc.fsdp import fsdp2


torch.distributed.init_process_group(backend="nccl", device_id=device)
odc.init_nvshmem()

fsdp2.patch_fsdp2()

for layer in model.layers:
    fully_shard(layer, **fsdp_kwargs)
fsdp_model = fully_shard(model, **fsdp_kwargs)

# Call patch_lazy_init just as how we call fully_shard above.
for layer in fsdp_model.layers:
    fsdp2.patch_lazy_init(layer)
fsdp2.patch_lazy_init(fsdp_model)

for epoch in range(10):
    fsdp2.pre_minibatch_start(fsdp_model)
    for minibatch in dataset:
        loss = loss_fn(model)
        fsdp2.pre_optimizer_step(model)
        optimizer.step()
        optimizer.zero_grad()

fsdp2.stop()
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
