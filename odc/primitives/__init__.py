import glob
import sys
from pathlib import Path

from .nvshmem_triton import (
    BC_PATH,
    LIB_NVSHMEM_PATH,
    NVSHMEM_EXTERN_LIBS,
    __syncthreads,
    getmem_nbi_block,
    int_atomic_compare_swap,
    int_atomic_swap,
    int_g,
    int_p,
    putmem_nbi_block,
    quiet,
    tid,
)

SO_FILE_NAME = "tensor_ipc*.so"


def get_tensor_ipc_lib_paths():
    paths = []
    so_files = glob.glob(str(Path(__file__).parent / SO_FILE_NAME))
    for so_file in so_files:
        paths.append(Path(so_file))
    project_root = Path(__file__).parent.parent.parent
    # scan the build directory for the .so file for development
    build_dir = project_root / "build"
    if build_dir.exists():
        so_files = glob.glob(str(build_dir / "**" / SO_FILE_NAME), recursive=True)
        # Add found .so files to paths
        for so_file in so_files:
            paths.append(Path(so_file))
    return paths


_lib_paths = get_tensor_ipc_lib_paths()
_lib_path = next((p for p in _lib_paths if p.exists()), None)
if _lib_path is None:
    raise ImportError(f"Could not find tensor_ipc shared library in {_lib_paths}")

_lib_dir = str(_lib_path.parent)
if _lib_dir not in sys.path:
    sys.path.insert(0, _lib_dir)

try:
    import tensor_ipc  # Import directly since we added the directory to sys.path

    get_ipc_handle = tensor_ipc.get_ipc_handle
    reconstruct_tensor = tensor_ipc.reconstruct_tensor
except ImportError as e:
    raise ImportError(f"Failed to import {_lib_path}: {e}") from e

__all__ = [
    # nvshmem_triton
    "int_atomic_compare_swap",
    "int_atomic_swap",
    "putmem_nbi_block",
    "getmem_nbi_block",
    "quiet",
    "int_p",
    "int_g",
    "tid",
    "__syncthreads",
    "BC_PATH",
    "LIB_NVSHMEM_PATH",
    "NVSHMEM_EXTERN_LIBS",
    # tensor_ipc
    "get_ipc_handle",
    "reconstruct_tensor",
]
