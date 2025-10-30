import os
import pathlib

import nvidia.nvshmem
import nvshmem.bindings
import triton
import triton.compiler.compiler as compiler_module
import triton.language as tl
from packaging import version
from triton.language import core

# Monkey patch the CompiledKernel._init_handles or it will crash on:
#     RuntimeError: CUDA error: an illegal memory access was encountered
original_init_handles = compiler_module.CompiledKernel._init_handles


nvshmem_init_kernels = set()


def patched_init_handles(self):
    original_init_handles(self)
    extern_libs = self.metadata.extern_libs
    enable_nvshmem = any(
        lib_name == LIB_NAME or path == LIB_NVSHMEM_PATH for lib_name, path in extern_libs
    )
    # print(f"extern_libs: {extern_libs} enable_nvshmem: {enable_nvshmem}")
    # Check if kernel uses NVSHMEM
    key = (self.name, self.module)
    if enable_nvshmem and key not in nvshmem_init_kernels:
        assert self.module is not None, "Module is None"
        nvshmem.bindings.nvshmem.cumodule_init(self.module)
        nvshmem_init_kernels.add(key)


compiler_module.CompiledKernel._init_handles = patched_init_handles


def get_nvshmem_home_path():
    if "NVSHMEM_HOME" in os.environ:
        return pathlib.Path(os.environ["NVSHMEM_HOME"])
    return pathlib.Path(nvidia.nvshmem.__path__[0])


def is_triton_version_supported():
    return version.parse(triton.__version__) >= version.parse("3.4.0")


LIB_NVSHMEM_PATH = str(get_nvshmem_home_path() / "lib" / "libnvshmem_device.bc")
LIB_NAME = "libnvshmem"


NVSHMEM_EXTERN_LIBS = {
    LIB_NAME: LIB_NVSHMEM_PATH,
}


@core.extern
def _tid_wrapper(axis: core.constexpr, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [],
        {
            (): (
                f"llvm.nvvm.read.ptx.sreg.tid.{axis.value}",
                core.dtype("int32"),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


# pylint: disable=inconsistent-return-statements
@core.extern
def tid(axis: core.constexpr, _semantic=None):
    if axis == 0:
        return _tid_wrapper(core.constexpr("x"), _semantic=_semantic)
    elif axis == 1:
        return _tid_wrapper(core.constexpr("y"), _semantic=_semantic)
    elif axis == 2:
        return _tid_wrapper(core.constexpr("z"), _semantic=_semantic)
    else:
        tl.static_assert(False, "axis must be 0, 1 or 2", _semantic=_semantic)


@triton.jit
def int_atomic_compare_swap(dest, cond, value, pe):
    return int_atomic_compare_swap_wrapper(dest.to(tl.int64), cond, value, pe)


@core.extern
def int_atomic_compare_swap_wrapper(dest, cond, value, pe, _semantic=None):
    return core.extern_elementwise(
        LIB_NAME,
        LIB_NVSHMEM_PATH,
        [dest, cond, value, pe],
        {
            (tl.int64, tl.int32, tl.int32, tl.int32): (
                "nvshmem_int_atomic_compare_swap",
                tl.int32,
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@triton.jit
def int_atomic_swap(dest, value, pe):
    return int_atomic_swap_wrapper(dest.to(tl.int64), value, pe)


@core.extern
def int_atomic_swap_wrapper(dest, value, pe, _semantic=None):
    return core.extern_elementwise(
        LIB_NAME,
        LIB_NVSHMEM_PATH,
        [dest, value, pe],
        {
            (tl.int64, tl.int32, tl.int32): (
                "nvshmem_int_atomic_swap",
                tl.int32,
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def _putmem_impl(
    dest,
    source,
    nbytes,
    pe,
    SCOPE_SUFFIX: core.constexpr,
    NBI: core.constexpr = core.constexpr(""),
    _semantic=None,
):
    # In 3.3.1, this will have: Parsing error near '.nvvm': syntax error
    # which is fixed in 3.4.0: https://github.com/triton-lang/triton/pull/6225
    assert (
        is_triton_version_supported()
    ), "putmem_nbi_block is only supported in Triton v3.4.0 and above"
    name = f"nvshmem{'x' if SCOPE_SUFFIX.value else ''}_putmem{NBI.value}{SCOPE_SUFFIX.value}"
    return core.extern_elementwise(
        LIB_NAME,
        LIB_NVSHMEM_PATH,
        [
            dest,
            source,
            nbytes,
            pe,
        ],
        {
            (
                tl.int64,
                tl.int64,
                tl.uint64,
                tl.int32,
            ): (name, tl.int32),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@triton.jit
def putmem_nbi_block(dest, source, nbytes, pe):
    return _putmem_impl(
        dest.to(tl.int64),
        source.to(tl.int64),
        nbytes.to(tl.uint64),
        pe,
        core.constexpr("_block"),
        core.constexpr("_nbi"),
    )


@core.extern
def _getmem_impl(
    dest,
    source,
    nbytes,
    pe,
    SCOPE_SUFFIX: core.constexpr,
    NBI: core.constexpr = core.constexpr(""),
    _semantic=None,
):
    return core.extern_elementwise(
        LIB_NAME,
        LIB_NVSHMEM_PATH,
        [
            dest,
            source,
            nbytes,
            pe,
        ],
        {
            (
                tl.int64,
                tl.int64,
                tl.uint64,
                tl.int32,
            ): (
                f"nvshmem{'x' if SCOPE_SUFFIX.value else ''}_getmem{NBI.value}{SCOPE_SUFFIX.value}",
                tl.int32,
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@triton.jit
def getmem_nbi_block(dest, source, nbytes, pe):
    return _getmem_impl(
        dest.to(tl.int64),
        source.to(tl.int64),
        nbytes.to(tl.uint64),
        pe,
        core.constexpr("_block"),
        core.constexpr("_nbi"),
    )


@core.extern
def quiet(_semantic=None):
    return core.extern_elementwise(
        LIB_NAME,
        LIB_NVSHMEM_PATH,
        [],
        {
            (): ("nvshmem_quiet", tl.int32),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@triton.jit
def int_p(dest, value, pe):
    return int_p_wrapper(dest.to(tl.int64), value, pe)


@core.extern
def int_p_wrapper(dest, value, pe, _semantic=None):
    return core.extern_elementwise(
        LIB_NAME,
        LIB_NVSHMEM_PATH,
        [dest, value, pe],
        {
            (tl.int64, tl.int32, tl.int32): ("nvshmem_int_p", tl.int32),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@triton.jit
def int_g(src, pe):
    return int_g_wrapper(src.to(tl.int64), pe)


@core.extern
def int_g_wrapper(src, pe, _semantic=None):
    return core.extern_elementwise(
        LIB_NAME,
        LIB_NVSHMEM_PATH,
        [src, pe],
        {
            (tl.int64, tl.int32): ("nvshmem_int_g", tl.int32),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def __syncthreads(_semantic=None):
    return tl.tensor(_semantic.builder.create_barrier(), tl.void)
