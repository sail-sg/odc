import builtins
import os
import pathlib

import nvidia.nvshmem
import nvshmem.bindings
import triton
import triton.compiler.compiler as compiler_module
import triton.language as tl
from packaging import version
from triton.language import core
from triton.language.core import builtin, dispatch

# Monkey patch the CompiledKernel._init_handles or it will crash on:
#     RuntimeError: CUDA error: an illegal memory access was encountered
original_init_handles = compiler_module.CompiledKernel._init_handles


nvshmem_init_kernels = set()


def patched_init_handles(self):
    original_init_handles(self)
    extern_libs = self.metadata.extern_libs
    enable_nvshmem = any(lib_name in (LIB_NAME, LIB_NVSHMEM_NAME) for lib_name, _ in extern_libs)
    # print(f"extern_libs: {extern_libs} enable_nvshmem: {enable_nvshmem}")
    # Check if kernel uses NVSHMEM
    key = (self.name, self.module)
    if enable_nvshmem and key not in nvshmem_init_kernels:
        assert self.module is not None, "Module is None"
        nvshmem.bindings.nvshmem.cumodule_init(self.module)
        nvshmem_init_kernels.add(key)


compiler_module.CompiledKernel._init_handles = patched_init_handles


class hashable_pointer_type(tl.pointer_type):
    """pointer_type is not hashable in old python versions"""

    def __hash__(self):
        return hash((self.element_ty, self.address_space, self.const))

    @classmethod
    def from_pointer_type(cls, pointer_type):
        return cls(pointer_type.element_ty, pointer_type.address_space, pointer_type.const)


def get_nvshmem_home_path():
    if "NVSHMEM_HOME" in os.environ:
        return pathlib.Path(os.environ["NVSHMEM_HOME"])
    return pathlib.Path(nvidia.nvshmem.__path__[0])


def is_triton_version_supported():
    return version.parse(triton.__version__) >= version.parse("3.4.0")


def get_bc_path():
    """Get the path to the nvshmem_wrapper.bc file."""
    # First try to find the .bc file in the package directory (installed package)
    bc_file_name = "nvshmem_wrapper_full.bc"
    package_dir = pathlib.Path(__file__).parent
    bc_file = package_dir / bc_file_name

    if bc_file.exists():
        return str(bc_file)

    # Fallback to build directory for development
    # scikit-build-core uses build/{wheel_tag}/ structure
    build_dir = package_dir.parent / "build"

    if build_dir.exists():
        # Search for .bc file in all subdirectories of build/
        for bc_candidate in build_dir.rglob(bc_file_name):
            return str(bc_candidate)

    # Last resort: check environment variable
    if "NVSHMEM_WRAPPER_BC_PATH" in os.environ:
        return os.environ["NVSHMEM_WRAPPER_BC_PATH"]

    raise FileNotFoundError(
        f"{bc_file_name} not found. Please ensure the package is properly installed "
        "or set NVSHMEM_WRAPPER_BC_PATH environment variable."
    )


BC_PATH = get_bc_path()
LIB_NVSHMEM_PATH = str(get_nvshmem_home_path() / "lib" / "libnvshmem_device.bc")
LIB_NAME = "nvshmem_wrapper"
LIB_NVSHMEM_NAME = "libnvshmem"


NVSHMEM_EXTERN_LIBS = {
    # LIB_NVSHMEM_NAME: LIB_NVSHMEM_PATH,
    LIB_NAME: BC_PATH,
}


@builtin
def extern_elementwise(
    lib_name: str,
    lib_path: str,
    args: list,
    arg_type_symbol_dict: dict,
    is_pure: bool,
    _semantic=None,
):
    curr_version = version.parse(triton.__version__)
    if curr_version >= version.parse("3.5.0"):
        return extern_elementwise_v35(
            lib_name, lib_path, args, arg_type_symbol_dict, is_pure, _semantic=_semantic
        )
    else:
        return extern_elementwise_v34(
            lib_name, lib_path, args, arg_type_symbol_dict, is_pure, _semantic=_semantic
        )


@builtin
def extern_elementwise_v34(
    lib_name: str,
    lib_path: str,
    args: list,
    arg_type_symbol_dict: dict,
    is_pure: bool,
    _semantic=None,
):
    """
    NOTE: This function is modified from triton.language.core.extern_elementwise
          to remove the type checking and broadcast logic.
        Dispatch an elementwise function to a library
        :param lib_name: the name of the library
        :param lib_path: the path of the library
        :param args: the arguments of the function
        :param arg_type_symbol_dict: the type of the arguments
        :param is_pure: whether the function is pure
        :return: the return value of the function
    """
    dispatch_args = args.copy()
    ret_shape = None
    arg_types = []
    for i in builtins.range(len(dispatch_args)):
        dispatch_args[i] = _semantic.to_tensor(dispatch_args[i])
        arg_types.append(dispatch_args[i].dtype)
    func = _semantic.builder.create_extern_elementwise
    return dispatch(
        func, lib_name, lib_path, dispatch_args, arg_type_symbol_dict, ret_shape, is_pure, _semantic
    )


@builtin
def extern_elementwise_v35(
    lib_name: str,
    lib_path: str,
    args: list,
    arg_type_symbol_dict: dict,
    is_pure: bool,
    _semantic=None,
):
    """
    Dispatch an elementwise function to a library
    :param lib_name: the name of the library
    :param lib_path: the path of the library
    :param args: the arguments of the function
    :param arg_type_symbol_dict: the type of the arguments
    :param is_pure: whether the function is pure
    :return: the return value of the function
    """
    dispatch_args = args.copy()
    arg_types = []
    for i in builtins.range(len(dispatch_args)):
        dispatch_args[i] = _semantic.to_tensor(dispatch_args[i])
        arg_types.append(dispatch_args[i].dtype)

    arg_types = tuple(arg_types)
    ret_type = arg_type_symbol_dict[arg_types][1]

    func = _semantic.builder.create_extern_elementwise
    return dispatch(
        func, lib_name, lib_path, dispatch_args, arg_type_symbol_dict, ret_type, is_pure, _semantic
    )


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


@core.extern
def int_atomic_compare_swap(arg0, arg1, arg2, arg3, _semantic=None):
    arg0.dtype = hashable_pointer_type.from_pointer_type(arg0.dtype)
    # arg0 = tl.cast(arg0, hashable_pointer_type(tl.int32), **kwargs)

    return extern_elementwise(
        LIB_NAME,
        BC_PATH,
        [arg0, arg1, arg2, arg3],
        {
            (hashable_pointer_type(tl.int32), tl.int32, tl.int32, tl.int32): (
                "nvshmem_int_atomic_compare_swap",
                tl.int32,
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def int_atomic_swap(dest, value, pe, _semantic=None):
    dest.dtype = hashable_pointer_type.from_pointer_type(dest.dtype)
    return extern_elementwise(
        LIB_NAME,
        BC_PATH,
        [dest, value, pe],
        {
            (hashable_pointer_type(tl.int32), tl.int32, tl.int32): (
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
    return extern_elementwise(
        LIB_NAME,
        BC_PATH,
        [
            tl.cast(dest, hashable_pointer_type(tl.void), _semantic=_semantic),
            tl.cast(source, hashable_pointer_type(tl.void, const=True), _semantic=_semantic),
            tl.cast(nbytes, tl.uint64, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
        ],
        {
            (
                hashable_pointer_type(tl.void),
                hashable_pointer_type(tl.void, const=True),
                tl.uint64,
                tl.int32,
            ): (f"tt_{name}", tl.int32),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def putmem_nbi_block(dest, source, nbytes, pe, _semantic=None):
    return _putmem_impl(
        dest,
        source,
        nbytes,
        pe,
        core.constexpr("_block"),
        core.constexpr("_nbi"),
        _semantic=_semantic,
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
    return extern_elementwise(
        LIB_NAME,
        BC_PATH,
        [
            tl.cast(dest, hashable_pointer_type(tl.void), _semantic=_semantic),
            tl.cast(source, hashable_pointer_type(tl.void, const=True), _semantic=_semantic),
            tl.cast(nbytes, tl.uint64, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
        ],
        {
            (
                hashable_pointer_type(tl.void),
                hashable_pointer_type(tl.void, const=True),
                tl.uint64,
                tl.int32,
            ): (
                f"tt_nvshmem{'x' if SCOPE_SUFFIX.value else ''}_getmem{NBI.value}{SCOPE_SUFFIX.value}",
                tl.int32,
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def getmem_nbi_block(dest, source, nbytes, pe, _semantic=None):
    return _getmem_impl(
        dest,
        source,
        nbytes,
        pe,
        core.constexpr("_block"),
        core.constexpr("_nbi"),
        _semantic=_semantic,
    )


@core.extern
def quiet(_semantic=None):
    name = "nvshmem_quiet"
    return extern_elementwise(
        LIB_NAME,
        BC_PATH,
        [],
        {
            (): (f"tt_{name}", core.dtype("int32")),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def int_p(dest, value, pe, _semantic=None):
    dest.dtype = hashable_pointer_type.from_pointer_type(dest.dtype)
    return extern_elementwise(
        LIB_NAME,
        BC_PATH,
        [dest, value, pe],
        {
            (hashable_pointer_type(tl.int32), tl.int32, tl.int32): ("tt_nvshmem_int_p", tl.int32),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def int_g(src, pe, _semantic=None):
    src.dtype = hashable_pointer_type.from_pointer_type(src.dtype)
    return extern_elementwise(
        LIB_NAME,
        BC_PATH,
        [src, pe],
        {
            (hashable_pointer_type(tl.int32), tl.int32): ("nvshmem_int_g", tl.int32),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def __syncthreads(_semantic=None):
    return tl.tensor(_semantic.builder.create_barrier(), tl.void)
