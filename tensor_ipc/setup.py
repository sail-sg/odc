from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='tensor_ipc',
    ext_modules=[
        CUDAExtension(
            name='tensor_ipc',
            sources=['binding.cpp', 'tensor_ipc.cu'],
            libraries=['cuda'],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
