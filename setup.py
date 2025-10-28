"""
Setup script for ODC package using torch cpp_extension for CUDA extensions.
"""

from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Define the tensor_ipc CUDA extension
# Install it as odc.primitives.tensor_ipc so it can be found by the import code
tensor_ipc_extension = CUDAExtension(
    name="odc.primitives.tensor_ipc",
    sources=[
        "csrc/tensor_ipc/binding.cpp",
        "csrc/tensor_ipc/tensor_ipc.cu",
    ],
    libraries=["cuda"],
)


# Note: nvshmem bitcode files are built using CMake, not during pip install
# The BuildNVSHMEMBitcode class has been removed to avoid duplicate builds

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="odc",
    use_scm_version={
        "write_to": "odc/_version.py",
        "fallback_version": "0.1.0",
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="On-Demand Communication",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sail-sg/odc",
    packages=find_packages(),
    ext_modules=[tensor_ipc_extension],
    cmdclass={
        "build_ext": BuildExtension,
    },
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "triton>=3.4.0",
        "nvidia-nvshmem-cu12",
        "nvshmem4py-cu12",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
