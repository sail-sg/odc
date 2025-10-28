cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(odc LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
enable_language(CUDA)

find_program(PYTHON_EXECUTABLE NAMES python3 python)
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} "-c"
          "from __future__ import print_function; import os; import torch;
print(os.path.dirname(torch.__file__),end='');"
  RESULT_VARIABLE _PYTHON_SUCCESS
  OUTPUT_VARIABLE TORCH_DIR)
if(NOT _PYTHON_SUCCESS MATCHES 0)
  message("PY:${PYTHONPATH}")
  message(FATAL_ERROR "Torch config Error.")
endif()
message(STATUS "TORCH_DIR: ${TORCH_DIR}")
list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})

# Find NVSHMEM
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} "-c" "import nvidia.nvshmem, pathlib; print(pathlib.Path(nvidia.nvshmem.__path__[0]))"
  OUTPUT_VARIABLE NVSHMEM_HOME
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NVSHMEM_HOME)
  set(ENV{NVSHMEM_HOME} ${NVSHMEM_HOME})
  message(STATUS "Found NVSHMEM_HOME: ${NVSHMEM_HOME}")
endif()

# Set output directory for all submodules
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Print configuration summary
message(STATUS "ODC Package Configuration:")
message(STATUS "  - C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  - CUDA Found: ${CUDA_FOUND}")
if(CUDA_FOUND)
    message(STATUS "  - CUDA Version: ${CUDA_VERSION}")
endif()
message(STATUS "  - Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  - Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "Note: tensor_ipc is built using torch cpp_extension via setuptools during pip install.")
message(STATUS "      Only nvshmem bitcode files are built via CMake.")

# Add subdirectories for each submodule
# nvshmem_triton: builds bitcode files needed for Triton kernel extern libraries
# Note: tensor_ipc is built using torch cpp_extension via setuptools (see setup.py)
add_subdirectory(csrc/nvshmem_triton)
