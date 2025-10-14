cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(odc LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/3rdparty")

# Add pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
)
FetchContent_MakeAvailable(pybind11)

# Find required packages
find_package(CUDA REQUIRED)

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
find_package(Torch REQUIRED)

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
    message(STATUS "  - CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "  - PyTorch Found: ${Torch_FOUND}")
if(Torch_FOUND)
    message(STATUS "  - PyTorch Version: ${Torch_VERSION}")
endif()
message(STATUS "  - Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  - Install Prefix: ${CMAKE_INSTALL_PREFIX}")
if(DEFINED SKBUILD_PLATLIB_DIR)
    message(STATUS "  - Python Install Dir: ${SKBUILD_PLATLIB_DIR}")
endif()

# Add subdirectories for each submodule
# nvshmem_triton is handled as a single Python file in odc/primitives/
# The bitcode files are built in csrc/nvshmem_triton/
add_subdirectory(csrc/nvshmem_triton)
add_subdirectory(csrc/tensor_ipc)

# Install Python package structure
if(DEFINED SKBUILD_PLATLIB_DIR)
    # Install all Python files from odc directory
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
            DESTINATION ${SKBUILD_PLATLIB_DIR}/odc/
            FILES_MATCHING 
            PATTERN "*.py"
            PATTERN "__pycache__" EXCLUDE
            PATTERN "*.pyc" EXCLUDE
            PATTERN "build" EXCLUDE
            PATTERN "*.egg-info" EXCLUDE)
    
    # Install specific directories that should be included
    # Note: nvshmem_triton bitcode files are installed by csrc/nvshmem_triton/CMakeLists.txt
    
    # Install tensor_ipc if it exists
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/csrc/tensor_ipc")
        install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tensor_ipc/
                DESTINATION ${SKBUILD_PLATLIB_DIR}/odc/tensor_ipc/
                FILES_MATCHING 
                PATTERN "*.py"
                PATTERN "__pycache__" EXCLUDE
                PATTERN "*.pyc" EXCLUDE)
    endif()
else()
    # Fallback for manual cmake builds
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
            DESTINATION odc/
            FILES_MATCHING 
            PATTERN "*.py"
            PATTERN "__pycache__" EXCLUDE
            PATTERN "*.pyc" EXCLUDE
            PATTERN "build" EXCLUDE
            PATTERN "*.egg-info" EXCLUDE)
endif()
