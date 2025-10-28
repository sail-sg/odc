# Tensor IPC submodule CMakeLists.txt
# This file builds the tensor_ipc shared object for PyTorch

# Set C++ standard (inherited from parent, but ensure it's set)
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# Find required packages (CUDA should already be found by parent)
if(NOT CUDA_FOUND)
    # find_package(CUDA REQUIRED)
    enable_language(CUDA)
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
# Find PyTorch
find_package(Torch REQUIRED)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

# Add the Python module
pybind11_add_module(tensor_ipc
    binding.cpp
    tensor_ipc.cu
)

# Set properties for the library
set_target_properties(tensor_ipc PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

message(STATUS "TORCH_DIR: ${TORCH_DIR}")
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_DIR}/lib")
message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")

# Link libraries
target_link_libraries(tensor_ipc PRIVATE
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
    # ${CUDA_LIBRARIES}
    cuda
)

# Include directories
target_include_directories(tensor_ipc PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

# Compile definitions
target_compile_definitions(tensor_ipc PRIVATE
    ${TORCH_DEFINITIONS}
)

# Install the shared library
if(DEFINED SKBUILD_PLATLIB_DIR)
    install(TARGETS tensor_ipc
            DESTINATION ${SKBUILD_PLATLIB_DIR}/odc/tensor_ipc/
            COMPONENT Runtime)
else()
    # Fallback for manual cmake builds
    install(TARGETS tensor_ipc
            DESTINATION odc/tensor_ipc/
            COMPONENT Runtime)
endif()

# Print Tensor IPC configuration summary
message(STATUS "Tensor IPC Configuration:")
message(STATUS "  - PyTorch Found: ${Torch_FOUND}")
message(STATUS "  - PyTorch Version: ${Torch_VERSION}")
message(STATUS "  - CUDA Found: ${CUDA_FOUND}")
if(DEFINED SKBUILD_PLATLIB_DIR)
    message(STATUS "  - Install Destination: ${SKBUILD_PLATLIB_DIR}/odc/tensor_ipc/")
else()
    message(STATUS "  - Install Destination: odc/tensor_ipc/")
endif()
