# NVSHMEM-Triton submodule CMakeLists.txt
# This file is included from the parent odc CMakeLists.txt

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

# Function to find NVSHMEM_HOME
function(find_nvshmem_home)
    # 1. Check if NVSHMEM_HOME environment variable is set
    if(DEFINED ENV{NVSHMEM_HOME})
        set(NVSHMEM_HOME $ENV{NVSHMEM_HOME} PARENT_SCOPE)
        message(STATUS "Found NVSHMEM_HOME from environment variable: $ENV{NVSHMEM_HOME}")
        return()
    endif()

    # 2. Try to find from Python command
    execute_process(
        COMMAND python3 -c "import nvidia.nvshmem, pathlib; print(pathlib.Path(nvidia.nvshmem.__path__[0]))"
        OUTPUT_VARIABLE PYTHON_NVSHMEM_HOME
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(PYTHON_NVSHMEM_HOME)
        set(NVSHMEM_HOME ${PYTHON_NVSHMEM_HOME} PARENT_SCOPE)
        message(STATUS "Found NVSHMEM_HOME from Python nvidia-nvshmem-cu12: ${PYTHON_NVSHMEM_HOME}")
        return()
    endif()

    # 3. Fallback to ldconfig (Linux only)
    if(UNIX AND NOT APPLE)
        execute_process(
            COMMAND bash -c "ldconfig -p | grep 'libnvshmem_host' | awk '{print $NF}' | xargs dirname | head -n 1"
            OUTPUT_VARIABLE LDCONFIG_NVSHMEM_HOME
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        
        if(LDCONFIG_NVSHMEM_HOME)
            set(NVSHMEM_HOME ${LDCONFIG_NVSHMEM_HOME} PARENT_SCOPE)
            message(STATUS "Found NVSHMEM_HOME from ldconfig: ${LDCONFIG_NVSHMEM_HOME}")
            return()
        endif()
    endif()

    message(WARNING "NVSHMEM_HOME could not be determined.")
endfunction()

# Find NVSHMEM
find_nvshmem_home()

if(NOT NVSHMEM_HOME)
    message(FATAL_ERROR "NVSHMEM_HOME not found. Please set NVSHMEM_HOME environment variable or install nvidia-nvshmem-cu12 and nvshmem4py-cu12.")
endif()

# Find clang and LLVM tools
find_program(CLANG_EXECUTABLE NAMES clang++-21 clang++-20 clang++-19 clang++)
find_program(LLVM_DIS_EXECUTABLE NAMES llvm-dis-21 llvm-dis-20 llvm-dis-19 llvm-dis)
find_program(LLVM_AS_EXECUTABLE NAMES llvm-as-21 llvm-as-20 llvm-as-19 llvm-as)
find_program(LLVM_LINK_EXECUTABLE NAMES llvm-link-21 llvm-link-20 llvm-link-19 llvm-link)

if(NOT CLANG_EXECUTABLE)
    message(FATAL_ERROR "clang++ not found. Please install clang.")
endif()

if(NOT LLVM_DIS_EXECUTABLE)
    message(FATAL_ERROR "llvm-dis not found. Please install LLVM tools.")
endif()

if(NOT LLVM_AS_EXECUTABLE)
    message(FATAL_ERROR "llvm-as not found. Please install LLVM tools.")
endif()

if(NOT LLVM_LINK_EXECUTABLE)
    message(FATAL_ERROR "llvm-link not found. Please install LLVM tools.")
endif()

message(STATUS "Using clang: ${CLANG_EXECUTABLE}")
message(STATUS "Using llvm-dis: ${LLVM_DIS_EXECUTABLE}")
message(STATUS "Using llvm-as: ${LLVM_AS_EXECUTABLE}")
message(STATUS "Using llvm-link: ${LLVM_LINK_EXECUTABLE}")

# Set output directory
set(OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Custom target to build nvshmem_wrapper.bc
add_custom_command(
    OUTPUT ${OUTPUT_DIR}/nvshmem_wrapper.bc
    COMMAND ${CLANG_EXECUTABLE} --cuda-device-only -c -emit-llvm 
            ${CMAKE_CURRENT_SOURCE_DIR}/wrapper.cu 
            -I${NVSHMEM_HOME}/include 
            -o ${OUTPUT_DIR}/nvshmem_wrapper_temp.bc
    COMMAND ${LLVM_DIS_EXECUTABLE} ${OUTPUT_DIR}/nvshmem_wrapper_temp.bc 
            -o ${OUTPUT_DIR}/nvshmem_wrapper.ll
    COMMAND sed -i.bak -E "s/!\"nvvm-reflect-ftz\", i32 0}/!\"nvvm-reflect-ftz\", i32 1}/g" 
            ${OUTPUT_DIR}/nvshmem_wrapper.ll
    COMMAND ${LLVM_AS_EXECUTABLE} ${OUTPUT_DIR}/nvshmem_wrapper.ll 
            -o ${OUTPUT_DIR}/nvshmem_wrapper.bc
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/wrapper.cu
    COMMENT "Building nvshmem_wrapper.bc"
    VERBATIM
)

# Custom command to link nvshmem_wrapper.bc with libnvshmem_device.bc
add_custom_command(
    OUTPUT ${OUTPUT_DIR}/nvshmem_wrapper_full.bc
    COMMAND ${LLVM_LINK_EXECUTABLE}
            ${OUTPUT_DIR}/nvshmem_wrapper.bc
            ${NVSHMEM_HOME}/lib/libnvshmem_device.bc
            -o ${OUTPUT_DIR}/nvshmem_wrapper_full.bc
    DEPENDS ${OUTPUT_DIR}/nvshmem_wrapper.bc
    COMMENT "Linking nvshmem_wrapper.bc with libnvshmem_device.bc to create nvshmem_wrapper_full.bc"
    VERBATIM
)

# Create a target for the bitcode file
add_custom_target(nvshmem_wrapper_bc ALL
    DEPENDS ${OUTPUT_DIR}/nvshmem_wrapper.bc
    DEPENDS ${OUTPUT_DIR}/nvshmem_wrapper_full.bc
)

# Install the bitcode files (for deployment/development builds)
# Note: These are installed to a standard location, but in practice the files
# are copied to odc/primitives/ by the parent CMakeLists.txt for development
install(FILES ${OUTPUT_DIR}/nvshmem_wrapper.bc
        DESTINATION odc/primitives/
        COMPONENT Runtime)
install(FILES ${OUTPUT_DIR}/nvshmem_wrapper_full.bc
        DESTINATION odc/primitives/
        COMPONENT Runtime)

# Clean up intermediate files
add_custom_command(
    TARGET nvshmem_wrapper_bc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove -f 
            ${OUTPUT_DIR}/nvshmem_wrapper_temp.bc
            ${OUTPUT_DIR}/nvshmem_wrapper.ll
            ${OUTPUT_DIR}/nvshmem_wrapper.ll.bak
    COMMENT "Cleaning up intermediate files"
)

# Print NVSHMEM-Triton configuration summary
message(STATUS "NVSHMEM-Triton Configuration:")
message(STATUS "  - NVSHMEM_HOME: ${NVSHMEM_HOME}")
message(STATUS "  - Output Directory: ${OUTPUT_DIR}")
message(STATUS "  - Clang: ${CLANG_EXECUTABLE}")
message(STATUS "  - LLVM Tools: ${LLVM_DIS_EXECUTABLE}, ${LLVM_AS_EXECUTABLE}, ${LLVM_LINK_EXECUTABLE}")
message(STATUS "  - Install Destination: odc/primitives/")


# Copy bitcode files to odc/primitives/ for local development
# This allows the module to find them without running 'pip install'
add_custom_command(TARGET nvshmem_wrapper_bc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/odc/primitives
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${OUTPUT_DIR}/nvshmem_wrapper.bc
        ${CMAKE_SOURCE_DIR}/odc/primitives/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${OUTPUT_DIR}/nvshmem_wrapper_full.bc
        ${CMAKE_SOURCE_DIR}/odc/primitives/
    COMMENT "Copying nvshmem bitcode files to odc/primitives/ for development and packaging"
)
