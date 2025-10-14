#include <torch/extension.h>
#include <cuda_runtime.h>


#define CUDA_CHECK(cmd) do {                                 \
  cudaError_t e = (cmd);                              \
  if (e != cudaSuccess) {                             \
      fprintf(stderr, "%s:%d CUDA error %d: %s\n",    \
              __FILE__, __LINE__, (int)e,             \
              cudaGetErrorString(e));                 \
      exit(1);                                        \
  }                                                   \
} while(0)

#define CU_CHECK(cmd) do {                                 \
  CUresult e = (cmd);                              \
  if (e != CUDA_SUCCESS) {                             \
      fprintf(stderr, "%s:%d CUDA Driver error %d\n",    \
              __FILE__, __LINE__, (int)e);                 \
      exit(1);                                        \
  }                                                   \
} while(0)


// __global__ void persistent_kernel(int *stop_flag) {
//     while (true) {
//         // Check stop flag in global memory
//         if (atomicAdd(stop_flag, 0) != 0) {
//             break;
//         }

//         // Do some fake work (or real work here)
//         __nanosleep(1000000); // ~1ms sleep to yield SM

//         // Note: Without nanosleep, this loop would hog the SM fully
//     }
// }

// void launch_persistent_kernel(at::Tensor stop_flag) {
//     persistent_kernel<<<1, 1>>>(stop_flag.data_ptr<int>());
// }

py::bytes get_ipc_handle(at::Tensor tensor) {
    void *ptr = (void*)tensor.data_ptr();

    // std::vector<float> data(tensor.numel());
    // CUDA_CHECK(cudaMemcpy(data.data(), ptr, tensor.numel() * sizeof(float), cudaMemcpyDeviceToHost));
    // printf("original: ");
    // for (int i = 0; i < tensor.numel(); i++) {
    //     printf("%f ", data[i]);
    // }
    // printf("\n");
    // // printf("ptr: %p\n", ptr);

    // cudaPointerAttributes attr;
    // CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));

    CUdeviceptr base;
    size_t size;
    CU_CHECK(cuMemGetAddressRange(&base, &size, (CUdeviceptr)ptr));
    // printf("ptr: %p, base ptr: %p, device: %d\n", ptr, base, attr.device);

    size_t pointer_offset = (size_t)ptr - (size_t)base;
    

    cudaIpcMemHandle_t mem_handle;
    CUDA_CHECK(cudaIpcGetMemHandle(&mem_handle, (void*)base));
    std::string s;
    s.resize(sizeof(mem_handle) + sizeof(pointer_offset));
    memcpy(s.data(), &mem_handle, sizeof(mem_handle));
    memcpy(s.data() + sizeof(mem_handle), &pointer_offset, sizeof(pointer_offset));
    return py::bytes(s);
}

at::Tensor reconstruct_tensor(py::bytes handle, std::vector<int64_t> shape, torch::ScalarType dtype) {
  std::string s = handle.cast<std::string>();
  cudaIpcMemHandle_t mem_handle;
  size_t pointer_offset;
  memcpy(&mem_handle, s.data(), sizeof(mem_handle));
  memcpy(&pointer_offset, s.data() + sizeof(mem_handle), sizeof(pointer_offset));
  uint8_t *ptr = nullptr;
  CUDA_CHECK(cudaIpcOpenMemHandle((void**)&ptr, mem_handle, cudaIpcMemLazyEnablePeerAccess));
  ptr += pointer_offset;

  // std::vector<float> data(shape[0]);
  // CUDA_CHECK(cudaMemcpy(data.data(), ptr, shape[0] * sizeof(float), cudaMemcpyDeviceToHost));
  // printf("reconstructed: ");
  // for (int i = 0; i < shape[0]; i++) {
  //   printf("%f ", data[i]);
  // }
  // printf("\n");

  return at::from_blob(ptr, shape, at::TensorOptions().dtype(dtype).device(at::kCUDA));
}