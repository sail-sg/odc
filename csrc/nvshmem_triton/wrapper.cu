#include <type_traits>
#include <cuda_runtime.h>


extern "C" {

static_assert(std::is_same_v<int, int32_t>, "int must be int32_t");
static_assert(std::is_same_v<size_t, uint64_t>, "size_t must be uint64_t");

__device__ void nvshmem_quiet();
__device__ int tt_nvshmem_quiet() { nvshmem_quiet(); return 0; }

__device__ int nvshmem_int_atomic_compare_swap(int *dest, int cond, int value, int pe);
__device__ int tt_nvshmem_int_atomic_compare_swap(int *dest, int cond, int value, int pe) {
  return nvshmem_int_atomic_compare_swap(dest, cond, value, pe);
}

__device__ int nvshmemx_putmem_nbi_block(void *dest,
  const void *source,
  size_t nelems, int pe);
__device__ int tt_nvshmemx_putmem_nbi_block(void *dest,
  const void *source,
  size_t nelems, int pe) {
  nvshmemx_putmem_nbi_block(dest, source, nelems, pe);
  return 0;
}

__device__ void nvshmemx_getmem_nbi_block(void *dest,
  const void *source,
  size_t bytes, int pe);
__device__ int tt_nvshmemx_getmem_nbi_block(void *dest,
  const void *source,
  size_t bytes, int pe) {
  nvshmemx_getmem_nbi_block(dest, source, bytes, pe);
  return 0;
}

__device__ void nvshmem_int_p(int *destination, int value, int peer);
__device__ int tt_nvshmem_int_p(int *destination, int value, int peer) {
  nvshmem_int_p(destination, value, peer);
  return 0;
}

}
