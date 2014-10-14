// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_DEVICE_TYPE_H
#define EIGEN_CXX11_TENSOR_TENSOR_DEVICE_TYPE_H


namespace Eigen {

// Default device for the machine (typically a single cpu core)
struct DefaultDevice {
  EIGEN_STRONG_INLINE void* allocate(size_t num_bytes) const {
    return internal::aligned_malloc(num_bytes);
  }
  EIGEN_STRONG_INLINE void deallocate(void* buffer) const {
    internal::aligned_free(buffer);
  }
  EIGEN_STRONG_INLINE void memcpy(void* dst, const void* src, size_t n) const {
    ::memcpy(dst, src, n);
  }
  EIGEN_STRONG_INLINE void memset(void* buffer, int c, size_t n) const {
    ::memset(buffer, c, n);
  }

  EIGEN_STRONG_INLINE size_t numThreads() const {
    return 1;
  }
};


// Multiple cpu cores
// We should really use a thread pool here but first we need to find a portable thread pool library.
#ifdef EIGEN_USE_THREADS

typedef std::future<void> Future;

struct ThreadPoolDevice {
  ThreadPoolDevice(/*ThreadPool* pool, */size_t num_cores) : num_threads_(num_cores) { }

  EIGEN_STRONG_INLINE void* allocate(size_t num_bytes) const {
    return internal::aligned_malloc(num_bytes);
  }

  EIGEN_STRONG_INLINE void deallocate(void* buffer) const {
    internal::aligned_free(buffer);
  }

  EIGEN_STRONG_INLINE void memcpy(void* dst, const void* src, size_t n) const {
    ::memcpy(dst, src, n);
  }

  EIGEN_STRONG_INLINE void memset(void* buffer, int c, size_t n) const {
    ::memset(buffer, c, n);
  }

  EIGEN_STRONG_INLINE size_t numThreads() const {
    return num_threads_;
  }

  template <class Function, class... Args>
  EIGEN_STRONG_INLINE Future enqueue(Function&& f, Args&&... args) const {
    return std::async(std::launch::async, f, args...);
  }
  template <class Function, class... Args>
  EIGEN_STRONG_INLINE void enqueueNoFuture(Function&& f, Args&&... args) const {
    std::async(std::launch::async, f, args...);
  }

 private:
  // todo: NUMA, ...
  size_t num_threads_;
};
#endif


// GPU offloading
#ifdef EIGEN_USE_GPU
static cudaDeviceProp m_deviceProperties;
static bool m_devicePropInitialized = false;

static void initializeDeviceProp() {
  if (!m_devicePropInitialized) {
    assert(cudaGetDeviceProperties(&m_deviceProperties, 0) == cudaSuccess);
    m_devicePropInitialized = true;
  }
}

static inline int getNumCudaMultiProcessors() {
  initializeDeviceProp();
  return m_deviceProperties.multiProcessorCount;
}
static inline int maxCudaThreadsPerBlock() {
  initializeDeviceProp();
  return m_deviceProperties.maxThreadsPerBlock;
}
static inline int maxCudaThreadsPerMultiProcessor() {
  initializeDeviceProp();
  return m_deviceProperties.maxThreadsPerMultiProcessor;
}
static inline int sharedMemPerBlock() {
  initializeDeviceProp();
  return m_deviceProperties.sharedMemPerBlock;
}


struct GpuDevice {
  // The cudastream is not owned: the caller is responsible for its initialization and eventual destruction.
  GpuDevice(const cudaStream_t* stream) : stream_(stream) { eigen_assert(stream); }

  EIGEN_STRONG_INLINE const cudaStream_t& stream() const { return *stream_; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void* allocate(size_t num_bytes) const {
#ifndef __CUDA_ARCH__
    void* result;
    assert(cudaMalloc(&result, num_bytes) == cudaSuccess);
    assert(result != NULL);
    return result;
#else
    assert(false && "The default device should be used instead to generate kernel code");
    return NULL;
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void deallocate(void* buffer) const {
#ifndef __CUDA_ARCH__
    assert(buffer != NULL);
    assert(cudaFree(buffer) == cudaSuccess);
#else
    assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpy(void* dst, const void* src, size_t n) const {
#ifndef __CUDA_ARCH__
    assert(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, *stream_) == cudaSuccess);
#else
    assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memset(void* buffer, int c, size_t n) const {
#ifndef __CUDA_ARCH__
    assert(cudaMemsetAsync(buffer, c, n, *stream_) == cudaSuccess);
#else
    assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t numThreads() const {
    // FIXME
    return 32;
  }

 private:
  // TODO: multigpu.
  const cudaStream_t* stream_;
};
#endif

}  // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_DEVICE_TYPE_H
