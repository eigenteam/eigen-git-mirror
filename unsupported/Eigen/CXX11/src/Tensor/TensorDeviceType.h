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
};


// Multiple cpu cores
// We should really use a thread pool here but first we need to find a portable thread pool library.
#ifdef EIGEN_USE_THREADS
struct ThreadPoolDevice {
 ThreadPoolDevice(/*ThreadPool* pool, */size_t num_cores) : /*pool_(pool), */num_threads_(num_cores) { }
  size_t numThreads() const { return num_threads_; }

  EIGEN_STRONG_INLINE void* allocate(size_t num_bytes) const {
    return internal::aligned_malloc(num_bytes);
  }
  EIGEN_STRONG_INLINE void deallocate(void* buffer) const {
    internal::aligned_free(buffer);
  }

 private:
  // todo: NUMA, ...
  size_t num_threads_;
};
#endif


// GPU offloading
#ifdef EIGEN_USE_GPU
struct GpuDevice {
  // The cudastream is not owned: the caller is responsible for its initialization and eventual destruction.
  GpuDevice(const cudaStream_t* stream) : stream_(stream) { eigen_assert(stream); }

  EIGEN_STRONG_INLINE const cudaStream_t& stream() const { return *stream_; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void* allocate(size_t num_bytes) const {
    void* result;
    cudaMalloc(&result, num_bytes);
    return result;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void deallocate(void* buffer) const {
    cudaFree(buffer);
  }

 private:
  // TODO: multigpu.
  const cudaStream_t* stream_;
};
#endif

}  // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_DEVICE_TYPE_H
