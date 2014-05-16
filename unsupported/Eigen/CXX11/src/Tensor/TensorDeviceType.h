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
};


// Multiple cpu cores
// We should really use a thread pool here but first we need to find a portable thread pool library.
#ifdef EIGEN_USE_THREADS
struct ThreadPoolDevice {
  ThreadPoolDevice(/*ThreadPool* pool, */size_t num_cores) : /*pool_(pool), */num_threads_(num_cores) { }
  size_t numThreads() const { return num_threads_; }
  /*ThreadPool* threadPool() const { return pool_; }*/

 private:
  // todo: NUMA, ...
  size_t num_threads_;
  /*ThreadPool* pool_;*/
};
#endif


// GPU offloading
#ifdef EIGEN_USE_GPU
struct GpuDevice {
  // todo: support for multiple gpu;
  GpuDevice() {
    cudaStreamCreate(&stream_);
  }
  ~GpuDevice() {
    cudaStreamDestroy(stream_);
  }
  const cudaStream_t& stream() const { return stream_; }

 private:
  cudaStream_t stream_;
};
#endif

}  // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_DEVICE_TYPE_H
