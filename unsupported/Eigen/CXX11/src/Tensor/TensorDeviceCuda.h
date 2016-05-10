// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#if defined(EIGEN_USE_GPU) && !defined(EIGEN_CXX11_TENSOR_TENSOR_DEVICE_CUDA_H)
#define EIGEN_CXX11_TENSOR_TENSOR_DEVICE_CUDA_H

namespace Eigen {

// This defines an interface that GPUDevice can take to use
// CUDA streams underneath.
class StreamInterface {
 public:
  virtual ~StreamInterface() {}

  virtual const cudaStream_t& stream() const = 0;
  virtual const cudaDeviceProp& deviceProperties() const = 0;

  // Allocate memory on the actual device where the computation will run
  virtual void* allocate(size_t num_bytes) const = 0;
  virtual void deallocate(void* buffer) const = 0;

  // Return a scratchpad buffer of size 1k
  virtual void* scratchpad() const = 0;
};

static cudaDeviceProp* m_deviceProperties;
static bool m_devicePropInitialized = false;

static void initializeDeviceProp() {
  if (!m_devicePropInitialized) {
    if (!m_devicePropInitialized) {
      int num_devices;
      cudaError_t status = cudaGetDeviceCount(&num_devices);
      if (status != cudaSuccess) {
        std::cerr << "Failed to get the number of CUDA devices: "
                  << cudaGetErrorString(status)
                  << std::endl;
        assert(status == cudaSuccess);
      }
      m_deviceProperties = new cudaDeviceProp[num_devices];
      for (int i = 0; i < num_devices; ++i) {
        status = cudaGetDeviceProperties(&m_deviceProperties[i], i);
        if (status != cudaSuccess) {
          std::cerr << "Failed to initialize CUDA device #"
                    << i
                    << ": "
                    << cudaGetErrorString(status)
                    << std::endl;
          assert(status == cudaSuccess);
        }
      }
      m_devicePropInitialized = true;
    }
  }
}

static const cudaStream_t default_stream = cudaStreamDefault;

class CudaStreamDevice : public StreamInterface {
 public:
  // Use the default stream on the current device
  CudaStreamDevice() : stream_(&default_stream), scratch_(NULL) {
    cudaGetDevice(&device_);
    initializeDeviceProp();
  }
  // Use the default stream on the specified device
  CudaStreamDevice(int device) : stream_(&default_stream), device_(device), scratch_(NULL) {
    initializeDeviceProp();
  }
  // Use the specified stream. Note that it's the
  // caller responsibility to ensure that the stream can run on
  // the specified device. If no device is specified the code
  // assumes that the stream is associated to the current gpu device.
  CudaStreamDevice(const cudaStream_t* stream, int device = -1)
      : stream_(stream), device_(device), scratch_(NULL) {
    if (device < 0) {
      cudaGetDevice(&device_);
    } else {
      int num_devices;
      cudaError_t err = cudaGetDeviceCount(&num_devices);
      EIGEN_UNUSED_VARIABLE(err)
      assert(err == cudaSuccess);
      assert(device < num_devices);
      device_ = device;
    }
    initializeDeviceProp();
  }

  virtual ~CudaStreamDevice() {
    if (scratch_) {
      deallocate(scratch_);
    }
  }

  const cudaStream_t& stream() const { return *stream_; }
  const cudaDeviceProp& deviceProperties() const {
    return m_deviceProperties[device_];
  }
  virtual void* allocate(size_t num_bytes) const {
    cudaError_t err = cudaSetDevice(device_);
    EIGEN_UNUSED_VARIABLE(err)
    assert(err == cudaSuccess);
    void* result;
    err = cudaMalloc(&result, num_bytes);
    assert(err == cudaSuccess);
    assert(result != NULL);
    return result;
  }
  virtual void deallocate(void* buffer) const {
    cudaError_t err = cudaSetDevice(device_);
    EIGEN_UNUSED_VARIABLE(err)
    assert(err == cudaSuccess);
    assert(buffer != NULL);
    err = cudaFree(buffer);
    assert(err == cudaSuccess);
  }

  virtual void* scratchpad() const {
    if (scratch_ == NULL) {
      scratch_ = allocate(1024);
    }
    return scratch_;
  }

 private:
  const cudaStream_t* stream_;
  int device_;
  mutable void* scratch_;
};

struct GpuDevice {
  // The StreamInterface is not owned: the caller is
  // responsible for its initialization and eventual destruction.
  explicit GpuDevice(const StreamInterface* stream) : stream_(stream), max_blocks_(INT_MAX) {
    eigen_assert(stream);
  }
  explicit GpuDevice(const StreamInterface* stream, int num_blocks) : stream_(stream), max_blocks_(num_blocks) {
    eigen_assert(stream);
  }
  // TODO(bsteiner): This is an internal API, we should not expose it.
  EIGEN_STRONG_INLINE const cudaStream_t& stream() const {
    return stream_->stream();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void* allocate(size_t num_bytes) const {
#ifndef __CUDA_ARCH__
    return stream_->allocate(num_bytes);
#else
    eigen_assert(false && "The default device should be used instead to generate kernel code");
    return NULL;
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void deallocate(void* buffer) const {
#ifndef __CUDA_ARCH__
    stream_->deallocate(buffer);
#else
    eigen_assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void* scratchpad() const {
#ifndef __CUDA_ARCH__
    return stream_->scratchpad();
#else
    eigen_assert(false && "The default device should be used instead to generate kernel code");
#endif
    return NULL;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpy(void* dst, const void* src, size_t n) const {
#ifndef __CUDA_ARCH__
    cudaError_t err = cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice,
                                      stream_->stream());
    EIGEN_UNUSED_VARIABLE(err)
    assert(err == cudaSuccess);
#else
    eigen_assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpyHostToDevice(void* dst, const void* src, size_t n) const {
#ifndef __CUDA_ARCH__
    cudaError_t err =
        cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, stream_->stream());
    EIGEN_UNUSED_VARIABLE(err)
    assert(err == cudaSuccess);
#else
    eigen_assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpyDeviceToHost(void* dst, const void* src, size_t n) const {
#ifndef __CUDA_ARCH__
    cudaError_t err =
        cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, stream_->stream());
    EIGEN_UNUSED_VARIABLE(err)
    assert(err == cudaSuccess);
#else
    eigen_assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memset(void* buffer, int c, size_t n) const {
#ifndef __CUDA_ARCH__
    cudaError_t err = cudaMemsetAsync(buffer, c, n, stream_->stream());
    EIGEN_UNUSED_VARIABLE(err)
    assert(err == cudaSuccess);
#else
    eigen_assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t numThreads() const {
    // FIXME
    return 32;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t firstLevelCacheSize() const {
    // FIXME
    return 48*1024;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t lastLevelCacheSize() const {
    // We won't try to take advantage of the l2 cache for the time being, and
    // there is no l3 cache on cuda devices.
    return firstLevelCacheSize();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void synchronize() const {
#if defined(__CUDACC__) && !defined(__CUDA_ARCH__)
    cudaError_t err = cudaStreamSynchronize(stream_->stream());
    if (err != cudaSuccess) {
      std::cerr << "Error detected in CUDA stream: "
                << cudaGetErrorString(err)
                << std::endl;
      assert(err == cudaSuccess);
    }
#else
    assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int getNumCudaMultiProcessors() const {
#ifndef __CUDA_ARCH__
    return stream_->deviceProperties().multiProcessorCount;
#else
    eigen_assert(false && "The default device should be used instead to generate kernel code");
    return 0;
#endif
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int maxCudaThreadsPerBlock() const {
#ifndef __CUDA_ARCH__
    return stream_->deviceProperties().maxThreadsPerBlock;
#else
    eigen_assert(false && "The default device should be used instead to generate kernel code");
    return 0;
#endif
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int maxCudaThreadsPerMultiProcessor() const {
#ifndef __CUDA_ARCH__
    return stream_->deviceProperties().maxThreadsPerMultiProcessor;
#else
    eigen_assert(false && "The default device should be used instead to generate kernel code");
    return 0;
#endif
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int sharedMemPerBlock() const {
#ifndef __CUDA_ARCH__
    return stream_->deviceProperties().sharedMemPerBlock;
#else
    eigen_assert(false && "The default device should be used instead to generate kernel code");
    return 0;
#endif
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int majorDeviceVersion() const {
#ifndef __CUDA_ARCH__
    return stream_->deviceProperties().major;
#else
    eigen_assert(false && "The default device should be used instead to generate kernel code");
    return 0;
#endif
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int minorDeviceVersion() const {
#ifndef __CUDA_ARCH__
    return stream_->deviceProperties().minor;
#else
    eigen_assert(false && "The default device should be used instead to generate kernel code");
    return 0;
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int maxBlocks() const {
    return max_blocks_;
  }

  // This function checks if the CUDA runtime recorded an error for the
  // underlying stream device.
  inline bool ok() const {
#ifdef __CUDACC__
    cudaError_t error = cudaStreamQuery(stream_->stream());
    return (error == cudaSuccess) || (error == cudaErrorNotReady);
#else
    return false;
#endif
  }

 private:
  const StreamInterface* stream_;
  int max_blocks_;
};

#define LAUNCH_CUDA_KERNEL(kernel, gridsize, blocksize, sharedmem, device, ...)             \
  (kernel) <<< (gridsize), (blocksize), (sharedmem), (device).stream() >>> (__VA_ARGS__);   \
  assert(cudaGetLastError() == cudaSuccess);


// FIXME: Should be device and kernel specific.
#ifdef __CUDACC__
static EIGEN_DEVICE_FUNC inline void setCudaSharedMemConfig(cudaSharedMemConfig config) {
#ifndef __CUDA_ARCH__
  cudaError_t status = cudaDeviceSetSharedMemConfig(config);
  EIGEN_UNUSED_VARIABLE(status)
  assert(status == cudaSuccess);
#else
  EIGEN_UNUSED_VARIABLE(config)
#endif
}
#endif

}  // end namespace Eigen

#endif  // EIGEN_CXX11_TENSOR_TENSOR_DEVICE_CUDA_H
