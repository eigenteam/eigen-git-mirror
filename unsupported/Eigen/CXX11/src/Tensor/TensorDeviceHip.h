// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#if defined(EIGEN_USE_GPU) && !defined(EIGEN_CXX11_TENSOR_TENSOR_DEVICE_HIP_H)
#define EIGEN_CXX11_TENSOR_TENSOR_DEVICE_HIP_H

#if defined(EIGEN_HIPCC)
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#endif
#include <unistd.h> //for sleep function

namespace Eigen {

static const int kHipScratchSize = 1024;

// This defines an interface that GPUDevice can take to use
// HIP streams underneath.
class StreamInterface {
 public:
  virtual ~StreamInterface() {}

  virtual const hipStream_t& stream() const = 0;
  virtual const hipDeviceProp_t& deviceProperties() const = 0;

  // Allocate memory on the actual device where the computation will run
  virtual void* allocate(size_t num_bytes) const = 0;
  virtual void deallocate(void* buffer) const = 0;

  // Return a scratchpad buffer of size 1k
  virtual void* scratchpad() const = 0;

  // Return a semaphore. The semaphore is initially initialized to 0, and
  // each kernel using it is responsible for resetting to 0 upon completion
  // to maintain the invariant that the semaphore is always equal to 0 upon
  // each kernel start.
  virtual unsigned int* semaphore() const = 0;
};

static hipDeviceProp_t* m_deviceProperties;
static bool m_devicePropInitialized = false;

static void initializeDeviceProp() {
  if (!m_devicePropInitialized) {
    // Attempts to ensure proper behavior in the case of multiple threads
    // calling this function simultaneously. This would be trivial to
    // implement if we could use std::mutex, but unfortunately mutex don't
    // compile with nvcc, so we resort to atomics and thread fences instead.
    // Note that if the caller uses a compiler that doesn't support c++11 we
    // can't ensure that the initialization is thread safe.
#if 0 && __cplusplus >= 201103L
    static std::atomic<bool> first(true);
    if (first.exchange(false)) {
#else
    static bool first = true;
    if (first) {
      first = false;
#endif
      // We're the first thread to reach this point.
      int num_devices;
      hipError_t status = hipGetDeviceCount(&num_devices);
      if (status != hipSuccess) {
        std::cerr << "Failed to get the number of HIP devices: "
                  << hipGetErrorString(status)
                  << std::endl;
        assert(status == hipSuccess);
      }
      m_deviceProperties = new hipDeviceProp_t[num_devices];
      for (int i = 0; i < num_devices; ++i) {
        status = hipGetDeviceProperties(&m_deviceProperties[i], i);
        if (status != hipSuccess) {
          std::cerr << "Failed to initialize HIP device #"
                    << i
                    << ": "
                    << hipGetErrorString(status)
                    << std::endl;
          assert(status == hipSuccess);
        }
      }

#if 0 && __cplusplus >= 201103L
      std::atomic_thread_fence(std::memory_order_release);
#endif
      m_devicePropInitialized = true;
    } else {
      // Wait for the other thread to inititialize the properties.
      while (!m_devicePropInitialized) {
#if 0 && __cplusplus >= 201103L
        std::atomic_thread_fence(std::memory_order_acquire);
#endif
        sleep(1);
      }
    }
  }
}

static const hipStream_t default_stream = 0x00;//TODO: Use hipStreamDefault instead of 0x00;

class HipStreamDevice : public StreamInterface {
 public:
  // Use the default stream on the current device
  HipStreamDevice() : stream_(&default_stream), scratch_(NULL), semaphore_(NULL) {
    hipGetDevice(&device_);
    initializeDeviceProp();
  }
  // Use the default stream on the specified device
  HipStreamDevice(int device) : stream_(&default_stream), device_(device), scratch_(NULL), semaphore_(NULL) {
    initializeDeviceProp();
  }
  // Use the specified stream. Note that it's the
  // caller responsibility to ensure that the stream can run on
  // the specified device. If no device is specified the code
  // assumes that the stream is associated to the current gpu device.
  HipStreamDevice(const hipStream_t* stream, int device = -1)
      : stream_(stream), device_(device), scratch_(NULL), semaphore_(NULL) {
    if (device < 0) {
      hipGetDevice(&device_);
    } else {
      int num_devices;
      hipError_t err = hipGetDeviceCount(&num_devices);
      EIGEN_UNUSED_VARIABLE(err)
      assert(err == hipSuccess);
      assert(device < num_devices);
      device_ = device;
    }
    initializeDeviceProp();
  }

  virtual ~HipStreamDevice() {
    if (scratch_) {
      deallocate(scratch_);
    }
  }

  const hipStream_t& stream() const { return *stream_; }
  const hipDeviceProp_t& deviceProperties() const {
    return m_deviceProperties[device_];
  }
  virtual void* allocate(size_t num_bytes) const {
    hipError_t err = hipSetDevice(device_);
    EIGEN_UNUSED_VARIABLE(err)
    assert(err == hipSuccess);
    void* result;
    err = hipMalloc(&result, num_bytes);
    assert(err == hipSuccess);
    assert(result != NULL);
    return result;
  }
  virtual void deallocate(void* buffer) const {
    hipError_t err = hipSetDevice(device_);
    EIGEN_UNUSED_VARIABLE(err)
    assert(err == hipSuccess);
    assert(buffer != NULL);
    err = hipFree(buffer);
    assert(err == hipSuccess);
  }

  virtual void* scratchpad() const {
    if (scratch_ == NULL) {
      scratch_ = allocate(kHipScratchSize + sizeof(unsigned int));
    }
    return scratch_;
  }

  virtual unsigned int* semaphore() const {
    if (semaphore_ == NULL) {
      char* scratch = static_cast<char*>(scratchpad()) + kHipScratchSize;
      semaphore_ = reinterpret_cast<unsigned int*>(scratch);
      //hipError_t err = hipMemsetAsync(semaphore_, 0, sizeof(unsigned int), *stream_);
      hipError_t err = hipMemset(semaphore_, 0, sizeof(unsigned int));
      EIGEN_UNUSED_VARIABLE(err)
      assert(err == hipSuccess);
    }
    return semaphore_;
  }

 private:
  const hipStream_t* stream_;
  int device_;
  mutable void* scratch_;
  mutable unsigned int* semaphore_;
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
  EIGEN_STRONG_INLINE const hipStream_t& stream() const {
    return stream_->stream();
  }

  EIGEN_STRONG_INLINE void* allocate(size_t num_bytes) const {
    return stream_->allocate(num_bytes);
  }

  EIGEN_STRONG_INLINE void deallocate(void* buffer) const {
    stream_->deallocate(buffer);
  }

  EIGEN_STRONG_INLINE void* scratchpad() const {
    return stream_->scratchpad();
  }

  EIGEN_STRONG_INLINE unsigned int* semaphore() const {
    return stream_->semaphore();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpy(void* dst, const void* src, size_t n) const {
#if !defined(EIGEN_HIP_DEVICE_COMPILE)
    hipError_t err = hipMemcpyAsync(dst, src, n, hipMemcpyDeviceToDevice,
                                      stream_->stream());
    EIGEN_UNUSED_VARIABLE(err)
    assert(err == hipSuccess);
#else
  eigen_assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_STRONG_INLINE void memcpyHostToDevice(void* dst, const void* src, size_t n) const {
    hipError_t err =
        hipMemcpyAsync(dst, src, n, hipMemcpyHostToDevice, stream_->stream());
    EIGEN_UNUSED_VARIABLE(err)
    assert(err == hipSuccess);
  }

   EIGEN_STRONG_INLINE void memcpyDeviceToHost(void* dst, const void* src, size_t n) const {
    hipError_t err =
        hipMemcpyAsync(dst, src, n, hipMemcpyDeviceToHost, stream_->stream());
    EIGEN_UNUSED_VARIABLE(err)
    assert(err == hipSuccess);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memset(void* buffer, int c, size_t n) const {
#if !defined(EIGEN_HIP_DEVICE_COMPILE)
    //TODO:hipError_t err = hipMemsetAsync(buffer, c, n, stream_->stream());
    hipError_t err = hipMemset(buffer, c, n);
    EIGEN_UNUSED_VARIABLE(err)
    assert(err == hipSuccess);
#else
  eigen_assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_STRONG_INLINE size_t numThreads() const {
    // FIXME
    return 32;
  }

  EIGEN_STRONG_INLINE size_t firstLevelCacheSize() const {
    // FIXME
    return 48*1024;
  }

  EIGEN_STRONG_INLINE size_t lastLevelCacheSize() const {
    // We won't try to take advantage of the l2 cache for the time being, and
    // there is no l3 cache on hip devices.
    return firstLevelCacheSize();
  }

// FIXME - this will move into HIP
#if defined(EIGEN_HIP_DEVICE_COMPILE)
#undef assert
#define assert(COND)
#endif

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void synchronize() const {
#if defined(EIGEN_HIPCC) && \
    !defined(EIGEN_HIP_DEVICE_COMPILE)
    hipError_t err = hipStreamSynchronize(stream_->stream());
    if (err != hipSuccess) {
      std::cerr << "Error detected in HIP stream: "
                << hipGetErrorString(err)
                << std::endl;
      assert(err == hipSuccess);
    }
#else
    assert(false && "The default device should be used instead to generate kernel code");
#endif
  }

  EIGEN_STRONG_INLINE int getNumHipMultiProcessors() const {
    return stream_->deviceProperties().multiProcessorCount;
  }
  EIGEN_STRONG_INLINE int maxHipThreadsPerBlock() const {
    return stream_->deviceProperties().maxThreadsPerBlock;
  }
  EIGEN_STRONG_INLINE int maxHipThreadsPerMultiProcessor() const {
    return stream_->deviceProperties().maxThreadsPerMultiProcessor;
  }
  EIGEN_STRONG_INLINE int sharedMemPerBlock() const {
    return stream_->deviceProperties().sharedMemPerBlock;
  }
   EIGEN_STRONG_INLINE int majorDeviceVersion() const {
    return stream_->deviceProperties().major;
  }
  EIGEN_STRONG_INLINE int minorDeviceVersion() const {
    return stream_->deviceProperties().minor;
  }

  EIGEN_STRONG_INLINE int maxBlocks() const {
    return max_blocks_;
  }

  // This function checks if the HIP runtime recorded an error for the
  // underlying stream device.
  inline bool ok() const {
#if defined(EIGEN_HIPCC)
    hipError_t error = hipStreamQuery(stream_->stream());
    return (error == hipSuccess) || (error == hipErrorNotReady);
#else
    return false;
#endif
  }

 private:
  const StreamInterface* stream_;
  int max_blocks_;
};

#define LAUNCH_HIP_KERNEL(kernel, gridsize, blocksize, sharedmem, device, ...)             \
  hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel), dim3(gridsize), dim3(blocksize), (sharedmem), (device).stream(), (__VA_ARGS__)); \
  assert(hipGetLastError() == hipSuccess);


// FIXME: Should be device and kernel specific.
#if defined(EIGEN_HIPCC)
static EIGEN_DEVICE_FUNC inline void setHipSharedMemConfig(hipSharedMemConfig config) {
#if !defined(EIGEN_HIP_DEVICE_COMPILE)
  hipError_t status = hipDeviceSetSharedMemConfig(config);
  EIGEN_UNUSED_VARIABLE(status)
  assert(status == hipSuccess);
#else
  EIGEN_UNUSED_VARIABLE(config)
#endif
}
#endif

}  // end namespace Eigen

#endif  // EIGEN_CXX11_TENSOR_TENSOR_DEVICE_HIP_H
