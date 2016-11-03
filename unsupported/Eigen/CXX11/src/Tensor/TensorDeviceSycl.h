// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>
// Mehdi Goli    Codeplay Software Ltd.
// Ralph Potter  Codeplay Software Ltd.
// Luke Iwanski  Codeplay Software Ltd.
// Cummins Chris PhD student at The University of Edinburgh.
// Contact: <eigen@codeplay.com>

//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#if defined(EIGEN_USE_SYCL) && !defined(EIGEN_CXX11_TENSOR_TENSOR_DEVICE_SYCL_H)
#define EIGEN_CXX11_TENSOR_TENSOR_DEVICE_SYCL_H

namespace Eigen {
/// \struct BufferT is used to specialise add_sycl_buffer function for
//  two types of buffer we have. When the MapAllocator is true, we create the
//  sycl buffer with MapAllocator.
/// We have to const_cast the input pointer in order to work around the fact
/// that sycl does not accept map allocator for const pointer.
template <typename T, bool MapAllocator>
struct BufferT {
  using Type = cl::sycl::buffer<T, 1, cl::sycl::map_allocator<T>>;
  static inline void add_sycl_buffer(
      const T *ptr, size_t num_bytes,
      std::map<const void *, std::shared_ptr<void>> &buffer_map) {
    buffer_map.insert(std::pair<const void *, std::shared_ptr<void>>(
        ptr, std::shared_ptr<void>(std::make_shared<Type>(
                 Type(const_cast<T *>(ptr), cl::sycl::range<1>(num_bytes))))));
  }
};

/// specialisation of the \ref BufferT when the MapAllocator is false. In this
/// case we only create the device-only buffer.
template <typename T>
struct BufferT<T, false> {
  using Type = cl::sycl::buffer<T, 1>;
  static inline void add_sycl_buffer(
      const T *ptr, size_t num_bytes,
      std::map<const void *, std::shared_ptr<void>> &buffer_map) {
    buffer_map.insert(std::pair<const void *, std::shared_ptr<void>>(
        ptr, std::shared_ptr<void>(
                 std::make_shared<Type>(Type(cl::sycl::range<1>(num_bytes))))));
  }
};

struct SyclDevice {
  /// class members
  /// sycl queue
  cl::sycl::queue &m_queue;
  /// std::map is the container used to make sure that we create only one buffer
  /// per pointer. The lifespan of the buffer
  /// now depends on the lifespan of SyclDevice. If a non-read-only pointer is
  /// needed to be accessed on the host we should manually deallocate it.
  mutable std::map<const void *, std::shared_ptr<void>> buffer_map;

  SyclDevice(cl::sycl::queue &q) : m_queue(q) {}
  // destructor
  ~SyclDevice() { deallocate_all(); }

  template <typename T>
  void deallocate(const T *p) const {
    auto it = buffer_map.find(p);
    if (it != buffer_map.end()) {
      buffer_map.erase(it);
    }
  }
  void deallocate_all() const { buffer_map.clear(); }

  /// creation of sycl accessor for a buffer. This function first tries to find
  /// the buffer in the buffer_map.
  /// If found it gets the accessor from it, if not, the function then adds an
  /// entry by creating a sycl buffer
  /// for that particular pointer.
  template <cl::sycl::access::mode AcMd, bool MapAllocator, typename T>
  inline cl::sycl::accessor<T, 1, AcMd, cl::sycl::access::target::global_buffer>
  get_sycl_accessor(size_t num_bytes, cl::sycl::handler &cgh,
                    const T *ptr) const {
    auto it = buffer_map.find(ptr);
    if (it == buffer_map.end()) {
      BufferT<T, MapAllocator>::add_sycl_buffer(ptr, num_bytes, buffer_map);
    }
    return (
        ((typename BufferT<T, MapAllocator>::Type *)(buffer_map.at(ptr).get()))
            ->template get_access<AcMd>(cgh));
  }

  /// allocating memory on the cpu
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void *allocate(size_t num_bytes) const {
    return internal::aligned_malloc(num_bytes);
  }

  // some runtime conditions that can be applied here
  bool isDeviceSuitable() const { return true; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void deallocate(void *buffer) const {
    internal::aligned_free(buffer);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpy(void *dst, const void *src,
                                                    size_t n) const {
    ::memcpy(dst, src, n);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpyHostToDevice(
      void *dst, const void *src, size_t n) const {
    memcpy(dst, src, n);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpyDeviceToHost(
      void *dst, const void *src, size_t n) const {
    memcpy(dst, src, n);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memset(void *buffer, int c,
                                                    size_t n) const {
    ::memset(buffer, c, n);
  }
};
}  // end namespace Eigen

#endif  // EIGEN_CXX11_TENSOR_TENSOR_DEVICE_SYCL_H
