// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Mehdi Goli    Codeplay Software Ltd.
// Ralph Potter  Codeplay Software Ltd.
// Luke Iwanski  Codeplay Software Ltd.
// Contact: <eigen@codeplay.com>
// Copyright (C) 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>

//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#if defined(EIGEN_USE_SYCL) && !defined(EIGEN_CXX11_TENSOR_TENSOR_DEVICE_SYCL_H)
#define EIGEN_CXX11_TENSOR_TENSOR_DEVICE_SYCL_H

namespace Eigen {
struct SyclDevice {
  /// class members
  /// sycl queue
  mutable cl::sycl::queue m_queue;
  /// std::map is the container used to make sure that we create only one buffer
  /// per pointer. The lifespan of the buffer now depends on the lifespan of SyclDevice.
  /// If a non-read-only pointer is needed to be accessed on the host we should manually deallocate it.
  mutable std::map<const void *, std::shared_ptr<void>> buffer_map;
  /// creating device by using selector
  template<typename dev_Selector> SyclDevice(dev_Selector s)
  :
#ifdef EIGEN_EXCEPTIONS
  m_queue(cl::sycl::queue(s, [=](cl::sycl::exception_list l) {
    for (const auto& e : l) {
      try {
        std::rethrow_exception(e);
      } catch (cl::sycl::exception e) {
          std::cout << e.what() << std::endl;
        }
    }
  }))
#else
  m_queue(cl::sycl::queue(s))
#endif
  {}
  // destructor
  ~SyclDevice() { deallocate_all(); }

  template <typename T> void deallocate(T *p) const {
    auto it = buffer_map.find(p);
    if (it != buffer_map.end()) {
      buffer_map.erase(it);
      internal::aligned_free(p);
    }
  }
  void deallocate_all() const {
    std::map<const void *, std::shared_ptr<void>>::iterator it=buffer_map.begin();
    while (it!=buffer_map.end()) {
      auto p=it->first;
      buffer_map.erase(it);
      internal::aligned_free(const_cast<void*>(p));
      it=buffer_map.begin();
    }
    buffer_map.clear();
  }

  /// creation of sycl accessor for a buffer. This function first tries to find
  /// the buffer in the buffer_map. If found it gets the accessor from it, if not,
  ///the function then adds an entry by creating a sycl buffer for that particular pointer.
  template <cl::sycl::access::mode AcMd, typename T> inline cl::sycl::accessor<T, 1, AcMd, cl::sycl::access::target::global_buffer>
  get_sycl_accessor(size_t num_bytes, cl::sycl::handler &cgh, const T * ptr) const {
    return (get_sycl_buffer<T>(num_bytes, ptr)->template get_access<AcMd, cl::sycl::access::target::global_buffer>(cgh));
  }

  template<typename T> inline  std::pair<std::map<const void *, std::shared_ptr<void>>::iterator,bool> add_sycl_buffer(const T *ptr, size_t num_bytes) const {
    using Type = cl::sycl::buffer<T, 1>;
    std::pair<std::map<const void *, std::shared_ptr<void>>::iterator,bool> ret;
    if(ptr!=nullptr){
       ret= buffer_map.insert(std::pair<const void *, std::shared_ptr<void>>(ptr, std::shared_ptr<void>(new Type(cl::sycl::range<1>(num_bytes)),
        [](void *dataMem) { delete static_cast<Type*>(dataMem); })));
      (static_cast<Type*>(ret.first->second.get()))->set_final_data(nullptr);
    } else {
      eigen_assert("The device memory is not allocated. Please call allocate on the device!!");
    }
    return ret;
  }

  template <typename T> inline cl::sycl::buffer<T, 1>* get_sycl_buffer(size_t num_bytes,const T * ptr) const {
    return static_cast<cl::sycl::buffer<T, 1>*>(add_sycl_buffer(ptr, num_bytes).first->second.get());
  }

  /// allocating memory on the cpu
  void *allocate(size_t) const {
    return internal::aligned_malloc(8);
  }

  // some runtime conditions that can be applied here
  bool isDeviceSuitable() const { return true; }

  void memcpy(void *dst, const void *src, size_t n) const {
    ::memcpy(dst, src, n);
  }

  template<typename T> void memcpyHostToDevice(T *dst, const T *src, size_t n) const {
    auto host_acc= (static_cast<cl::sycl::buffer<T, 1>*>(add_sycl_buffer(dst, n).first->second.get()))-> template get_access<cl::sycl::access::mode::discard_write, cl::sycl::access::target::host_buffer>();
    memcpy(host_acc.get_pointer(), src, n);
  }

 inline void parallel_for_setup(size_t n, size_t &tileSize, size_t &rng, size_t &GRange)  const {
      tileSize =m_queue.get_device(). template get_info<cl::sycl::info::device::max_work_group_size>()/2;
      rng = n;
      if (rng==0) rng=1;
       GRange=rng;
      if (tileSize>GRange) tileSize=GRange;
      else if(GRange>tileSize){
        size_t xMode = GRange % tileSize;
        if (xMode != 0) GRange += (tileSize - xMode);
      }
    }

  template<typename T> void memcpyDeviceToHost(T *dst, const T *src, size_t n) const {
    auto it = buffer_map.find(src);
    if (it != buffer_map.end()) {
    size_t rng, GRange, tileSize;
    parallel_for_setup(n/sizeof(T), tileSize, rng, GRange);

    auto dest_buf = cl::sycl::buffer<T, 1, cl::sycl::map_allocator<T>>(dst, cl::sycl::range<1>(rng));
    typedef decltype(dest_buf) SYCLDTOH;
    m_queue.submit([&](cl::sycl::handler &cgh) {
      auto src_acc= (static_cast<cl::sycl::buffer<T, 1>*>(it->second.get()))-> template get_access<cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer>(cgh);
      auto dst_acc =dest_buf.template get_access<cl::sycl::access::mode::discard_write, cl::sycl::access::target::global_buffer>(cgh);
      cgh.parallel_for<SYCLDTOH>( cl::sycl::nd_range<1>(cl::sycl::range<1>(GRange), cl::sycl::range<1>(tileSize)), [=](cl::sycl::nd_item<1> itemID) {
      auto globalid=itemID.get_global_linear_id();
      if (globalid< dst_acc.get_size()) {
          dst_acc[globalid] = src_acc[globalid];
      }
      });
    });
    m_queue.throw_asynchronous();

    } else{
      eigen_assert("no device memory found. The memory might be destroyed before creation");
    }
  }

  template<typename T>  void memset(T *buff, int c, size_t n) const {

      size_t rng, GRange, tileSize;
      parallel_for_setup(n/sizeof(T), tileSize, rng, GRange);
      m_queue.submit([&](cl::sycl::handler &cgh) {
        auto buf_acc =(static_cast<cl::sycl::buffer<T, 1>*>(add_sycl_buffer(buff, n).first->second.get()))-> template get_access<cl::sycl::access::mode::discard_write, cl::sycl::access::target::global_buffer>(cgh);
        cgh.parallel_for<SyclDevice>( cl::sycl::nd_range<1>(cl::sycl::range<1>(GRange), cl::sycl::range<1>(tileSize)), [=](cl::sycl::nd_item<1> itemID) {
        auto globalid=itemID.get_global_linear_id();
        auto buf_ptr= reinterpret_cast<typename cl::sycl::global_ptr<unsigned char>::pointer_t>((&(*buf_acc.get_pointer())));
        if (globalid< buf_acc.get_size()) {
          for(size_t i=0; i<sizeof(T); i++)
            buf_ptr[globalid*sizeof(T) + i] = c;
        }
        });
      });
      m_queue.throw_asynchronous();
  }
  int majorDeviceVersion() const {
  return 1;
  }
};

}  // end namespace Eigen

#endif  // EIGEN_CXX11_TENSOR_TENSOR_DEVICE_SYCL_H
