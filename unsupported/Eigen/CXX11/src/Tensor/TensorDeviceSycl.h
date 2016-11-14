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
  /// class members:

  /// sycl queue
  mutable cl::sycl::queue m_queue;

  /// std::map is the container used to make sure that we create only one buffer
  /// per pointer. The lifespan of the buffer now depends on the lifespan of SyclDevice.
  /// If a non-read-only pointer is needed to be accessed on the host we should manually deallocate it.
  mutable std::map<const void *, std::shared_ptr<void>> buffer_map;

  /// creating device by using selector
  template<typename dev_Selector>  explicit SyclDevice(dev_Selector s):
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

  /// This is used to deallocate the device pointer. p is used as a key inside
  /// the map to find the device buffer and delete it.
  template <typename T> EIGEN_STRONG_INLINE void deallocate(T *p) const {
    auto it = buffer_map.find(p);
    if (it != buffer_map.end()) {
      buffer_map.erase(it);
      internal::aligned_free(p);
    }
  }

  /// This is called by the SyclDevice destructor to release all allocated memory if the user didn't already do so.
  /// We also free the host pointer that we have dedicated as a key to accessing the device buffer.
  EIGEN_STRONG_INLINE void deallocate_all() const {
    std::map<const void *, std::shared_ptr<void>>::iterator it=buffer_map.begin();
    while (it!=buffer_map.end()) {
      auto p=it->first;
      buffer_map.erase(it);
      internal::aligned_free(const_cast<void*>(p));
      it=buffer_map.begin();
    }
    buffer_map.clear();
  }

  /// Creation of sycl accessor for a buffer. This function first tries to find
  /// the buffer in the buffer_map. If found it gets the accessor from it, if not,
  /// the function then adds an entry by creating a sycl buffer for that particular pointer.
  template <cl::sycl::access::mode AcMd, typename T> EIGEN_STRONG_INLINE cl::sycl::accessor<T, 1, AcMd, cl::sycl::access::target::global_buffer>
  get_sycl_accessor(size_t num_bytes, cl::sycl::handler &cgh, const T * ptr) const {
    return (get_sycl_buffer<T>(num_bytes, ptr)->template get_access<AcMd, cl::sycl::access::target::global_buffer>(cgh));
  }

  /// Inserting a new sycl buffer. For every allocated device pointer only one buffer would be created. The buffer type is a device- only buffer.
  /// The key pointer used to access the device buffer(the device pointer(ptr) ) must be initialised by the allocate function.
  template<typename T> EIGEN_STRONG_INLINE  std::pair<std::map<const void *, std::shared_ptr<void>>::iterator,bool> add_sycl_buffer(size_t num_bytes, const T *ptr) const {
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

  /// Accessing the created sycl device buffer for the device pointer
  template <typename T> EIGEN_STRONG_INLINE cl::sycl::buffer<T, 1>* get_sycl_buffer(size_t num_bytes,const T * ptr) const {
    return static_cast<cl::sycl::buffer<T, 1>*>(add_sycl_buffer(num_bytes, ptr).first->second.get());
  }

  /// This is used to prepare the number of threads and also the number of threads per block for sycl kernels
  EIGEN_STRONG_INLINE void parallel_for_setup(size_t n, size_t &tileSize, size_t &rng, size_t &GRange)  const {
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

  /// Allocating device pointer. This pointer is actually an 8 bytes host pointer used as key to access the sycl device buffer.
  /// The reason is that we cannot use device buffer as a pointer as a m_data in Eigen leafNode expressions. So we create a key
  /// pointer to be used in Eigen expression construction. When we convert the Eigen construction into the sycl construction we
  /// use this pointer as a key in our buffer_map and we make sure that we dedicate only one buffer only for this pointer.
  /// The device pointer would be deleted by calling deallocate function.
  EIGEN_STRONG_INLINE void *allocate(size_t) const {
    return internal::aligned_malloc(8);
  }

  // some runtime conditions that can be applied here
  EIGEN_STRONG_INLINE bool isDeviceSuitable() const { return true; }

  template <typename T> EIGEN_STRONG_INLINE std::map<const void *, std::shared_ptr<void>>::iterator find_nearest(const T* ptr) const {
    auto it1 = buffer_map.find(ptr);
    if (it1 != buffer_map.end()){
      return it1;
    }
    else{
      for(std::map<const void *, std::shared_ptr<void>>::iterator it=buffer_map.begin(); it!=buffer_map.end(); ++it){
        auto size = ((cl::sycl::buffer<T, 1>*)it->second.get())->get_size();
        if((static_cast<const T*>(it->first) <  ptr) && (ptr < (static_cast<const T*>(it->first)) + size)) return it;
      }
    }
    return buffer_map.end();
  }

  /// the memcpy function
  template<typename T> EIGEN_STRONG_INLINE void memcpy(void *dst, const T *src, size_t n) const {
    auto it1 = find_nearest(src);
    auto it2 = find_nearest(static_cast<T*>(dst));
    if ((it1 != buffer_map.end()) && (it2!=buffer_map.end())) {
      auto offset= (src - (static_cast<const T*>(it1->first)));
      auto i= ((static_cast<T*>(dst)) - const_cast<T*>((static_cast<const T*>(it2->first))));
      size_t rng, GRange, tileSize;
      parallel_for_setup(n/sizeof(T), tileSize, rng, GRange);
      m_queue.submit([&](cl::sycl::handler &cgh) {
        auto src_acc =((cl::sycl::buffer<T, 1>*)it1->second.get())-> template get_access<cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer>(cgh);
        auto dst_acc =((cl::sycl::buffer<T, 1>*)it2->second.get())->  template get_access<cl::sycl::access::mode::discard_write, cl::sycl::access::target::global_buffer>(cgh);
        typedef decltype(src_acc) DevToDev;
        cgh.parallel_for<DevToDev>( cl::sycl::nd_range<1>(cl::sycl::range<1>(GRange), cl::sycl::range<1>(tileSize)), [=](cl::sycl::nd_item<1> itemID) {
          auto globalid=itemID.get_global_linear_id();
          if (globalid< rng) {
            dst_acc[globalid+i ]=src_acc[globalid+offset];
          }
        });
      });
      m_queue.throw_asynchronous();
    } else{
      eigen_assert("no source or destination device memory found.");
    }
    //::memcpy(dst, src, n);
  }

  /// The memcpyHostToDevice is used to copy the device only pointer to a host pointer. Using the device
  /// pointer created as a key we find the sycl buffer and get the host accessor with discard_write mode
  /// on it. Using a discard_write accessor guarantees that we do not bring back the current value of the
  /// buffer to host. Then we use the memcpy to copy the data to the host accessor. The first time that
  /// this buffer is accessed, the data will be copied to the device.
  template<typename T> EIGEN_STRONG_INLINE void memcpyHostToDevice(T *dst, const T *src, size_t n) const {

    auto host_acc= get_sycl_buffer(n, dst)-> template get_access<cl::sycl::access::mode::discard_write, cl::sycl::access::target::host_buffer>();
    ::memcpy(host_acc.get_pointer(), src, n);
  }
  /// The memcpyDeviceToHost is used to copy the data from host to device. Here, in order to avoid double copying the data. We create a sycl
  /// buffer with map_allocator for the destination pointer with a discard_write accessor on it. The lifespan of the buffer is bound to the
  /// lifespan of the memcpyDeviceToHost function. We create a kernel to copy the data, from the device- only source buffer to the destination
  /// buffer with map_allocator on the gpu in parallel. At the end of the function call the destination buffer would be destroyed and the data
  /// would be available on the dst pointer using fast copy technique (map_allocator). In this case we can make sure that we copy the data back
  /// to the cpu only once per function call.
  template<typename T> EIGEN_STRONG_INLINE void memcpyDeviceToHost(T *dst, const T *src, size_t n) const {
    auto it = find_nearest(src);
    auto offset = src- (static_cast<const T*>(it->first));
    if (it != buffer_map.end()) {
    size_t rng, GRange, tileSize;
    parallel_for_setup(n/sizeof(T), tileSize, rng, GRange);
    // Assuming that the dst is the start of the destination pointer
    auto dest_buf = cl::sycl::buffer<T, 1, cl::sycl::map_allocator<T>>(dst, cl::sycl::range<1>(rng));
    typedef decltype(dest_buf) SYCLDTOH;
    m_queue.submit([&](cl::sycl::handler &cgh) {
      auto src_acc= (static_cast<cl::sycl::buffer<T, 1>*>(it->second.get()))-> template get_access<cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer>(cgh);
      auto dst_acc =dest_buf.template get_access<cl::sycl::access::mode::discard_write, cl::sycl::access::target::global_buffer>(cgh);
      cgh.parallel_for<SYCLDTOH>( cl::sycl::nd_range<1>(cl::sycl::range<1>(GRange), cl::sycl::range<1>(tileSize)), [=](cl::sycl::nd_item<1> itemID) {
        auto globalid=itemID.get_global_linear_id();
        if (globalid< dst_acc.get_size()) {
          dst_acc[globalid] = src_acc[globalid + offset];
        }
      });
    });
    m_queue.throw_asynchronous();

    } else{
      eigen_assert("no device memory found. The memory might be destroyed before creation");
    }
  }

  /// Here is the implementation of memset function on sycl.
  template<typename T>  EIGEN_STRONG_INLINE void memset(T *buff, int c, size_t n) const {
      size_t rng, GRange, tileSize;
      parallel_for_setup(n/sizeof(T), tileSize, rng, GRange);
      m_queue.submit([&](cl::sycl::handler &cgh) {
        auto buf_acc =get_sycl_buffer(n, buff)-> template get_access<cl::sycl::access::mode::discard_write, cl::sycl::access::target::global_buffer>(cgh);
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
  /// No need for sycl it should act the same as CPU version
  EIGEN_STRONG_INLINE int majorDeviceVersion() const {
  return 1;
  }
  /// There is no need to synchronise the stream in sycl as it is automatically handled by sycl runtime scheduler.
  EIGEN_STRONG_INLINE void synchronize() const {}
};

}  // end namespace Eigen

#endif  // EIGEN_CXX11_TENSOR_TENSOR_DEVICE_SYCL_H
