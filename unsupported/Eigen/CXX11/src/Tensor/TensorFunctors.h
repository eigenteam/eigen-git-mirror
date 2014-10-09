// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_FUNCTORS_H
#define EIGEN_CXX11_TENSOR_TENSOR_FUNCTORS_H

namespace Eigen {
namespace internal {

// Standard reduction functors
template <typename T> struct SumReducer
{
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE SumReducer() : m_sum(0) { }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T t) {
    m_sum += t;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalize() const {
    return m_sum;
  }

 private:
  typename internal::remove_all<T>::type m_sum;
};

template <typename T> struct MaxReducer
{
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE MaxReducer() : m_max(-(std::numeric_limits<T>::max)()) { }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T t) {
    if (t > m_max) { m_max = t; }
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalize() const {
    return m_max;
  }

 private:
  typename internal::remove_all<T>::type m_max;
};

template <typename T> struct MinReducer
{
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE MinReducer() : m_min((std::numeric_limits<T>::max)()) { }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T t) {
    if (t < m_min) { m_min = t; }
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalize() const {
    return m_min;
  }

 private:
  typename internal::remove_all<T>::type m_min;
};


#if !defined (EIGEN_USE_GPU) || !defined(__CUDACC__) || !defined(__CUDA_ARCH__)
// We're not compiling a cuda kernel
template <typename T> struct UniformRandomGenerator {
  template<typename Index>
  T operator()(Index, Index = 0) const {
    return random<T>();
  }
  template<typename Index>
  typename internal::packet_traits<T>::type packetOp(Index, Index = 0) const {
    const int packetSize = internal::packet_traits<T>::size;
    EIGEN_ALIGN_DEFAULT T values[packetSize];
    for (int i = 0; i < packetSize; ++i) {
      values[i] = random<T>();
    }
    return internal::pload<typename internal::packet_traits<T>::type>(values);
  }
};

#else

// We're compiling a cuda kernel
template <typename T> struct UniformRandomGenerator;

template <> struct UniformRandomGenerator<float> {
  UniformRandomGenerator() {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(0, tid, 0, &m_state);
  }

  template<typename Index>
  float operator()(Index, Index = 0) const {
    return curand_uniform(&m_state);
  }
  template<typename Index>
  float4 packetOp(Index, Index = 0) const {
    return curand_uniform4(&m_state);
  }

 private:
  mutable curandStatePhilox4_32_10_t m_state;
};

template <> struct UniformRandomGenerator<double> {
  UniformRandomGenerator() {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(0, tid, 0, &m_state);
  }
  template<typename Index>
  double operator()(Index, Index = 0) const {
    return curand_uniform_double(&m_state);
  }
  template<typename Index>
  double2 packetOp(Index, Index = 0) const {
    return curand_uniform2_double(&m_state);
  }

 private:
  mutable curandStatePhilox4_32_10_t m_state;
};

#endif


} // end namespace internal
} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_FUNCTORS_H
