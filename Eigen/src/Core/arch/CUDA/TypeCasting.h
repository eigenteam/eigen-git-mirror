// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TYPE_CASTING_CUDA_H
#define EIGEN_TYPE_CASTING_CUDA_H

namespace Eigen {

namespace internal {

#if defined(EIGEN_HAS_CUDA_FP16)

template<>
struct scalar_cast_op<float, half> {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_cast_op)
  typedef half result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half operator() (const float& a) const {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
      return __float2half(a);
    #else
      return half(a);
    #endif
  }
};

template<>
struct functor_traits<scalar_cast_op<float, half> >
{ enum { Cost = NumTraits<float>::AddCost, PacketAccess = false }; };


template<>
struct scalar_cast_op<int, half> {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_cast_op)
  typedef half result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half operator() (const int& a) const {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
      return __float2half(static_cast<float>(a));
    #else
      return half(static_cast<float>(a));
    #endif
  }
};

template<>
struct functor_traits<scalar_cast_op<int, half> >
{ enum { Cost = NumTraits<float>::AddCost, PacketAccess = false }; };


template<>
struct scalar_cast_op<half, float> {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_cast_op)
  typedef float result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float operator() (const half& a) const {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
      return __half2float(a);
    #else
      return static_cast<float>(a);
    #endif
  }
};

template<>
struct functor_traits<scalar_cast_op<half, float> >
{ enum { Cost = NumTraits<float>::AddCost, PacketAccess = false }; };



#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300

template <>
struct type_casting_traits<half, float> {
  enum {
    VectorizedCast = 1,
    SrcCoeffRatio = 2,
    TgtCoeffRatio = 1
  };
};

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pcast<half2, float4>(const half2& a, const half2& b) {
  float2 r1 = __half22float2(a);
  float2 r2 = __half22float2(b);
  return make_float4(r1.x, r1.y, r2.x, r2.y);
}

template <>
struct type_casting_traits<float, half> {
  enum {
    VectorizedCast = 1,
    SrcCoeffRatio = 1,
    TgtCoeffRatio = 2
  };
};

template<> EIGEN_STRONG_INLINE half2 pcast<float4, half2>(const float4& a) {
  // Simply discard the second half of the input
  return __float22half2_rn(make_float2(a.x, a.y));
}

#endif
#endif

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_TYPE_CASTING_CUDA_H
