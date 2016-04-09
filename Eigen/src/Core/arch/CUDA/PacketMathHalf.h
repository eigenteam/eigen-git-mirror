// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PACKET_MATH_HALF_CUDA_H
#define EIGEN_PACKET_MATH_HALF_CUDA_H

#if defined(EIGEN_HAS_CUDA_FP16)

// Make sure this is only available when targeting a GPU: we don't want to
// introduce conflicts between these packet_traits definitions and the ones
// we'll use on the host side (SSE, AVX, ...)
#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)

// Most of the following operations require arch >= 5.3
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530

namespace Eigen {
namespace internal {

template<> struct is_arithmetic<half2> { enum { value = true }; };

template<> struct packet_traits<half> : default_packet_traits
{
  typedef half2 type;
  typedef half2 half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size=2,
    HasHalfPacket = 0,
    HasDiv  = 1
  };
};


template<> struct unpacket_traits<half2> { typedef half type; enum {size=2, alignment=Aligned16}; typedef half2 half; };

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pset1<half2>(const half& from) {
  return __half2half2(from);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pload<half2>(const half* from) {
  return *reinterpret_cast<const half2*>(from);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 ploadu<half2>(const half* from) {
  return __halves2half2(from[0], from[1]);
}

template<> EIGEN_STRONG_INLINE half2 ploaddup<half2>(const half*  from) {
  return __halves2half2(from[0], from[0]);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pstore<half>(half* to, const half2& from) {
  *reinterpret_cast<half2*>(to) = from;
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pstoreu<half>(half* to, const half2& from) {
  to[0] = __low2half(from);
  to[1] = __high2half(from);
}

template<>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE half2 ploadt_ro<half2, Aligned>(const half* from) {
  return __ldg((const half2*)from);
}

template<>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE half2 ploadt_ro<half2, Unaligned>(const half* from) {
  return __halves2half2(__ldg(from+0), __ldg(from+1));
}

template<> EIGEN_DEVICE_FUNC inline half2 pgather<half, half2>(const half* from, Index stride) {
  return __halves2half2(from[0*stride], from[1*stride]);
}

template<> EIGEN_DEVICE_FUNC inline void pscatter<half, half2>(half* to, const half2& from, Index stride) {
  to[stride*0] = __low2half(from);
  to[stride*1] = __high2half(from);
}

template<> EIGEN_DEVICE_FUNC inline half pfirst<half2>(const half2& a) {
  return __low2half(a);
}

template<> EIGEN_DEVICE_FUNC inline half2 pabs<half2>(const half2& a) {
  half2 result;
  result.x = a.x & 0x7FFF7FFF;
  return result;
}


EIGEN_DEVICE_FUNC inline void
ptranspose(PacketBlock<half2,2>& kernel) {
  half a1 = __low2half(kernel.packet[0]);
  half a2 = __high2half(kernel.packet[0]);
  half b1 = __low2half(kernel.packet[1]);
  half b2 = __high2half(kernel.packet[1]);
  kernel.packet[0] = __halves2half2(a1, b1);
  kernel.packet[1] = __halves2half2(a2, b2);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 plset<half2>(const half& a) {
  return __halves2half2(a, __hadd(a, __float2half(1.0f)));
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 padd<half2>(const half2& a, const half2& b) {
  return __hadd2(a, b);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 psub<half2>(const half2& a, const half2& b) {
  return __hsub2(a, b);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pnegate(const half2& a) {
  return __hneg2(a);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pconj(const half2& a) { return a; }

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pmul<half2>(const half2& a, const half2& b) {
  return __hmul2(a, b);
}

 template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pmadd<half2>(const half2& a, const half2& b, const half2& c) {
   return __hfma2(a, b, c);
 }

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pdiv<half2>(const half2& a, const half2& b) {
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float b1 = __low2float(b);
  float b2 = __high2float(b);
  float r1 = a1 / b1;
  float r2 = a2 / b2;
  return __floats2half2_rn(r1, r2);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pmin<half2>(const half2& a, const half2& b) {
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float b1 = __low2float(b);
  float b2 = __high2float(b);
  half r1 = a1 < b1 ? __low2half(a) : __low2half(b);
  half r2 = a2 < b2 ? __high2half(a) : __high2half(b);
  return __halves2half2(r1, r2);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pmax<half2>(const half2& a, const half2& b) {
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float b1 = __low2float(b);
  float b2 = __high2float(b);
  half r1 = a1 > b1 ? __low2half(a) : __low2half(b);
  half r2 = a2 > b2 ? __high2half(a) : __high2half(b);
  return __halves2half2(r1, r2);
}

template<> EIGEN_DEVICE_FUNC inline half predux<half2>(const half2& a) {
  return __hadd(__low2half(a), __high2half(a));
}

template<> EIGEN_DEVICE_FUNC inline half predux_max<half2>(const half2& a) {
  half first = __low2half(a);
  half second = __high2half(a);
  return __hgt(first, second) ? first : second;
}

template<> EIGEN_DEVICE_FUNC inline half predux_min<half2>(const half2& a) {
  half first = __low2half(a);
  half second = __high2half(a);
  return __hlt(first, second) ? first : second;
}

template<> EIGEN_DEVICE_FUNC inline half predux_mul<half2>(const half2& a) {
  return __hmul(__low2half(a), __high2half(a));
}

} // end namespace internal

} // end namespace Eigen

#endif
#endif
#endif
#endif // EIGEN_PACKET_MATH_HALF_CUDA_H
