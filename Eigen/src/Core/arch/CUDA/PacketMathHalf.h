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


namespace Eigen {
namespace internal {

// Most of the following operations require arch >= 3.0
#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300

template<> struct is_arithmetic<half2> { enum { value = true }; };

template<> struct packet_traits<Eigen::half> : default_packet_traits
{
  typedef half2 type;
  typedef half2 half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size=2,
    HasHalfPacket = 0,
    HasAdd    = 1,
    HasMul    = 1,
    HasDiv    = 1,
    HasSqrt   = 1,
    HasRsqrt  = 1,
    HasExp    = 1,
    HasLog    = 1,
    HasLog1p  = 1
  };
};

template<> struct unpacket_traits<half2> { typedef Eigen::half type; enum {size=2, alignment=Aligned16}; typedef half2 half; };

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pset1<half2>(const Eigen::half& from) {
  return __half2half2(from);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pload<half2>(const Eigen::half* from) {
  return *reinterpret_cast<const half2*>(from);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 ploadu<half2>(const Eigen::half* from) {
  return __halves2half2(from[0], from[1]);
}

template<> EIGEN_STRONG_INLINE half2 ploaddup<half2>(const Eigen::half*  from) {
  return __halves2half2(from[0], from[0]);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pstore<Eigen::half>(Eigen::half* to, const half2& from) {
  *reinterpret_cast<half2*>(to) = from;
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pstoreu<Eigen::half>(Eigen::half* to, const half2& from) {
  to[0] = __low2half(from);
  to[1] = __high2half(from);
}

template<>
 EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE half2 ploadt_ro<half2, Aligned>(const Eigen::half* from) {
#if __CUDA_ARCH__ >= 350
   return __ldg((const half2*)from);
#else
  return __halves2half2(*(from+0), *(from+1));
#endif
}

template<>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE half2 ploadt_ro<half2, Unaligned>(const Eigen::half* from) {
#if __CUDA_ARCH__ >= 350
   return __halves2half2(__ldg(from+0), __ldg(from+1));
#else
  return __halves2half2(*(from+0), *(from+1));
#endif
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pgather<Eigen::half, half2>(const Eigen::half* from, Index stride) {
  return __halves2half2(from[0*stride], from[1*stride]);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pscatter<Eigen::half, half2>(Eigen::half* to, const half2& from, Index stride) {
  to[stride*0] = __low2half(from);
  to[stride*1] = __high2half(from);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half pfirst<half2>(const half2& a) {
  return __low2half(a);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pabs<half2>(const half2& a) {
  half2 result;
  result.x = a.x & 0x7FFF7FFF;
  return result;
}


EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void
ptranspose(PacketBlock<half2,2>& kernel) {
  __half a1 = __low2half(kernel.packet[0]);
  __half a2 = __high2half(kernel.packet[0]);
  __half b1 = __low2half(kernel.packet[1]);
  __half b2 = __high2half(kernel.packet[1]);
  kernel.packet[0] = __halves2half2(a1, b1);
  kernel.packet[1] = __halves2half2(a2, b2);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 plset<half2>(const Eigen::half& a) {
#if __CUDA_ARCH__ >= 530
  return __halves2half2(a, __hadd(a, __float2half(1.0f)));
#else
  float f = __half2float(a) + 1.0f;
  return __halves2half2(a, __float2half(f));
#endif
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 padd<half2>(const half2& a, const half2& b) {
#if __CUDA_ARCH__ >= 530
  return __hadd2(a, b);
#else
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float b1 = __low2float(b);
  float b2 = __high2float(b);
  float r1 = a1 + b1;
  float r2 = a2 + b2;
  return __floats2half2_rn(r1, r2);
#endif
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 psub<half2>(const half2& a, const half2& b) {
#if __CUDA_ARCH__ >= 530
  return __hsub2(a, b);
#else
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float b1 = __low2float(b);
  float b2 = __high2float(b);
  float r1 = a1 - b1;
  float r2 = a2 - b2;
  return __floats2half2_rn(r1, r2);
#endif
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pnegate(const half2& a) {
#if __CUDA_ARCH__ >= 530
  return __hneg2(a);
#else
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  return __floats2half2_rn(-a1, -a2);
#endif
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pconj(const half2& a) { return a; }

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pmul<half2>(const half2& a, const half2& b) {
#if __CUDA_ARCH__ >= 530
  return __hmul2(a, b);
#else
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float b1 = __low2float(b);
  float b2 = __high2float(b);
  float r1 = a1 * b1;
  float r2 = a2 * b2;
  return __floats2half2_rn(r1, r2);
#endif
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pmadd<half2>(const half2& a, const half2& b, const half2& c) {
#if __CUDA_ARCH__ >= 530
   return __hfma2(a, b, c);
#else
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float b1 = __low2float(b);
  float b2 = __high2float(b);
  float c1 = __low2float(c);
  float c2 = __high2float(c);
  float r1 = a1 * b1 + c1;
  float r2 = a2 * b2 + c2;
  return __floats2half2_rn(r1, r2);
#endif
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
  __half r1 = a1 < b1 ? __low2half(a) : __low2half(b);
  __half r2 = a2 < b2 ? __high2half(a) : __high2half(b);
  return __halves2half2(r1, r2);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pmax<half2>(const half2& a, const half2& b) {
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float b1 = __low2float(b);
  float b2 = __high2float(b);
  __half r1 = a1 > b1 ? __low2half(a) : __low2half(b);
  __half r2 = a2 > b2 ? __high2half(a) : __high2half(b);
  return __halves2half2(r1, r2);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half predux<half2>(const half2& a) {
#if __CUDA_ARCH__ >= 530
  return __hadd(__low2half(a), __high2half(a));
#else
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  return Eigen::half(half_impl::raw_uint16_to_half(__float2half_rn(a1 + a2)));
#endif
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half predux_max<half2>(const half2& a) {
#if __CUDA_ARCH__ >= 530
  __half first = __low2half(a);
  __half second = __high2half(a);
  return __hgt(first, second) ? first : second;
#else
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  return a1 > a2 ? __low2half(a) : __high2half(a);
#endif
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half predux_min<half2>(const half2& a) {
#if __CUDA_ARCH__ >= 530
  __half first = __low2half(a);
  __half second = __high2half(a);
  return __hlt(first, second) ? first : second;
#else
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  return a1 < a2 ? __low2half(a) : __high2half(a);
#endif
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half predux_mul<half2>(const half2& a) {
#if __CUDA_ARCH__ >= 530
  return __hmul(__low2half(a), __high2half(a));
#else
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  return Eigen::half(half_impl::raw_uint16_to_half(__float2half_rn(a1 * a2)));
#endif
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 plog1p<half2>(const half2& a) {
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float r1 = log1pf(a1);
  float r2 = log1pf(a2);
  return __floats2half2_rn(r1, r2);
}

#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000 && defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 530

template<>  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
half2 plog<half2>(const half2& a) {
  return h2log(a);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
half2 pexp<half2>(const half2& a) {
  return h2exp(a);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
half2 psqrt<half2>(const half2& a) {
  return h2sqrt(a);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
half2 prsqrt<half2>(const half2& a) {
  return h2rsqrt(a);
}

#else

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 plog<half2>(const half2& a) {
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float r1 = logf(a1);
  float r2 = logf(a2);
  return __floats2half2_rn(r1, r2);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pexp<half2>(const half2& a) {
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float r1 = expf(a1);
  float r2 = expf(a2);
  return __floats2half2_rn(r1, r2);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 psqrt<half2>(const half2& a) {
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float r1 = sqrtf(a1);
  float r2 = sqrtf(a2);
  return __floats2half2_rn(r1, r2);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 prsqrt<half2>(const half2& a) {
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float r1 = rsqrtf(a1);
  float r2 = rsqrtf(a2);
  return __floats2half2_rn(r1, r2);
}

#endif

#elif defined EIGEN_VECTORIZE_AVX

typedef struct {
  __m128i x;
} Packet8h;


template<> struct is_arithmetic<Packet8h> { enum { value = true }; };

template <>
struct packet_traits<Eigen::half> : default_packet_traits {
  typedef Packet8h type;
  // There is no half-size packet for Packet8h.
  typedef Packet8h half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 8,
    HasHalfPacket = 0,
    HasAdd    = 0,
    HasSub    = 0,
    HasMul    = 0,
    HasNegate = 0,
    HasAbs    = 0,
    HasAbs2   = 0,
    HasMin    = 0,
    HasMax    = 0,
    HasConj   = 0,
    HasSetLinear = 0,
    HasDiv = 0,
    HasSqrt = 0,
    HasRsqrt = 0,
    HasExp = 0,
    HasLog = 0,
    HasBlend = 0
  };
};


template<> struct unpacket_traits<Packet8h> { typedef Eigen::half type; enum {size=8, alignment=Aligned16}; typedef Packet8h half; };

template<> EIGEN_STRONG_INLINE Packet8h pset1<Packet8h>(const Eigen::half& from) {
  Packet8h result;
  result.x = _mm_set1_epi16(from.x);
  return result;
}

template<> EIGEN_STRONG_INLINE Eigen::half pfirst<Packet8h>(const Packet8h& from) {
  return half_impl::raw_uint16_to_half(static_cast<unsigned short>(_mm_extract_epi16(from.x, 0)));
}

template<> EIGEN_STRONG_INLINE Packet8h pload<Packet8h>(const Eigen::half* from) {
  Packet8h result;
  result.x = _mm_load_si128(reinterpret_cast<const __m128i*>(from));
  return result;
}

template<> EIGEN_STRONG_INLINE Packet8h ploadu<Packet8h>(const Eigen::half* from) {
  Packet8h result;
  result.x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(from));
  return result;
}

template<> EIGEN_STRONG_INLINE void pstore<Eigen::half>(Eigen::half* to, const Packet8h& from) {
  _mm_store_si128(reinterpret_cast<__m128i*>(to), from.x);
}

template<> EIGEN_STRONG_INLINE void pstoreu<Eigen::half>(Eigen::half* to, const Packet8h& from) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(to), from.x);
}

template<> EIGEN_STRONG_INLINE Packet8h
ploadquad<Packet8h>(const Eigen::half* from) {
  Packet8h result;
  unsigned short a = from[0].x;
  unsigned short b = from[1].x;
  result.x = _mm_set_epi16(b, b, b, b, a, a, a, a);
  return result;
}

EIGEN_STRONG_INLINE Packet8f half2float(const Packet8h& a) {
#ifdef EIGEN_HAS_FP16_C
  return _mm256_cvtph_ps(a.x);
#else
  EIGEN_ALIGN32 Eigen::half aux[8];
  pstore(aux, a);
  float f0(aux[0]);
  float f1(aux[1]);
  float f2(aux[2]);
  float f3(aux[3]);
  float f4(aux[4]);
  float f5(aux[5]);
  float f6(aux[6]);
  float f7(aux[7]);

  return _mm256_set_ps(f7, f6, f5, f4, f3, f2, f1, f0);
#endif
}

EIGEN_STRONG_INLINE Packet8h float2half(const Packet8f& a) {
#ifdef EIGEN_HAS_FP16_C
  Packet8h result;
  result.x = _mm256_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
  return result;
#else
  EIGEN_ALIGN32 float aux[8];
  pstore(aux, a);
  Eigen::half h0(aux[0]);
  Eigen::half h1(aux[1]);
  Eigen::half h2(aux[2]);
  Eigen::half h3(aux[3]);
  Eigen::half h4(aux[4]);
  Eigen::half h5(aux[5]);
  Eigen::half h6(aux[6]);
  Eigen::half h7(aux[7]);

  Packet8h result;
  result.x = _mm_set_epi16(h7.x, h6.x, h5.x, h4.x, h3.x, h2.x, h1.x, h0.x);
  return result;
#endif
}

template<> EIGEN_STRONG_INLINE Packet8h pconj(const Packet8h& a) { return a; }

template<> EIGEN_STRONG_INLINE Packet8h padd<Packet8h>(const Packet8h& a, const Packet8h& b) {
  Packet8f af = half2float(a);
  Packet8f bf = half2float(b);
  Packet8f rf = padd(af, bf);
  return float2half(rf);
}

template<> EIGEN_STRONG_INLINE Packet8h pmul<Packet8h>(const Packet8h& a, const Packet8h& b) {
  Packet8f af = half2float(a);
  Packet8f bf = half2float(b);
  Packet8f rf = pmul(af, bf);
  return float2half(rf);
}

template<> EIGEN_STRONG_INLINE Packet8h pgather<Eigen::half, Packet8h>(const Eigen::half* from, Index stride)
{
  Packet8h result;
  result.x = _mm_set_epi16(from[7*stride].x, from[6*stride].x, from[5*stride].x, from[4*stride].x, from[3*stride].x, from[2*stride].x, from[1*stride].x, from[0*stride].x);
  return result;
}

template<> EIGEN_STRONG_INLINE void pscatter<Eigen::half, Packet8h>(Eigen::half* to, const Packet8h& from, Index stride)
{
  EIGEN_ALIGN32 Eigen::half aux[8];
  pstore(aux, from);
  to[stride*0].x = aux[0].x;
  to[stride*1].x = aux[1].x;
  to[stride*2].x = aux[2].x;
  to[stride*3].x = aux[3].x;
  to[stride*4].x = aux[4].x;
  to[stride*5].x = aux[5].x;
  to[stride*6].x = aux[6].x;
  to[stride*7].x = aux[7].x;
}

EIGEN_STRONG_INLINE void
ptranspose(PacketBlock<Packet8h,8>& kernel) {
  __m128i a = kernel.packet[0].x;
  __m128i b = kernel.packet[1].x;
  __m128i c = kernel.packet[2].x;
  __m128i d = kernel.packet[3].x;
  __m128i e = kernel.packet[4].x;
  __m128i f = kernel.packet[5].x;
  __m128i g = kernel.packet[6].x;
  __m128i h = kernel.packet[7].x;

  __m128i a03b03 = _mm_unpacklo_epi16(a, b);
  __m128i c03d03 = _mm_unpacklo_epi16(c, d);
  __m128i e03f03 = _mm_unpacklo_epi16(e, f);
  __m128i g03h03 = _mm_unpacklo_epi16(g, h);
  __m128i a47b47 = _mm_unpackhi_epi16(a, b);
  __m128i c47d47 = _mm_unpackhi_epi16(c, d);
  __m128i e47f47 = _mm_unpackhi_epi16(e, f);
  __m128i g47h47 = _mm_unpackhi_epi16(g, h);

  __m128i a01b01c01d01 = _mm_unpacklo_epi32(a03b03, c03d03);
  __m128i a23b23c23d23 = _mm_unpackhi_epi32(a03b03, c03d03);
  __m128i e01f01g01h01 = _mm_unpacklo_epi32(e03f03, g03h03);
  __m128i e23f23g23h23 = _mm_unpackhi_epi32(e03f03, g03h03);
  __m128i a45b45c45d45 = _mm_unpacklo_epi32(a47b47, c47d47);
  __m128i a67b67c67d67 = _mm_unpackhi_epi32(a47b47, c47d47);
  __m128i e45f45g45h45 = _mm_unpacklo_epi32(e47f47, g47h47);
  __m128i e67f67g67h67 = _mm_unpackhi_epi32(e47f47, g47h47);

  __m128i a0b0c0d0e0f0g0h0 = _mm_unpacklo_epi64(a01b01c01d01, e01f01g01h01);
  __m128i a1b1c1d1e1f1g1h1 = _mm_unpackhi_epi64(a01b01c01d01, e01f01g01h01);
  __m128i a2b2c2d2e2f2g2h2 = _mm_unpacklo_epi64(a23b23c23d23, e23f23g23h23);
  __m128i a3b3c3d3e3f3g3h3 = _mm_unpackhi_epi64(a23b23c23d23, e23f23g23h23);
  __m128i a4b4c4d4e4f4g4h4 = _mm_unpacklo_epi64(a45b45c45d45, e45f45g45h45);
  __m128i a5b5c5d5e5f5g5h5 = _mm_unpackhi_epi64(a45b45c45d45, e45f45g45h45);
  __m128i a6b6c6d6e6f6g6h6 = _mm_unpacklo_epi64(a67b67c67d67, e67f67g67h67);
  __m128i a7b7c7d7e7f7g7h7 = _mm_unpackhi_epi64(a67b67c67d67, e67f67g67h67);

  kernel.packet[0].x = a0b0c0d0e0f0g0h0;
  kernel.packet[1].x = a1b1c1d1e1f1g1h1;
  kernel.packet[2].x = a2b2c2d2e2f2g2h2;
  kernel.packet[3].x = a3b3c3d3e3f3g3h3;
  kernel.packet[4].x = a4b4c4d4e4f4g4h4;
  kernel.packet[5].x = a5b5c5d5e5f5g5h5;
  kernel.packet[6].x = a6b6c6d6e6f6g6h6;
  kernel.packet[7].x = a7b7c7d7e7f7g7h7;
}

EIGEN_STRONG_INLINE void
ptranspose(PacketBlock<Packet8h,4>& kernel) {
  EIGEN_ALIGN32 Eigen::half in[4][8];
  pstore<Eigen::half>(in[0], kernel.packet[0]);
  pstore<Eigen::half>(in[1], kernel.packet[1]);
  pstore<Eigen::half>(in[2], kernel.packet[2]);
  pstore<Eigen::half>(in[3], kernel.packet[3]);

  EIGEN_ALIGN32 Eigen::half out[4][8];

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      out[i][j] = in[j][2*i];
    }
    for (int j = 0; j < 4; ++j) {
      out[i][j+4] = in[j][2*i+1];
    }
  }

  kernel.packet[0] = pload<Packet8h>(out[0]);
  kernel.packet[1] = pload<Packet8h>(out[1]);
  kernel.packet[2] = pload<Packet8h>(out[2]);
  kernel.packet[3] = pload<Packet8h>(out[3]);
}


// Disable the following code since it's broken on too many platforms / compilers.
//#elif defined(EIGEN_VECTORIZE_SSE) && (!EIGEN_ARCH_x86_64) && (!EIGEN_COMP_MSVC)
#elif 0

typedef struct {
  __m64 x;
} Packet4h;


template<> struct is_arithmetic<Packet4h> { enum { value = true }; };

template <>
struct packet_traits<Eigen::half> : default_packet_traits {
  typedef Packet4h type;
  // There is no half-size packet for Packet4h.
  typedef Packet4h half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 4,
    HasHalfPacket = 0,
    HasAdd    = 0,
    HasSub    = 0,
    HasMul    = 0,
    HasNegate = 0,
    HasAbs    = 0,
    HasAbs2   = 0,
    HasMin    = 0,
    HasMax    = 0,
    HasConj   = 0,
    HasSetLinear = 0,
    HasDiv = 0,
    HasSqrt = 0,
    HasRsqrt = 0,
    HasExp = 0,
    HasLog = 0,
    HasBlend = 0
  };
};


template<> struct unpacket_traits<Packet4h> { typedef Eigen::half type; enum {size=4, alignment=Aligned16}; typedef Packet4h half; };

template<> EIGEN_STRONG_INLINE Packet4h pset1<Packet4h>(const Eigen::half& from) {
  Packet4h result;
  result.x = _mm_set1_pi16(from.x);
  return result;
}

template<> EIGEN_STRONG_INLINE Eigen::half pfirst<Packet4h>(const Packet4h& from) {
  return half_impl::raw_uint16_to_half(static_cast<unsigned short>(_mm_cvtsi64_si32(from.x)));
}

template<> EIGEN_STRONG_INLINE Packet4h pconj(const Packet4h& a) { return a; }

template<> EIGEN_STRONG_INLINE Packet4h padd<Packet4h>(const Packet4h& a, const Packet4h& b) {
  __int64_t a64 = _mm_cvtm64_si64(a.x);
  __int64_t b64 = _mm_cvtm64_si64(b.x);

  Eigen::half h[4];

  Eigen::half ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64));
  Eigen::half hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64));
  h[0] = ha + hb;
  ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64 >> 16));
  hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64 >> 16));
  h[1] = ha + hb;
  ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64 >> 32));
  hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64 >> 32));
  h[2] = ha + hb;
  ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64 >> 48));
  hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64 >> 48));
  h[3] = ha + hb;
  Packet4h result;
  result.x = _mm_set_pi16(h[3].x, h[2].x, h[1].x, h[0].x);
  return result;
}

template<> EIGEN_STRONG_INLINE Packet4h pmul<Packet4h>(const Packet4h& a, const Packet4h& b) {
  __int64_t a64 = _mm_cvtm64_si64(a.x);
  __int64_t b64 = _mm_cvtm64_si64(b.x);

  Eigen::half h[4];

  Eigen::half ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64));
  Eigen::half hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64));
  h[0] = ha * hb;
  ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64 >> 16));
  hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64 >> 16));
  h[1] = ha * hb;
  ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64 >> 32));
  hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64 >> 32));
  h[2] = ha * hb;
  ha = half_impl::raw_uint16_to_half(static_cast<unsigned short>(a64 >> 48));
  hb = half_impl::raw_uint16_to_half(static_cast<unsigned short>(b64 >> 48));
  h[3] = ha * hb;
  Packet4h result;
  result.x = _mm_set_pi16(h[3].x, h[2].x, h[1].x, h[0].x);
  return result;
}

template<> EIGEN_STRONG_INLINE Packet4h pload<Packet4h>(const Eigen::half* from) {
  Packet4h result;
  result.x = _mm_cvtsi64_m64(*reinterpret_cast<const __int64_t*>(from));
  return result;
}

template<> EIGEN_STRONG_INLINE Packet4h ploadu<Packet4h>(const Eigen::half* from) {
  Packet4h result;
  result.x = _mm_cvtsi64_m64(*reinterpret_cast<const __int64_t*>(from));
  return result;
}

template<> EIGEN_STRONG_INLINE void pstore<Eigen::half>(Eigen::half* to, const Packet4h& from) {
  __int64_t r = _mm_cvtm64_si64(from.x);
  *(reinterpret_cast<__int64_t*>(to)) = r;
}

template<> EIGEN_STRONG_INLINE void pstoreu<Eigen::half>(Eigen::half* to, const Packet4h& from) {
  __int64_t r = _mm_cvtm64_si64(from.x);
  *(reinterpret_cast<__int64_t*>(to)) = r;
}

template<> EIGEN_STRONG_INLINE Packet4h
ploadquad<Packet4h>(const Eigen::half* from) {
  return pset1<Packet4h>(*from);
}

template<> EIGEN_STRONG_INLINE Packet4h pgather<Eigen::half, Packet4h>(const Eigen::half* from, Index stride)
{
  Packet4h result;
  result.x = _mm_set_pi16(from[3*stride].x, from[2*stride].x, from[1*stride].x, from[0*stride].x);
  return result;
}

template<> EIGEN_STRONG_INLINE void pscatter<Eigen::half, Packet4h>(Eigen::half* to, const Packet4h& from, Index stride)
{
  __int64_t a = _mm_cvtm64_si64(from.x);
  to[stride*0].x = static_cast<unsigned short>(a);
  to[stride*1].x = static_cast<unsigned short>(a >> 16);
  to[stride*2].x = static_cast<unsigned short>(a >> 32);
  to[stride*3].x = static_cast<unsigned short>(a >> 48);
}

EIGEN_STRONG_INLINE void
ptranspose(PacketBlock<Packet4h,4>& kernel) {
  __m64 T0 = _mm_unpacklo_pi16(kernel.packet[0].x, kernel.packet[1].x);
  __m64 T1 = _mm_unpacklo_pi16(kernel.packet[2].x, kernel.packet[3].x);
  __m64 T2 = _mm_unpackhi_pi16(kernel.packet[0].x, kernel.packet[1].x);
  __m64 T3 = _mm_unpackhi_pi16(kernel.packet[2].x, kernel.packet[3].x);

  kernel.packet[0].x = _mm_unpacklo_pi32(T0, T1);
  kernel.packet[1].x = _mm_unpackhi_pi32(T0, T1);
  kernel.packet[2].x = _mm_unpacklo_pi32(T2, T3);
  kernel.packet[3].x = _mm_unpackhi_pi32(T2, T3);
}

#endif

}
}

#endif // EIGEN_PACKET_MATH_HALF_CUDA_H
