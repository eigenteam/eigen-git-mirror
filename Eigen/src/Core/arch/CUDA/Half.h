// Standard 16-bit float type, mostly useful for GPUs. Defines a new
// class Eigen::half (inheriting from CUDA's __half struct) with
// operator overloads such that it behaves basically as an arithmetic
// type. It will be quite slow on CPUs (so it is recommended to stay
// in fp32 for CPUs, except for simple parameter conversions, I/O
// to disk and the likes), but fast on GPUs.
//
//
// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// The conversion routines are Copyright (c) Fabian Giesen, 2016.
// The original license follows:
//
// Copyright (c) Fabian Giesen, 2016
// All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef EIGEN_HALF_CUDA_H
#define EIGEN_HALF_CUDA_H

#if __cplusplus > 199711L
#define EIGEN_EXPLICIT_CAST(tgt_type) explicit operator tgt_type()
#else
#define EIGEN_EXPLICIT_CAST(tgt_type) operator tgt_type()
#endif


#if !defined(EIGEN_HAS_CUDA_FP16)

// Make our own __half definition that is similar to CUDA's.
struct __half {
  EIGEN_DEVICE_FUNC __half() {}
  explicit EIGEN_DEVICE_FUNC __half(unsigned short raw) : x(raw) {}
  unsigned short x;
};

#endif

namespace Eigen {

namespace internal {

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC __half raw_uint16_to_half(unsigned short x);
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC __half float_to_half_rtne(float ff);
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC float half_to_float(__half h);

} // end namespace internal

// Class definition.
struct half : public __half {
  EIGEN_DEVICE_FUNC half() {}

  EIGEN_DEVICE_FUNC half(const __half& h) : __half(h) {}
  EIGEN_DEVICE_FUNC half(const half& h) : __half(h) {}

  explicit EIGEN_DEVICE_FUNC half(bool b)
      : __half(internal::raw_uint16_to_half(b ? 0x3c00 : 0)) {}
  template<class T>
  explicit EIGEN_DEVICE_FUNC half(const T& val)
      : __half(internal::float_to_half_rtne(static_cast<float>(val))) {}
  explicit EIGEN_DEVICE_FUNC half(float f)
      : __half(internal::float_to_half_rtne(f)) {}

  EIGEN_DEVICE_FUNC EIGEN_EXPLICIT_CAST(bool) const {
    // +0.0 and -0.0 become false, everything else becomes true.
    return (x & 0x7fff) != 0;
  }
  EIGEN_DEVICE_FUNC EIGEN_EXPLICIT_CAST(signed char) const {
    return static_cast<signed char>(internal::half_to_float(*this));
  }
  EIGEN_DEVICE_FUNC EIGEN_EXPLICIT_CAST(unsigned char) const {
    return static_cast<unsigned char>(internal::half_to_float(*this));
  }
  EIGEN_DEVICE_FUNC EIGEN_EXPLICIT_CAST(short) const {
    return static_cast<short>(internal::half_to_float(*this));
  }
  EIGEN_DEVICE_FUNC EIGEN_EXPLICIT_CAST(unsigned short) const {
    return static_cast<unsigned short>(internal::half_to_float(*this));
  }
  EIGEN_DEVICE_FUNC EIGEN_EXPLICIT_CAST(int) const {
    return static_cast<int>(internal::half_to_float(*this));
  }
  EIGEN_DEVICE_FUNC EIGEN_EXPLICIT_CAST(unsigned int) const {
    return static_cast<unsigned int>(internal::half_to_float(*this));
  }
  EIGEN_DEVICE_FUNC EIGEN_EXPLICIT_CAST(long) const {
    return static_cast<long>(internal::half_to_float(*this));
  }
  EIGEN_DEVICE_FUNC EIGEN_EXPLICIT_CAST(unsigned long) const {
    return static_cast<unsigned long>(internal::half_to_float(*this));
  }
  EIGEN_DEVICE_FUNC EIGEN_EXPLICIT_CAST(long long) const {
    return static_cast<long long>(internal::half_to_float(*this));
  }
  EIGEN_DEVICE_FUNC EIGEN_EXPLICIT_CAST(unsigned long long) const {
    return static_cast<unsigned long long>(internal::half_to_float(*this));
  }
  EIGEN_DEVICE_FUNC EIGEN_EXPLICIT_CAST(float) const {
    return internal::half_to_float(*this);
  }
  EIGEN_DEVICE_FUNC EIGEN_EXPLICIT_CAST(double) const {
    return static_cast<double>(internal::half_to_float(*this));
  }

  EIGEN_DEVICE_FUNC half& operator=(const half& other) {
    x = other.x;
    return *this;
  }
};

#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530

// Intrinsics for native fp16 support. Note that on current hardware,
// these are no faster than fp32 arithmetic (you need to use the half2
// versions to get the ALU speed increased), but you do save the
// conversion steps back and forth.

__device__ half operator + (const half& a, const half& b) {
  return __hadd(a, b);
}
__device__ half operator * (const half& a, const half& b) {
  return __hmul(a, b);
}
__device__ half operator - (const half& a, const half& b) {
  return __hsub(a, b);
}
__device__ half operator / (const half& a, const half& b) {
  float num = __half2float(a);
  float denom = __half2float(b);
  return __float2half(num / denom);
}
__device__ half operator - (const half& a) {
  return __hneg(a);
}
__device__ half& operator += (half& a, const half& b) {
  a = a + b;
  return a;
}
__device__ half& operator *= (half& a, const half& b) {
  a = a * b;
  return a;
}
__device__ half& operator -= (half& a, const half& b) {
  a = a - b;
  return a;
}
__device__ half& operator /= (half& a, const half& b) {
  a = a / b;
  return a;
}
__device__ bool operator == (const half& a, const half& b) {
  return __heq(a, b);
}
__device__ bool operator != (const half& a, const half& b) {
  return __hne(a, b);
}
__device__ bool operator < (const half& a, const half& b) {
  return __hlt(a, b);
}
__device__ bool operator <= (const half& a, const half& b) {
  return __hle(a, b);
}
__device__ bool operator > (const half& a, const half& b) {
  return __hgt(a, b);
}
__device__ bool operator >= (const half& a, const half& b) {
  return __hge(a, b);
}

#else  // Emulate support for half floats

// Definitions for CPUs and older CUDA, mostly working through conversion
// to/from fp32.

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half operator + (const half& a, const half& b) {
  return half(float(a) + float(b));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half operator * (const half& a, const half& b) {
  return half(float(a) * float(b));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half operator - (const half& a, const half& b) {
  return half(float(a) - float(b));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half operator / (const half& a, const half& b) {
  return half(float(a) / float(b));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half operator - (const half& a) {
  half result;
  result.x = a.x ^ 0x8000;
  return result;
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half& operator += (half& a, const half& b) {
  a = half(float(a) + float(b));
  return a;
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half& operator *= (half& a, const half& b) {
  a = half(float(a) * float(b));
  return a;
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half& operator -= (half& a, const half& b) {
  a = half(float(a) - float(b));
  return a;
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half& operator /= (half& a, const half& b) {
  a = half(float(a) / float(b));
  return a;
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator == (const half& a, const half& b) {
  return float(a) == float(b);
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator != (const half& a, const half& b) {
  return float(a) != float(b);
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator < (const half& a, const half& b) {
  return float(a) < float(b);
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator <= (const half& a, const half& b) {
  return float(a) <= float(b);
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator > (const half& a, const half& b) {
  return float(a) > float(b);
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator >= (const half& a, const half& b) {
  return float(a) >= float(b);
}

#endif  // Emulate support for half floats

// Division by an index. Do it in full float precision to avoid accuracy
// issues in converting the denominator to half.
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half operator / (const half& a, Index b) {
  return Eigen::half(static_cast<float>(a) / static_cast<float>(b));
}

// Conversion routines, including fallbacks for the host or older CUDA.
// Note that newer Intel CPUs (Haswell or newer) have vectorized versions of
// these in hardware. If we need more performance on older/other CPUs, they are
// also possible to vectorize directly.

namespace internal {

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC __half raw_uint16_to_half(unsigned short x) {
  __half h;
  h.x = x;
  return h;
}

union FP32 {
  unsigned int u;
  float f;
};

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC __half float_to_half_rtne(float ff) {
#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
  return __float2half(ff);

#elif defined(EIGEN_HAS_FP16_C)
  __half h;
  h.x = _cvtss_sh(ff, 0);
  return h;

#else
  FP32 f; f.f = ff;

  const FP32 f32infty = { 255 << 23 };
  const FP32 f16max = { (127 + 16) << 23 };
  const FP32 denorm_magic = { ((127 - 15) + (23 - 10) + 1) << 23 };
  unsigned int sign_mask = 0x80000000u;
  __half o;
  o.x = static_cast<unsigned short>(0x0u);

  unsigned int sign = f.u & sign_mask;
  f.u ^= sign;

  // NOTE all the integer compares in this function can be safely
  // compiled into signed compares since all operands are below
  // 0x80000000. Important if you want fast straight SSE2 code
  // (since there's no unsigned PCMPGTD).

  if (f.u >= f16max.u) {  // result is Inf or NaN (all exponent bits set)
    o.x = (f.u > f32infty.u) ? 0x7e00 : 0x7c00; // NaN->qNaN and Inf->Inf
  } else {  // (De)normalized number or zero
    if (f.u < (113 << 23)) {  // resulting FP16 is subnormal or zero
      // use a magic value to align our 10 mantissa bits at the bottom of
      // the float. as long as FP addition is round-to-nearest-even this
      // just works.
      f.f += denorm_magic.f;

      // and one integer subtract of the bias later, we have our final float!
      o.x = static_cast<unsigned short>(f.u - denorm_magic.u);
    } else {
      unsigned int mant_odd = (f.u >> 13) & 1; // resulting mantissa is odd

      // update exponent, rounding bias part 1
      f.u += ((unsigned int)(15 - 127) << 23) + 0xfff;
      // rounding bias part 2
      f.u += mant_odd;
      // take the bits!
      o.x = static_cast<unsigned short>(f.u >> 13);
    }
  }

  o.x |= static_cast<unsigned short>(sign >> 16);
  return o;
#endif
}

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC float half_to_float(__half h) {
#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
  return __half2float(h);

#elif defined(EIGEN_HAS_FP16_C)
  return _cvtsh_ss(h.x);

#else
  const FP32 magic = { 113 << 23 };
  const unsigned int shifted_exp = 0x7c00 << 13; // exponent mask after shift
  FP32 o;

  o.u = (h.x & 0x7fff) << 13;             // exponent/mantissa bits
  unsigned int exp = shifted_exp & o.u;   // just the exponent
  o.u += (127 - 15) << 23;                // exponent adjust

  // handle exponent special cases
  if (exp == shifted_exp) {     // Inf/NaN?
    o.u += (128 - 16) << 23;    // extra exp adjust
  } else if (exp == 0) {        // Zero/Denormal?
    o.u += 1 << 23;             // extra exp adjust
    o.f -= magic.f;             // renormalize
  }

  o.u |= (h.x & 0x8000) << 16;    // sign bit
  return o.f;
#endif
}

} // end namespace internal

// Traits.

namespace internal {

template<> struct is_arithmetic<half> { enum { value = true }; };

} // end namespace internal

template<> struct NumTraits<Eigen::half>
    : GenericNumTraits<Eigen::half>
{
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Eigen::half epsilon() {
    return internal::raw_uint16_to_half(0x0800);
  }
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Eigen::half dummy_precision() { return half(1e-2f); }
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Eigen::half highest() {
    return internal::raw_uint16_to_half(0x7bff);
  }
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Eigen::half lowest() {
    return internal::raw_uint16_to_half(0xfbff);
  }
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Eigen::half infinity() {
    return internal::raw_uint16_to_half(0x7c00);
  }
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Eigen::half quiet_NaN() {
    return internal::raw_uint16_to_half(0x7c01);
  }
};

// Infinity/NaN checks.

namespace numext {

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool (isinf)(const Eigen::half& a) {
  return (a.x & 0x7fff) == 0x7c00;
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool (isnan)(const Eigen::half& a) {
#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hisnan(a);
#else
  return (a.x & 0x7fff) > 0x7c00;
#endif
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool (isfinite)(const Eigen::half& a) {
  return !(Eigen::numext::isinf)(a) && !(Eigen::numext::isnan)(a);
}

template<> EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half abs(const Eigen::half& a) {
  Eigen::half result;
  result.x = a.x & 0x7FFF;
  return result;
}
template<> EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half exp(const Eigen::half& a) {
  return Eigen::half(::expf(float(a)));
}
template<> EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half log(const Eigen::half& a) {
  return Eigen::half(::logf(float(a)));
}
template<> EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half sqrt(const Eigen::half& a) {
  return Eigen::half(::sqrtf(float(a)));
}
template<> EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half pow(const Eigen::half& a, const Eigen::half& b) {
  return Eigen::half(::powf(float(a), float(b)));
}
template<> EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half sin(const Eigen::half& a) {
  return Eigen::half(::sinf(float(a)));
}
template<> EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half cos(const Eigen::half& a) {
  return Eigen::half(::cosf(float(a)));
}
template<> EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half tan(const Eigen::half& a) {
  return Eigen::half(::tanf(float(a)));
}
template<> EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half tanh(const Eigen::half& a) {
  return Eigen::half(::tanhf(float(a)));
}
template<> EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half floor(const Eigen::half& a) {
  return Eigen::half(::floorf(float(a)));
}
template<> EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half ceil(const Eigen::half& a) {
  return Eigen::half(::ceilf(float(a)));
}

template <> EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half mini(const Eigen::half& a, const Eigen::half& b) {
#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hlt(b, a) ? b : a;
#else
  const float f1 = static_cast<float>(a);
  const float f2 = static_cast<float>(b);
  return f2 < f1 ? b : a;
#endif
}
template <> EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half maxi(const Eigen::half& a, const Eigen::half& b) {
#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hlt(a, b) ? b : a;
#else
  const float f1 = static_cast<float>(a);
  const float f2 = static_cast<float>(b);
  return f1 < f2 ? b : a;
#endif
}

#if EIGEN_HAS_C99_MATH
template<> EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half lgamma(const Eigen::half& a) {
  return Eigen::half(Eigen::numext::lgamma(static_cast<float>(a)));
}
template<> EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half digamma(const Eigen::half& a) {
  return Eigen::half(Eigen::numext::digamma(static_cast<float>(a)));
}
template<> EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half zeta(const Eigen::half& x, const Eigen::half& q) {
  return Eigen::half(Eigen::numext::zeta(static_cast<float>(x), static_cast<float>(q)));
}
template<> EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half polygamma(const Eigen::half& n, const Eigen::half& x) {
  return Eigen::half(Eigen::numext::polygamma(static_cast<float>(n), static_cast<float>(x)));
}
template<> EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half erf(const Eigen::half& a) {
  return Eigen::half(Eigen::numext::erf(static_cast<float>(a)));
}
template<> EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half erfc(const Eigen::half& a) {
  return Eigen::half(Eigen::numext::erfc(static_cast<float>(a)));
}
template<> EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half igamma(const Eigen::half& a, const Eigen::half& x) {
  return Eigen::half(Eigen::numext::igamma(static_cast<float>(a), static_cast<float>(x)));
}
template<> EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half igammac(const Eigen::half& a, const Eigen::half& x) {
  return Eigen::half(Eigen::numext::igammac(static_cast<float>(a), static_cast<float>(x)));
}
#endif
} // end namespace numext

} // end namespace Eigen

// Standard mathematical functions and trancendentals.
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half fabsh(const Eigen::half& a) {
  Eigen::half result;
  result.x = a.x & 0x7FFF;
  return result;
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half exph(const Eigen::half& a) {
  return Eigen::half(::expf(float(a)));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half logh(const Eigen::half& a) {
  return Eigen::half(::logf(float(a)));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half sqrth(const Eigen::half& a) {
  return Eigen::half(::sqrtf(float(a)));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half powh(const Eigen::half& a, const Eigen::half& b) {
  return Eigen::half(::powf(float(a), float(b)));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half floorh(const Eigen::half& a) {
  return Eigen::half(::floorf(float(a)));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half ceilh(const Eigen::half& a) {
  return Eigen::half(::ceilf(float(a)));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC int (isnan)(const Eigen::half& a) {
  return (Eigen::numext::isnan)(a);
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC int (isinf)(const Eigen::half& a) {
  return (Eigen::numext::isinf)(a);
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC int (isfinite)(const Eigen::half& a) {
  return !(Eigen::numext::isinf)(a) && !(Eigen::numext::isnan)(a);
}


namespace std {

EIGEN_ALWAYS_INLINE ostream& operator << (ostream& os, const Eigen::half& v) {
  os << static_cast<float>(v);
  return os;
}

#if __cplusplus > 199711L
template <>
struct hash<Eigen::half> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::size_t operator()(const Eigen::half& a) const {
    return static_cast<std::size_t>(a.x);
  }
};
#endif

} // end namespace std


// Add the missing shfl_xor intrinsic
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
__device__ EIGEN_STRONG_INLINE Eigen::half __shfl_xor(Eigen::half var, int laneMask, int width=warpSize) {
  return static_cast<Eigen::half>(__shfl_xor(static_cast<float>(var), laneMask, width));
}
#endif

// ldg() has an overload for __half, but we also need one for Eigen::half.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half __ldg(const Eigen::half* ptr) {
  return Eigen::internal::raw_uint16_to_half(
      __ldg(reinterpret_cast<const unsigned short*>(ptr)));
}
#endif


#endif // EIGEN_HALF_CUDA_H
