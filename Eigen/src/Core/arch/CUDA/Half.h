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

#if !defined(EIGEN_HAS_CUDA_FP16)

// Make our own __half definition that is similar to CUDA's.
struct __half {
  unsigned short x;
};

#endif

namespace Eigen {

namespace internal {

static inline EIGEN_DEVICE_FUNC __half raw_uint16_to_half(unsigned short x);
static inline EIGEN_DEVICE_FUNC __half float_to_half_rtne(float ff);
static inline EIGEN_DEVICE_FUNC float half_to_float(__half h);

} // end namespace internal

// Class definition.
struct half : public __half {
  EIGEN_DEVICE_FUNC half() : __half(internal::raw_uint16_to_half(0)) {}

  // TODO(sesse): Should these conversions be marked as explicit?
  EIGEN_DEVICE_FUNC half(float f) : __half(internal::float_to_half_rtne(f)) {}
  EIGEN_DEVICE_FUNC half(int i) : __half(internal::float_to_half_rtne(i)) {}
  EIGEN_DEVICE_FUNC half(double d) : __half(internal::float_to_half_rtne(d)) {}
  EIGEN_DEVICE_FUNC half(bool b)
      : __half(internal::raw_uint16_to_half(b ? 0x3c00 : 0)) {}
  EIGEN_DEVICE_FUNC half(const __half& h) : __half(h) {}
  EIGEN_DEVICE_FUNC half(const half& h) : __half(h) {}
  EIGEN_DEVICE_FUNC half(const volatile half& h)
      : __half(internal::raw_uint16_to_half(h.x)) {}

  EIGEN_DEVICE_FUNC operator float() const {
    return internal::half_to_float(*this);
  }
  EIGEN_DEVICE_FUNC operator double() const {
    return internal::half_to_float(*this);
  }

  EIGEN_DEVICE_FUNC half& operator=(const half& other) {
    x = other.x;
    return *this;
  }
  EIGEN_DEVICE_FUNC half& operator=(const volatile half& other) {
    x = other.x;
    return *this;
  }
  EIGEN_DEVICE_FUNC volatile half& operator=(const half& other) volatile {
    x = other.x;
    return *this;
  }
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530

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
  return __hle(a, b);
}
__device__ bool operator > (const half& a, const half& b) {
  return __hgt(a, b);
}

#else  // Not CUDA 530

// Definitions for CPUs and older CUDA, mostly working through conversion
// to/from fp32.

static inline EIGEN_DEVICE_FUNC half operator + (const half& a, const half& b) {
  return half(float(a) + float(b));
}
static inline EIGEN_DEVICE_FUNC half operator * (const half& a, const half& b) {
  return half(float(a) * float(b));
}
static inline EIGEN_DEVICE_FUNC half operator - (const half& a, const half& b) {
  return half(float(a) - float(b));
}
static inline EIGEN_DEVICE_FUNC half operator / (const half& a, const half& b) {
  return half(float(a) / float(b));
}
static inline EIGEN_DEVICE_FUNC half operator - (const half& a) {
  half result;
  result.x = a.x ^ 0x8000;
  return result;
}
static inline EIGEN_DEVICE_FUNC half& operator += (half& a, const half& b) {
  a = half(float(a) + float(b));
  return a;
}
static inline EIGEN_DEVICE_FUNC half& operator *= (half& a, const half& b) {
  a = half(float(a) * float(b));
  return a;
}
static inline EIGEN_DEVICE_FUNC half& operator -= (half& a, const half& b) {
  a = half(float(a) - float(b));
  return a;
}
static inline EIGEN_DEVICE_FUNC half& operator /= (half& a, const half& b) {
  a = half(float(a) / float(b));
  return a;
}
static inline EIGEN_DEVICE_FUNC bool operator == (const half& a, const half& b) {
  return float(a) == float(b);
}
static inline EIGEN_DEVICE_FUNC bool operator != (const half& a, const half& b) {
  return float(a) != float(b);
}
static inline EIGEN_DEVICE_FUNC bool operator < (const half& a, const half& b) {
  return float(a) < float(b);
}
static inline EIGEN_DEVICE_FUNC bool operator > (const half& a, const half& b) {
  return float(a) > float(b);
}

#endif // Not CUDA 530

// Conversion routines, including fallbacks for the host or older CUDA.
// Note that newer Intel CPUs (Haswell or newer) have vectorized versions of
// these in hardware. If we need more performance on older/other CPUs, they are
// also possible to vectorize directly.

namespace internal {

static inline EIGEN_DEVICE_FUNC __half raw_uint16_to_half(unsigned short x) {
  __half h;
  h.x = x;
  return h;
}

union FP32 {
  unsigned int u;
  float f;
};

static inline EIGEN_DEVICE_FUNC __half float_to_half_rtne(float ff) {
#if defined(__CUDA_ARCH__) && defined(EIGEN_HAS_CUDA_FP16)
  return __float2half(ff);
#else
  FP32 f; f.f = ff;

  const FP32 f32infty = { 255 << 23 };
  const FP32 f16max = { (127 + 16) << 23 };
  const FP32 denorm_magic = { ((127 - 15) + (23 - 10) + 1) << 23 };
  unsigned int sign_mask = 0x80000000u;
  __half o = { 0 };

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
      o.x = f.u - denorm_magic.u;
    } else {
      unsigned int mant_odd = (f.u >> 13) & 1; // resulting mantissa is odd

      // update exponent, rounding bias part 1
      f.u += ((unsigned int)(15 - 127) << 23) + 0xfff;
      // rounding bias part 2
      f.u += mant_odd;
      // take the bits!
      o.x = f.u >> 13;
    }
  }

  o.x |= sign >> 16;
  return o;
#endif
}

static inline EIGEN_DEVICE_FUNC float half_to_float(__half h) {
#if defined(__CUDA_ARCH__) && defined(EIGEN_HAS_CUDA_FP16)
  return __half2float(h);
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

// Infinity/NaN checks.

namespace numext {

static inline EIGEN_DEVICE_FUNC bool (isinf)(const Eigen::half& a) {
  return (a.x & 0x7fff) == 0x7c00;
}
static inline EIGEN_HALF_CUDA_H bool (isnan)(const Eigen::half& a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hisnan(x);
#else
  return (a.x & 0x7fff) > 0x7c00;
#endif
}

} // end namespace numext

} // end namespace Eigen

// Standard mathematical functions and trancendentals.

namespace std {

static inline EIGEN_DEVICE_FUNC Eigen::half abs(const Eigen::half& a) {
  Eigen::half result;
  result.x = a.x & 0x7FFF;
  return result;
}
static inline EIGEN_DEVICE_FUNC Eigen::half exp(const Eigen::half& a) {
  return Eigen::half(expf(float(a)));
}
static inline EIGEN_DEVICE_FUNC Eigen::half log(const Eigen::half& a) {
  return Eigen::half(logf(float(a)));
}

} // end namespace std

#endif // EIGEN_HALF_CUDA_H
