// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_INTDIV_H
#define EIGEN_CXX11_TENSOR_TENSOR_INTDIV_H


namespace Eigen {

/** \internal
  *
  * \class TensorIntDiv
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Fast integer division by a constant.
  *
  * See the paper from Granlund and Montgomery for explanation.
  *   (at http://dx.doi.org/10.1145/773473.178249)
  *
  * \sa Tensor
  */

namespace internal {

namespace {
  // Note: result is undefined if val == 0
  template <typename T>
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE int count_leading_zeros(const T val)
  {
#ifdef __CUDA_ARCH__
    return __clz(val);
#elif EIGEN_COMP_MSVC
    DWORD leading_zero = 0;
    _BitScanReverse( &leading_zero, value);
    return 31 - leading_zero;
#else
    return __builtin_clz(static_cast<uint32_t>(val));
#endif
  }
}

template <typename T>
struct TensorIntDivisor {
 public:
   EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorIntDivisor() {
    multiplier = 0;
    shift1 = 0;
    shift2 = 0;
  }

  // Must have 1 <= divider <= 2^31-1
   EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorIntDivisor(const T divider) {
    const int N = 32;
    eigen_assert(divider > 0);
    eigen_assert(divider <= (1<<(N-1)) - 1);

    // fast ln2
    const int leading_zeros = count_leading_zeros(divider);
    const int log_div = N - (leading_zeros+1);

    multiplier = (static_cast<uint64_t>(1) << (N+log_div)) / divider - (static_cast<uint64_t>(1) << N) + 1;
    shift1 = log_div > 1 ? 1 : log_div;
    shift2 = log_div > 1 ? log_div-1 : 0;
  }

  // Must have 0 <= numerator <= 2^32-1
   EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T divide(const T numerator) const {
    const int N = 32;
    eigen_assert(numerator >= 0);
    eigen_assert(numerator <= (1ull<<N) - 1);

    uint32_t t1 = (multiplier * numerator) >> 32;
    uint32_t t = (static_cast<uint32_t>(numerator) - t1) >> shift1;
    return (t1 + t) >> shift2;
  }

 private:
  uint64_t multiplier;
  int32_t shift1;
  int32_t shift2;
};


template <typename T>
static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator / (const T& numerator, const TensorIntDivisor<T>& divisor) {
  return divisor.divide(numerator);
}


} // end namespace internal
} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_INTDIV_H
