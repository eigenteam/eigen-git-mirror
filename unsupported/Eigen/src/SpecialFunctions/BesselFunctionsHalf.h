// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_BESSELFUNCTIONS_HALF_H
#define EIGEN_BESSELFUNCTIONS_HALF_H

namespace Eigen {
namespace numext {

#if EIGEN_HAS_C99_MATH
template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half i0(const Eigen::half& x) {
  return Eigen::half(Eigen::numext::i0(static_cast<float>(x)));
}
template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half i0e(const Eigen::half& x) {
  return Eigen::half(Eigen::numext::i0e(static_cast<float>(x)));
}
template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half i1(const Eigen::half& x) {
  return Eigen::half(Eigen::numext::i1(static_cast<float>(x)));
}
template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half i1e(const Eigen::half& x) {
  return Eigen::half(Eigen::numext::i1e(static_cast<float>(x)));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half j0(const Eigen::half& x) {
  return Eigen::half(Eigen::numext::j0(static_cast<float>(x)));
}
template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half j1(const Eigen::half& x) {
  return Eigen::half(Eigen::numext::j1(static_cast<float>(x)));
}
template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half y0(const Eigen::half& x) {
  return Eigen::half(Eigen::numext::y0(static_cast<float>(x)));
}
template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half y1(const Eigen::half& x) {
  return Eigen::half(Eigen::numext::y1(static_cast<float>(x)));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half k0(const Eigen::half& x) {
  return Eigen::half(Eigen::numext::k0(static_cast<float>(x)));
}
template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half k0e(const Eigen::half& x) {
  return Eigen::half(Eigen::numext::k0e(static_cast<float>(x)));
}
template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half k1(const Eigen::half& x) {
  return Eigen::half(Eigen::numext::k1(static_cast<float>(x)));
}
template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half k1e(const Eigen::half& x) {
  return Eigen::half(Eigen::numext::k1e(static_cast<float>(x)));
}
#endif

}  // end namespace numext
}  // end namespace Eigen

#endif  // EIGEN_BESSELFUNCTIONS_HALF_H
