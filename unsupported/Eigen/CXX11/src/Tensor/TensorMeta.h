// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_META_H
#define EIGEN_CXX11_TENSOR_TENSOR_META_H

namespace Eigen {

template<bool cond> struct Cond {};

template<typename T1, typename T2> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
const T1& choose(Cond<true>, const T1& first, const T2&) {
  return first;
}

template<typename T1, typename T2> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
const T2& choose(Cond<false>, const T1&, const T2& second) {
  return second;
}

template <size_t n> struct max_n_1 {
  static const size_t size = n;
};
template <> struct max_n_1<0> {
  static const size_t size = 1;
};

}  // namespace Eigen

#endif  // EIGEN_CXX11_TENSOR_TENSOR_META_H
