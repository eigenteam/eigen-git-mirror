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
  T m_sum;
};

template <typename T> struct MaxReducer
{
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE MaxReducer() : m_max((std::numeric_limits<T>::min)()) { }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T t) {
    if (t > m_max) { m_max = t; }
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalize() const {
    return m_max;
  }

 private:
  T m_max;
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
  T m_min;
};

} // end namespace internal
} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_FUNCTORS_H
