// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_BASE_H
#define EIGEN_CXX11_TENSOR_TENSOR_BASE_H

namespace Eigen {

/** \class TensorBase
  * \ingroup CXX11_Tensor_Module
  *
  * \brief The tensor base class.
  *
  * This class is the common parent of the Tensor and TensorMap class, thus
  * making it possible to use either class interchangably in expressions.
  */

template<typename Derived>
class TensorBase
{
  public:
    typedef typename internal::traits<Derived>::Scalar Scalar;
    typedef typename internal::traits<Derived>::Index Index;
    typedef Scalar CoeffReturnType;

    Derived& setZero() {
      return setConstant(Scalar(0));
    }

    Derived& setConstant(const Scalar& val) {
      Scalar* data = derived().data();
      for (int i = 0; i < derived().size(); ++i) {
        data[i] = val;
      }
      return derived();
    }

    Derived& setRandom() {
      Scalar* data = derived().data();
      for (int i = 0; i < derived().size(); ++i) {
        data[i] = internal::random_default_impl<Scalar, false, false>::run();
      }
      return derived();
    }

    // Coefficient-wise unary operators
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_opposite_op<Scalar>, const Derived>
    operator-() const { return derived(); }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_sqrt_op<Scalar>, const Derived>
    cwiseSqrt() const { return derived(); }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_abs_op<Scalar>, const Derived>
    cwiseAbs() const { return derived(); }

    // Coefficient-wise binary operators.
    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<internal::scalar_sum_op<Scalar>, const Derived, const OtherDerived>
    operator+(const OtherDerived& other) const  {
      return TensorCwiseBinaryOp<internal::scalar_sum_op<Scalar>, const Derived, const OtherDerived>(derived(), other.derived());
    }

  protected:
    template <typename OtherDerived> friend class TensorBase;
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Derived& derived() { return *static_cast<Derived*>(this); }
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const Derived& derived() const { return *static_cast<const Derived*>(this); }
};

} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_BASE_H
