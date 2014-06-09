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
class TensorBase<Derived, ReadOnlyAccessors>
{
  public:
    typedef typename internal::traits<Derived>::Scalar Scalar;
    typedef typename internal::traits<Derived>::Index Index;
    typedef Scalar CoeffReturnType;
    typedef typename internal::packet_traits<Scalar>::type PacketReturnType;

    // Nullary operators
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseNullaryOp<internal::scalar_constant_op<Scalar>, const Derived>
    constant(const Scalar& value) const {
      return TensorCwiseNullaryOp<internal::scalar_constant_op<Scalar>, const Derived>
          (derived(), internal::scalar_constant_op<Scalar>(value));
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseNullaryOp<internal::scalar_random_op<Scalar>, const Derived>
    random() const {
      return TensorCwiseNullaryOp<internal::scalar_random_op<Scalar>, const Derived>(derived());
    }

    // Coefficient-wise unary operators
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_opposite_op<Scalar>, const Derived>
    operator-() const { return derived(); }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_sqrt_op<Scalar>, const Derived>
    sqrt() const { return derived(); }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_square_op<Scalar>, const Derived>
    square() const { return derived(); }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_inverse_op<Scalar>, const Derived>
    inverse() const { return derived(); }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_exp_op<Scalar>, const Derived>
    exp() const { return derived(); }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_log_op<Scalar>, const Derived>
    log() const { return derived(); }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_abs_op<Scalar>, const Derived>
    abs() const { return derived(); }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_pow_op<Scalar>, const Derived>
    pow(Scalar exponent) const {
      return TensorCwiseUnaryOp<internal::scalar_pow_op<Scalar>, const Derived>
          (derived(), internal::scalar_pow_op<Scalar>(exponent));
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_multiple_op<Scalar>, const Derived>
    operator * (Scalar scale) const {
      return TensorCwiseUnaryOp<internal::scalar_multiple_op<Scalar>, const Derived>
          (derived(), internal::scalar_multiple_op<Scalar>(scale));
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseBinaryOp<internal::scalar_max_op<Scalar>, const Derived, const TensorCwiseNullaryOp<internal::scalar_constant_op<Scalar>, const Derived> >
    cwiseMax(Scalar threshold) const {
      return cwiseMax(constant(threshold));
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseBinaryOp<internal::scalar_min_op<Scalar>, const Derived, const TensorCwiseNullaryOp<internal::scalar_constant_op<Scalar>, const Derived> >
    cwiseMin(Scalar threshold) const {
      return cwiseMin(constant(threshold));
    }

    template <typename CustomUnaryOp> EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<CustomUnaryOp, const Derived>
    unaryExpr(const CustomUnaryOp& func) const {
      return TensorCwiseUnaryOp<CustomUnaryOp, const Derived>(derived(), func);
    }

    template <typename NewType> EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_cast_op<Scalar, NewType>, const Derived>
    cast() const {
      return derived();
    }

    // Coefficient-wise binary operators.
    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<internal::scalar_sum_op<Scalar>, const Derived, const OtherDerived>
    operator+(const OtherDerived& other) const {
      return TensorCwiseBinaryOp<internal::scalar_sum_op<Scalar>, const Derived, const OtherDerived>(derived(), other.derived());
    }

    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<internal::scalar_difference_op<Scalar>, const Derived, const OtherDerived>
    operator-(const OtherDerived& other) const {
      return TensorCwiseBinaryOp<internal::scalar_difference_op<Scalar>, const Derived, const OtherDerived>(derived(), other.derived());
    }

    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<internal::scalar_product_op<Scalar>, const Derived, const OtherDerived>
    operator*(const OtherDerived& other) const {
      return TensorCwiseBinaryOp<internal::scalar_product_op<Scalar>, const Derived, const OtherDerived>(derived(), other.derived());
    }

    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<internal::scalar_quotient_op<Scalar>, const Derived, const OtherDerived>
    operator/(const OtherDerived& other) const {
      return TensorCwiseBinaryOp<internal::scalar_quotient_op<Scalar>, const Derived, const OtherDerived>(derived(), other.derived());
    }

    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<internal::scalar_max_op<Scalar>, const Derived, const OtherDerived>
    cwiseMax(const OtherDerived& other) const {
      return TensorCwiseBinaryOp<internal::scalar_max_op<Scalar>, const Derived, const OtherDerived>(derived(), other.derived());
    }

    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<internal::scalar_min_op<Scalar>, const Derived, const OtherDerived>
    cwiseMin(const OtherDerived& other) const {
      return TensorCwiseBinaryOp<internal::scalar_min_op<Scalar>, const Derived, const OtherDerived>(derived(), other.derived());
    }

    // Comparisons and tests.
    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<std::less<Scalar>, const Derived, const OtherDerived>
    operator<(const OtherDerived& other) const {
      return TensorCwiseBinaryOp<std::less<Scalar>, const Derived, const OtherDerived>(derived(), other.derived());
    }
    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<std::less_equal<Scalar>, const Derived, const OtherDerived>
    operator<=(const OtherDerived& other) const {
      return TensorCwiseBinaryOp<std::less_equal<Scalar>, const Derived, const OtherDerived>(derived(), other.derived());
    }
    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<std::greater<Scalar>, const Derived, const OtherDerived>
    operator>(const OtherDerived& other) const {
      return TensorCwiseBinaryOp<std::greater<Scalar>, const Derived, const OtherDerived>(derived(), other.derived());
    }
    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<std::greater_equal<Scalar>, const Derived, const OtherDerived>
    operator>=(const OtherDerived& other) const {
      return TensorCwiseBinaryOp<std::greater_equal<Scalar>, const Derived, const OtherDerived>(derived(), other.derived());
    }

    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<std::equal_to<Scalar>, const Derived, const OtherDerived>
    operator==(const OtherDerived& other) const {
      return TensorCwiseBinaryOp<std::equal_to<Scalar>, const Derived, const OtherDerived>(derived(), other.derived());
    }
    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<std::not_equal_to<Scalar>, const Derived, const OtherDerived>
    operator!=(const OtherDerived& other) const {
      return TensorCwiseBinaryOp<std::not_equal_to<Scalar>, const Derived, const OtherDerived>(derived(), other.derived());
    }

    // Contractions.
    typedef std::pair<Index, Index> DimensionPair;

    template<typename OtherDerived, typename Dimensions> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorContractionOp<const Dimensions, const Derived, const OtherDerived>
    contract(const OtherDerived& other, const Dimensions& dims) const {
      return TensorContractionOp<const Dimensions, const Derived, const OtherDerived>(derived(), other.derived(), dims);
    }

    // Convolutions.
    template<typename KernelDerived, typename Dimensions> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorConvolutionOp<const Dimensions, const Derived, const KernelDerived>
    convolve(const KernelDerived& kernel, const Dimensions& dims) const {
      return TensorConvolutionOp<const Dimensions, const Derived, const KernelDerived>(derived(), kernel.derived(), dims);
    }

    // Coefficient-wise ternary operators.
    template<typename ThenDerived, typename ElseDerived>
    inline const TensorSelectOp<const Derived, const ThenDerived, const ElseDerived>
    select(const ThenDerived& thenTensor, const ElseDerived& elseTensor) const {
      return TensorSelectOp<const Derived, const ThenDerived, const ElseDerived>(derived(), thenTensor.derived(), elseTensor.derived());
    }

    // Morphing operators (slicing tbd).
    template <typename NewDimensions>
    inline const TensorReshapingOp<const Derived, const NewDimensions>
    reshape(const NewDimensions& newDimensions) const {
      return TensorReshapingOp<const Derived, const NewDimensions>(derived(), newDimensions);
    }

  protected:
    template <typename OtherDerived, int AccessLevel> friend class TensorBase;
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const Derived& derived() const { return *static_cast<const Derived*>(this); }
};


template<typename Derived>
class TensorBase<Derived, WriteAccessors> : public TensorBase<Derived, ReadOnlyAccessors> {
 public:
    typedef typename internal::traits<Derived>::Scalar Scalar;
    typedef typename internal::traits<Derived>::Index Index;
    typedef Scalar CoeffReturnType;
    typedef typename internal::packet_traits<Scalar>::type PacketReturnType;

    template <typename OtherDerived, int AccessLevel> friend class TensorBase;

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Derived& setZero() {
      return setConstant(Scalar(0));
    }
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Derived& setConstant(const Scalar& val) {
      return derived() = this->constant(val);
    }
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Derived& setRandom() {
      return derived() = this->random();
    }

    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    Derived& operator+=(const OtherDerived& other) {
      return derived() = TensorCwiseBinaryOp<internal::scalar_sum_op<Scalar>, const Derived, const OtherDerived>(derived(), other.derived());
    }

    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    Derived& operator-=(const OtherDerived& other) {
      return derived() = TensorCwiseBinaryOp<internal::scalar_difference_op<Scalar>, const Derived, const OtherDerived>(derived(), other.derived());
    }

    // Select the device on which to evaluate the expression.
    template <typename DeviceType>
    TensorDevice<Derived, DeviceType> device(const DeviceType& device) {
      return TensorDevice<Derived, DeviceType>(device, derived());
    }

 protected:
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Derived& derived() { return *static_cast<Derived*>(this); }
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const Derived& derived() const { return *static_cast<const Derived*>(this); }
};

} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_BASE_H
