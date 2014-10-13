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
    EIGEN_STRONG_INLINE const TensorCwiseNullaryOp<internal::UniformRandomGenerator<Scalar>, const Derived>
    random() const {
      return TensorCwiseNullaryOp<internal::UniformRandomGenerator<Scalar>, const Derived>(derived());
    }
    template <typename RandomGenerator> EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseNullaryOp<RandomGenerator, const Derived>
    random() const {
      return TensorCwiseNullaryOp<RandomGenerator, const Derived>(derived());
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
    template<typename ThenDerived, typename ElseDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorSelectOp<const Derived, const ThenDerived, const ElseDerived>
    select(const ThenDerived& thenTensor, const ElseDerived& elseTensor) const {
      return TensorSelectOp<const Derived, const ThenDerived, const ElseDerived>(derived(), thenTensor.derived(), elseTensor.derived());
    }

    // Reductions.
    template <typename Dims> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorReductionOp<internal::SumReducer<Scalar>, const Dims, const Derived>
    sum(const Dims& dims) const {
      return TensorReductionOp<internal::SumReducer<Scalar>, const Dims, const Derived>(derived(), dims, internal::SumReducer<Scalar>());
    }
    template <typename Dims> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorReductionOp<internal::MaxReducer<Scalar>, const Dims, const Derived>
    maximum(const Dims& dims) const {
      return TensorReductionOp<internal::MaxReducer<Scalar>, const Dims, const Derived>(derived(), dims, internal::MaxReducer<Scalar>());
    }
    template <typename Dims> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorReductionOp<internal::MinReducer<Scalar>, const Dims, const Derived>
    minimum(const Dims& dims) const {
      return TensorReductionOp<internal::MinReducer<Scalar>, const Dims, const Derived>(derived(), dims, internal::MinReducer<Scalar>());
    }
    template <typename Reducer, typename Dims> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorReductionOp<Reducer, const Dims, const Derived>
    reduce(const Dims& dims, const Reducer& reducer) const {
      return TensorReductionOp<Reducer, const Dims, const Derived>(derived(), dims, reducer);
    }

    template <typename Broadcast> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorBroadcastingOp<const Broadcast, const Derived>
    broadcast(const Broadcast& broadcast) const {
      return TensorBroadcastingOp<const Broadcast, const Derived>(derived(), broadcast);
    }

    template <typename Axis, typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorConcatenationOp<Axis, const Derived, const OtherDerived>
    concatenate(const OtherDerived& other, Axis axis) const {
      return TensorConcatenationOp<Axis, const Derived, const OtherDerived>(derived(), other.derived(), axis);
    }

    template <typename PatchDims> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorPatchOp<const PatchDims, const Derived>
    extract_patches(const PatchDims& patch_dims) const {
      return TensorPatchOp<const PatchDims, const Derived>(derived(), patch_dims);
    }

    // Morphing operators.
    template <typename NewDimensions> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorReshapingOp<const NewDimensions, const Derived>
    reshape(const NewDimensions& newDimensions) const {
      return TensorReshapingOp<const NewDimensions, const Derived>(derived(), newDimensions);
    }
    template <typename StartIndices, typename Sizes> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorSlicingOp<const StartIndices, const Sizes, const Derived>
    slice(const StartIndices& startIndices, const Sizes& sizes) const {
      return TensorSlicingOp<const StartIndices, const Sizes, const Derived>(derived(), startIndices, sizes);
    }
    template <std::size_t DimId> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorChippingOp<DimId, const Derived>
    chip(const Index offset) const {
       return TensorChippingOp<DimId, const Derived>(derived(), offset);
    }
    template <typename PaddingDimensions> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorPaddingOp<const PaddingDimensions, const Derived>
    pad(const PaddingDimensions& padding) const {
      return TensorPaddingOp<const PaddingDimensions, const Derived>(derived(), padding);
    }
    template <typename Shuffle> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorShufflingOp<const Shuffle, const Derived>
    shuffle(const Shuffle& shuffle) const {
      return TensorShufflingOp<const Shuffle, const Derived>(derived(), shuffle);
    }
    template <typename Strides> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorStridingOp<const Strides, const Derived>
    stride(const Strides& strides) const {
      return TensorStridingOp<const Strides, const Derived>(derived(), strides);
    }

    // Force the evaluation of the expression.
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorForcedEvalOp<const Derived> eval() const {
      return TensorForcedEvalOp<const Derived>(derived());
    }

  protected:
    template <typename Scalar, std::size_t NumIndices, int Options> friend class Tensor;
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

    template <typename Scalar, std::size_t NumIndices, int Options> friend class Tensor;
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
    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    Derived& operator*=(const OtherDerived& other) {
      return derived() = TensorCwiseBinaryOp<internal::scalar_product_op<Scalar>, const Derived, const OtherDerived>(derived(), other.derived());
    }
    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    Derived& operator/=(const OtherDerived& other) {
      return derived() = TensorCwiseBinaryOp<internal::scalar_quotient_op<Scalar>, const Derived, const OtherDerived>(derived(), other.derived());
    }

    template <typename NewDimensions> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    TensorReshapingOp<const NewDimensions, Derived>
    reshape(const NewDimensions& newDimensions) const {
      return TensorReshapingOp<const NewDimensions, Derived>(derived(), newDimensions);
    }
    template <typename StartIndices, typename Sizes> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    TensorSlicingOp<const StartIndices, const Sizes, Derived>
    slice(const StartIndices& startIndices, const Sizes& sizes) const {
      return TensorSlicingOp<const StartIndices, const Sizes, Derived>(derived(), startIndices, sizes);
    }
    template <std::size_t DimId> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    TensorChippingOp<DimId, Derived>
    chip(const Index offset) const {
       return TensorChippingOp<DimId, Derived>(derived(), offset);
    }
    template <typename Shuffle> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    TensorShufflingOp<const Shuffle, Derived>
    shuffle(const Shuffle& shuffle) const {
      return TensorShufflingOp<const Shuffle, Derived>(derived(), shuffle);
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
