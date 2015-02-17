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
    typedef internal::traits<Derived> DerivedTraits;
    typedef typename DerivedTraits::Scalar Scalar;
    typedef typename DerivedTraits::Index Index;
    typedef typename internal::remove_const<Scalar>::type CoeffReturnType;
    typedef typename internal::packet_traits<CoeffReturnType>::type PacketReturnType;
    static const int NumDimensions = DerivedTraits::NumDimensions;

    // Generic nullary operation support.
    template <typename CustomNullaryOp> EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseNullaryOp<CustomNullaryOp, const Derived>
    nullaryExpr(const CustomNullaryOp& func) const {
      return TensorCwiseNullaryOp<CustomNullaryOp, const Derived>(derived(), func);
    }

    // Coefficient-wise nullary operators
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseNullaryOp<internal::scalar_constant_op<Scalar>, const Derived>
    constant(const Scalar& value) const {
      return nullaryExpr(internal::scalar_constant_op<Scalar>(value));
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseNullaryOp<internal::UniformRandomGenerator<Scalar>, const Derived>
    random() const {
      return nullaryExpr(internal::UniformRandomGenerator<Scalar>());
    }
    template <typename RandomGenerator> EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseNullaryOp<RandomGenerator, const Derived>
    random() const {
      return nullaryExpr(RandomGenerator());
    }

    // Generic unary operation support.
    template <typename CustomUnaryOp> EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<CustomUnaryOp, const Derived>
    unaryExpr(const CustomUnaryOp& func) const {
      return TensorCwiseUnaryOp<CustomUnaryOp, const Derived>(derived(), func);
    }

    // Coefficient-wise unary operators
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_opposite_op<Scalar>, const Derived>
    operator-() const {
      return unaryExpr(internal::scalar_opposite_op<Scalar>());
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_sqrt_op<Scalar>, const Derived>
    sqrt() const {
      return unaryExpr(internal::scalar_sqrt_op<Scalar>());
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_square_op<Scalar>, const Derived>
    square() const {
      return unaryExpr(internal::scalar_square_op<Scalar>());
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_cube_op<Scalar>, const Derived>
    cube() const {
      return unaryExpr(internal::scalar_cube_op<Scalar>());
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_inverse_op<Scalar>, const Derived>
    inverse() const {
      return unaryExpr(internal::scalar_inverse_op<Scalar>());
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_exp_op<Scalar>, const Derived>
    exp() const {
      return unaryExpr(internal::scalar_exp_op<Scalar>());
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_log_op<Scalar>, const Derived>
    log() const {
      return unaryExpr(internal::scalar_log_op<Scalar>());
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_abs_op<Scalar>, const Derived>
    abs() const {
      return unaryExpr(internal::scalar_abs_op<Scalar>());
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_pow_op<Scalar>, const Derived>
    pow(Scalar exponent) const {
      return unaryExpr(internal::scalar_pow_op<Scalar>(exponent));
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_add_op<Scalar>, const Derived>
    operator+ (Scalar rhs) const {
      return unaryExpr(internal::scalar_add_op<Scalar>(rhs));
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_sub_op<Scalar>, const Derived>
    operator- (Scalar rhs) const {
      EIGEN_STATIC_ASSERT((std::numeric_limits<Scalar>::is_signed || internal::is_same<Scalar, const std::complex<float> >::value), YOU_MADE_A_PROGRAMMING_MISTAKE);
      return unaryExpr(internal::scalar_sub_op<Scalar>(rhs));
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_multiple_op<Scalar>, const Derived>
    operator* (Scalar rhs) const {
      return unaryExpr(internal::scalar_multiple_op<Scalar>(rhs));
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_quotient1_op<Scalar>, const Derived>
    operator/ (Scalar rhs) const {
      // EIGEN_STATIC_ASSERT(!std::numeric_limits<Scalar>::is_integer, YOU_MADE_A_PROGRAMMING_MISTAKE);
      return unaryExpr(internal::scalar_quotient1_op<Scalar>(rhs));
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

    template <typename NewType> EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseUnaryOp<internal::scalar_cast_op<Scalar, NewType>, const Derived>
    cast() const {
      return unaryExpr(internal::scalar_cast_op<Scalar, NewType>());
    }

    // Generic binary operation support.
    template <typename CustomBinaryOp, typename OtherDerived> EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const TensorCwiseBinaryOp<CustomBinaryOp, const Derived, const OtherDerived>
    binaryExpr(const OtherDerived& other, const CustomBinaryOp& func) const {
      return TensorCwiseBinaryOp<CustomBinaryOp, const Derived, const OtherDerived>(derived(), other, func);
    }

    // Coefficient-wise binary operators.
    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<internal::scalar_sum_op<Scalar>, const Derived, const OtherDerived>
    operator+(const OtherDerived& other) const {
      return binaryExpr(other.derived(), internal::scalar_sum_op<Scalar>());
    }

    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<internal::scalar_difference_op<Scalar>, const Derived, const OtherDerived>
    operator-(const OtherDerived& other) const {
      return binaryExpr(other.derived(), internal::scalar_difference_op<Scalar>());
    }

    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<internal::scalar_product_op<Scalar>, const Derived, const OtherDerived>
    operator*(const OtherDerived& other) const {
      return binaryExpr(other.derived(), internal::scalar_product_op<Scalar>());
    }

    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<internal::scalar_quotient_op<Scalar>, const Derived, const OtherDerived>
    operator/(const OtherDerived& other) const {
      return binaryExpr(other.derived(), internal::scalar_quotient_op<Scalar>());
    }

    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<internal::scalar_max_op<Scalar>, const Derived, const OtherDerived>
    cwiseMax(const OtherDerived& other) const {
      return binaryExpr(other.derived(), internal::scalar_max_op<Scalar>());
    }

    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<internal::scalar_min_op<Scalar>, const Derived, const OtherDerived>
    cwiseMin(const OtherDerived& other) const {
      return binaryExpr(other.derived(), internal::scalar_min_op<Scalar>());
    }

    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<internal::scalar_boolean_and_op, const Derived, const OtherDerived>
    operator&&(const OtherDerived& other) const {
      return binaryExpr(other.derived(), internal::scalar_boolean_and_op());
    }

    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<internal::scalar_boolean_or_op, const Derived, const OtherDerived>
    operator||(const OtherDerived& other) const {
      return binaryExpr(other.derived(), internal::scalar_boolean_or_op());
    }

    // Comparisons and tests.
    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<std::less<Scalar>, const Derived, const OtherDerived>
    operator<(const OtherDerived& other) const {
      return binaryExpr(other.derived(), std::less<Scalar>());
    }
    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<std::less_equal<Scalar>, const Derived, const OtherDerived>
    operator<=(const OtherDerived& other) const {
      return binaryExpr(other.derived(), std::less_equal<Scalar>());
    }
    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<std::greater<Scalar>, const Derived, const OtherDerived>
    operator>(const OtherDerived& other) const {
      return binaryExpr(other.derived(), std::greater<Scalar>());
    }
    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<std::greater_equal<Scalar>, const Derived, const OtherDerived>
    operator>=(const OtherDerived& other) const {
      return binaryExpr(other.derived(), std::greater_equal<Scalar>());
    }

    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<std::equal_to<Scalar>, const Derived, const OtherDerived>
    operator==(const OtherDerived& other) const {
      return binaryExpr(other.derived(), std::equal_to<Scalar>());
    }
    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorCwiseBinaryOp<std::not_equal_to<Scalar>, const Derived, const OtherDerived>
    operator!=(const OtherDerived& other) const {
      return binaryExpr(other.derived(), std::not_equal_to<Scalar>());
    }

    // Coefficient-wise ternary operators.
    template<typename ThenDerived, typename ElseDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorSelectOp<const Derived, const ThenDerived, const ElseDerived>
    select(const ThenDerived& thenTensor, const ElseDerived& elseTensor) const {
      return TensorSelectOp<const Derived, const ThenDerived, const ElseDerived>(derived(), thenTensor.derived(), elseTensor.derived());
    }

    // Contractions.
    typedef Eigen::IndexPair<Index> DimensionPair;

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

    // Reductions.
    template <typename Dims> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorReductionOp<internal::SumReducer<CoeffReturnType>, const Dims, const Derived>
    sum(const Dims& dims) const {
      return TensorReductionOp<internal::SumReducer<CoeffReturnType>, const Dims, const Derived>(derived(), dims, internal::SumReducer<CoeffReturnType>());
    }

    const TensorReductionOp<internal::SumReducer<CoeffReturnType>, const array<Index, NumDimensions>, const Derived>
    sum() const {
      array<Index, NumDimensions> in_dims;
      for (int i = 0; i < NumDimensions; ++i) in_dims[i] = i;
      return TensorReductionOp<internal::SumReducer<CoeffReturnType>, const array<Index, NumDimensions>, const Derived>(derived(), in_dims, internal::SumReducer<CoeffReturnType>());
    }

    template <typename Dims> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorReductionOp<internal::MeanReducer<CoeffReturnType>, const Dims, const Derived>
    mean(const Dims& dims) const {
      return TensorReductionOp<internal::MeanReducer<CoeffReturnType>, const Dims, const Derived>(derived(), dims, internal::MeanReducer<CoeffReturnType>());
    }

    const TensorReductionOp<internal::MeanReducer<CoeffReturnType>, const array<Index, NumDimensions>, const Derived>
    mean() const {
      array<Index, NumDimensions> in_dims;
      for (int i = 0; i < NumDimensions; ++i) in_dims[i] = i;
      return TensorReductionOp<internal::MeanReducer<CoeffReturnType>, const array<Index, NumDimensions>, const Derived>(derived(), in_dims, internal::MeanReducer<CoeffReturnType>());
    }

    template <typename Dims> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorReductionOp<internal::ProdReducer<CoeffReturnType>, const Dims, const Derived>
    prod(const Dims& dims) const {
      return TensorReductionOp<internal::ProdReducer<CoeffReturnType>, const Dims, const Derived>(derived(), dims, internal::ProdReducer<CoeffReturnType>());
    }

    const TensorReductionOp<internal::ProdReducer<CoeffReturnType>, const array<Index, NumDimensions>, const Derived>
    prod() const {
      array<Index, NumDimensions> in_dims;
      for (int i = 0; i < NumDimensions; ++i) in_dims[i] = i;
      return TensorReductionOp<internal::ProdReducer<CoeffReturnType>, const array<Index, NumDimensions>, const Derived>(derived(), in_dims, internal::ProdReducer<CoeffReturnType>());
    }

    template <typename Dims> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorReductionOp<internal::MaxReducer<CoeffReturnType>, const Dims, const Derived>
    maximum(const Dims& dims) const {
      return TensorReductionOp<internal::MaxReducer<CoeffReturnType>, const Dims, const Derived>(derived(), dims, internal::MaxReducer<CoeffReturnType>());
    }

    const TensorReductionOp<internal::MaxReducer<CoeffReturnType>, const array<Index, NumDimensions>, const Derived>
    maximum() const {
      array<Index, NumDimensions> in_dims;
      for (int i = 0; i < NumDimensions; ++i) in_dims[i] = i;
      return TensorReductionOp<internal::MaxReducer<CoeffReturnType>, const array<Index, NumDimensions>, const Derived>(derived(), in_dims, internal::MaxReducer<CoeffReturnType>());
    }

    template <typename Dims> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorReductionOp<internal::MinReducer<CoeffReturnType>, const Dims, const Derived>
    minimum(const Dims& dims) const {
      return TensorReductionOp<internal::MinReducer<CoeffReturnType>, const Dims, const Derived>(derived(), dims, internal::MinReducer<CoeffReturnType>());
    }

    const TensorReductionOp<internal::MinReducer<CoeffReturnType>, const array<Index, NumDimensions>, const Derived>
    minimum() const {
      array<Index, NumDimensions> in_dims;
      for (int i = 0; i < NumDimensions; ++i) in_dims[i] = i;
      return TensorReductionOp<internal::MinReducer<CoeffReturnType>, const array<Index, NumDimensions>, const Derived>(derived(), in_dims, internal::MinReducer<CoeffReturnType>());
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

    template <Index Rows, Index Cols> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorImagePatchOp<Rows, Cols, const Derived>
    extract_image_patches() const {
      return TensorImagePatchOp<Rows, Cols, const Derived>(derived(), Rows, Cols, 1, 1, PADDING_SAME);
    }

    template <Index Rows, Index Cols> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorImagePatchOp<Rows, Cols, const Derived>
    extract_image_patches(const PaddingType padding_type) const {
      return TensorImagePatchOp<Rows, Cols, const Derived>(derived(), Rows, Cols, 1, 1, padding_type);
    }

    template <Index Rows, Index Cols> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorImagePatchOp<Rows, Cols, const Derived>
    extract_image_patches(const Index stride, const PaddingType padding_type) const {
      return TensorImagePatchOp<Rows, Cols, const Derived>(derived(), Rows, Cols, stride, stride, padding_type);
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorImagePatchOp<Dynamic, Dynamic, const Derived>
    extract_image_patches(const Index patch_rows, const Index patch_cols,
                          const Index row_stride = 1, const Index col_stride = 1) const {
      return TensorImagePatchOp<Dynamic, Dynamic, const Derived>(derived(), patch_rows, patch_cols, row_stride, col_stride,
                                                                 PADDING_SAME);
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorImagePatchOp<Dynamic, Dynamic, const Derived>
    extract_image_patches(const Index patch_rows, const Index patch_cols,
                          const Index row_stride, const Index col_stride,
                          const PaddingType padding_type) const {
      return TensorImagePatchOp<Dynamic, Dynamic, const Derived>(derived(), patch_rows, patch_cols, row_stride, col_stride,
                                                                 padding_type);
    }

    // Morphing operators.
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorLayoutSwapOp<const Derived>
    swap_layout() const {
      return TensorLayoutSwapOp<const Derived>(derived());
    }
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
    template <Index DimId> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorChippingOp<DimId, const Derived>
    chip(const Index offset) const {
      return TensorChippingOp<DimId, const Derived>(derived(), offset, DimId);
    }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorChippingOp<Dynamic, const Derived>
    chip(const Index offset, const Index dim) const {
      return TensorChippingOp<Dynamic, const Derived>(derived(), offset, dim);
    }
    template <typename ReverseDimensions> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorReverseOp<const ReverseDimensions, const Derived>
    reverse(const ReverseDimensions& rev) const {
      return TensorReverseOp<const ReverseDimensions, const Derived>(derived(), rev);
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
    template <typename Scalar, int Options> friend class TensorVarDim;
    template <typename OtherDerived, int AccessLevel> friend class TensorBase;
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const Derived& derived() const { return *static_cast<const Derived*>(this); }
};

template<typename Derived>
class TensorBase<Derived, WriteAccessors> : public TensorBase<Derived, ReadOnlyAccessors> {
 public:
    typedef internal::traits<Derived> DerivedTraits;
    typedef typename DerivedTraits::Scalar Scalar;
    typedef typename DerivedTraits::Index Index;
    typedef Scalar CoeffReturnType;
    typedef typename internal::packet_traits<Scalar>::type PacketReturnType;
    static const int NumDimensions = DerivedTraits::NumDimensions;

    template <typename Scalar, std::size_t NumIndices, int Options> friend class Tensor;
    template <typename Scalar, int Options> friend class TensorVarDim;
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
    template <typename RandomGenerator> EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Derived& setRandom() {
      return derived() = this->template random<RandomGenerator>();
    }

#ifdef EIGEN_HAS_VARIADIC_TEMPLATES
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Derived& setValues(
        const typename internal::Initializer<Derived, NumDimensions>::InitList& vals) {
      TensorEvaluator<Derived, DefaultDevice> eval(derived(), DefaultDevice());
      internal::initialize_tensor<Derived, NumDimensions>(eval, vals);
      return derived();
    }
#endif  // EIGEN_HAS_VARIADIC_TEMPLATES

    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    Derived& operator+=(const OtherDerived& other) {
      return derived() = derived() + other.derived();
    }
    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    Derived& operator-=(const OtherDerived& other) {
      return derived() = derived() - other.derived();
    }
    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    Derived& operator*=(const OtherDerived& other) {
      return derived() = derived() * other.derived();
    }
    template<typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    Derived& operator/=(const OtherDerived& other) {
      return derived() = derived() / other.derived();
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    TensorLayoutSwapOp<Derived>
    swap_layout() const {
      return TensorLayoutSwapOp<Derived>(derived());
    }
    template <typename Axis, typename OtherDerived> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    TensorConcatenationOp<const Axis, Derived, OtherDerived>
    concatenate(const OtherDerived& other, const Axis& axis) const {
      return TensorConcatenationOp<const Axis, Derived, OtherDerived>(derived(), other.derived(), axis);
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
    template <DenseIndex DimId> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    TensorChippingOp<DimId, Derived>
    chip(const Index offset) const {
      return TensorChippingOp<DimId, Derived>(derived(), offset, DimId);
    }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    TensorChippingOp<Dynamic, Derived>
    chip(const Index offset, const Index dim) const {
      return TensorChippingOp<Dynamic, Derived>(derived(), offset, dim);
    }
    template <typename Shuffle> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    TensorShufflingOp<const Shuffle, Derived>
    shuffle(const Shuffle& shuffle) const {
      return TensorShufflingOp<const Shuffle, Derived>(derived(), shuffle);
    }
    template <typename Strides> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    TensorStridingOp<const Strides, Derived>
    stride(const Strides& strides) const {
      return TensorStridingOp<const Strides, Derived>(derived(), strides);
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
