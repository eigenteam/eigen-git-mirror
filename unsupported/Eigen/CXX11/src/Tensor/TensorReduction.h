// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_H
#define EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_H

namespace Eigen {

/** \class TensorReduction
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor reduction class.
  *
  */

namespace internal {
template<typename Op, typename Dims, typename XprType>
struct traits<TensorReductionOp<Op, Dims, XprType> >
 : traits<XprType>
{
  typedef typename traits<XprType>::Scalar Scalar;
  typedef typename internal::packet_traits<Scalar>::type Packet;
  typedef typename traits<XprType>::StorageKind StorageKind;
  typedef typename traits<XprType>::Index Index;
  typedef typename XprType::Nested Nested;
};

template<typename Op, typename Dims, typename XprType>
struct eval<TensorReductionOp<Op, Dims, XprType>, Eigen::Dense>
{
  typedef const TensorReductionOp<Op, Dims, XprType>& type;
};

template<typename Op, typename Dims, typename XprType>
struct nested<TensorReductionOp<Op, Dims, XprType>, 1, typename eval<TensorReductionOp<Op, Dims, XprType> >::type>
{
  typedef TensorReductionOp<Op, Dims, XprType> type;
};

}  // end namespace internal


template <typename Op, typename Dims, typename XprType>
class TensorReductionOp : public TensorBase<TensorReductionOp<Op, Dims, XprType>, ReadOnlyAccessors> {
  public:
    typedef typename Eigen::internal::traits<TensorReductionOp>::Scalar Scalar;
    typedef typename Eigen::internal::traits<TensorReductionOp>::Packet Packet;
    typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
    typedef typename XprType::CoeffReturnType CoeffReturnType;
    typedef typename XprType::PacketReturnType PacketReturnType;
    typedef typename Eigen::internal::nested<TensorReductionOp>::type Nested;
    typedef typename Eigen::internal::traits<TensorReductionOp>::StorageKind StorageKind;
    typedef typename Eigen::internal::traits<TensorReductionOp>::Index Index;

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    TensorReductionOp(const XprType& expr, const Dims& dims) : m_expr(expr), m_dims(dims)
    { }
    TensorReductionOp(const XprType& expr, const Dims& dims, const Op& reducer) : m_expr(expr), m_dims(dims), m_reducer(reducer)
    { }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const XprType& expression() const { return m_expr; }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const Dims& dims() const { return m_dims; }
    const Op& reducer() const { return m_reducer; }

  protected:
    typename XprType::Nested m_expr;
    const Dims m_dims;
    const Op m_reducer;
};


// Eval as rvalue
template<typename Op, typename Dims, typename ArgType, typename Device>
struct TensorEvaluator<const TensorReductionOp<Op, Dims, ArgType>, Device>
{
  typedef TensorReductionOp<Op, Dims, ArgType> XprType;
  typedef typename XprType::Index Index;
  static const int NumInputDims = internal::array_size<typename TensorEvaluator<ArgType, Device>::Dimensions>::value;
  static const int NumReducedDims = internal::array_size<Dims>::value;
  static const int NumDims = (NumInputDims==NumReducedDims) ? 1 : NumInputDims - NumReducedDims;
  typedef DSizes<Index, NumDims> Dimensions;
  typedef typename XprType::Scalar Scalar;

  enum {
    IsAligned = false,
    PacketAccess = false,  // The code isn't vectorized properly yet
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
      : m_impl(op.expression(), device), m_reducer(op.reducer())
  {
    EIGEN_STATIC_ASSERT(NumInputDims >= NumReducedDims, YOU_MADE_A_PROGRAMMING_MISTAKE);

    array<bool, NumInputDims> reduced;
    for (int i = 0; i < NumInputDims; ++i) {
      reduced[i] = false;
    }
    for (int i = 0; i < NumReducedDims; ++i) {
      eigen_assert(op.dims()[i] >= 0);
      eigen_assert(op.dims()[i] < NumInputDims);
      reduced[op.dims()[i]] = true;
    }

    const typename TensorEvaluator<ArgType, Device>::Dimensions& input_dims = m_impl.dimensions();
    int outputIndex = 0;
    int reduceIndex = 0;
    for (int i = 0; i < NumInputDims; ++i) {
      if (reduced[i]) {
        m_reducedDims[reduceIndex] = input_dims[i];
        ++reduceIndex;
      } else {
        m_dimensions[outputIndex] = input_dims[i];
        ++outputIndex;
      }
    }

    m_outputStrides[0] = 1;
    for (int i = 1; i < NumDims; ++i) {
      m_outputStrides[i] = m_outputStrides[i-1] * m_dimensions[i-1];
    }

    array<Index, NumInputDims> strides;
    strides[0] = 1;
    for (int i = 1; i < NumInputDims; ++i) {
      strides[i] = strides[i-1] * input_dims[i-1];
    }
    outputIndex = 0;
    reduceIndex = 0;
    for (int i = 0; i < NumInputDims; ++i) {
      if (reduced[i]) {
        m_reducedStrides[reduceIndex] = strides[i];
        ++reduceIndex;
      } else {
        m_preservedStrides[outputIndex] = strides[i];
        ++outputIndex;
      }
    }

    // Special case for full reductions
    if (NumInputDims == NumReducedDims) {
      m_dimensions[0] = 1;
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dimensions; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(Scalar* /*data*/) {
    m_impl.evalSubExprsIfNeeded(NULL);
    return true;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_impl.cleanup();
  }

  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const
  {
    Op reducer(m_reducer);
    reduce(firstInput(index), 0, reducer);
    return reducer.finalize();
  }

  // TODO(bsteiner): provide a more efficient implementation.
  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const
  {
    const int packetSize = internal::unpacket_traits<PacketReturnType>::size;
    EIGEN_STATIC_ASSERT(packetSize > 1, YOU_MADE_A_PROGRAMMING_MISTAKE)
    eigen_assert(index + packetSize - 1 < dimensions().TotalSize());

    EIGEN_ALIGN_DEFAULT CoeffReturnType values[packetSize];
    for (int i = 0; i < packetSize; ++i) {
      values[i] = coeff(index+i);
    }
    PacketReturnType rslt = internal::pload<PacketReturnType>(values);
    return rslt;
  }

  Scalar* data() const { return NULL; }

  private:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index firstInput(Index index) const {
    Index startInput = 0;
    for (int i = NumDims - 1; i > 0; --i) {
      const Index idx = index / m_outputStrides[i];
      startInput += idx * m_preservedStrides[i];
      index -= idx * m_outputStrides[i];
    }
    startInput += index * m_preservedStrides[0];
    return startInput;
  }

  EIGEN_DEVICE_FUNC void reduce(Index firstIndex, int DimIndex, Op& reducer) const {
    for (int j = 0; j < m_reducedDims[DimIndex]; ++j) {
      const Index input = firstIndex + j * m_reducedStrides[DimIndex];
      if (DimIndex < NumReducedDims-1) {
        reduce(input, DimIndex+1, reducer);
      } else {
        reducer.reduce(m_impl.coeff(input));
      }
    }
  }

  Dimensions m_dimensions;
  array<Index, NumDims> m_outputStrides;
  array<Index, NumDims> m_preservedStrides;
  array<Index, NumReducedDims> m_reducedStrides;
  array<Index, NumReducedDims> m_reducedDims;
  TensorEvaluator<ArgType, Device> m_impl;
  Op m_reducer;
};

} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_H
