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


template <typename ReducedDims, int NumTensorDims, int Layout>
struct are_inner_most_dims {
  static const bool value = false;
};
template <typename ReducedDims, int NumTensorDims, int Layout>
struct preserve_inner_most_dims {
  static const bool value = false;
};

#ifdef EIGEN_HAS_CONSTEXPR
template <typename ReducedDims, int NumTensorDims>
struct are_inner_most_dims<ReducedDims, NumTensorDims, ColMajor>{
  static const bool value = indices_statically_known_to_increase<ReducedDims>()() &&
                            index_statically_eq<ReducedDims>()(0, 0) &&
                            index_statically_eq<ReducedDims>()(array_size<ReducedDims>::value-1, array_size<ReducedDims>::value-1);
};
template <typename ReducedDims, int NumTensorDims>
struct are_inner_most_dims<ReducedDims, NumTensorDims, RowMajor>{
  static const bool value = indices_statically_known_to_increase<ReducedDims>()() &&
                            index_statically_eq<ReducedDims>()(0, NumTensorDims - array_size<ReducedDims>::value) &&
                            index_statically_eq<ReducedDims>()(array_size<ReducedDims>::value - 1, NumTensorDims - 1);
};
template <typename ReducedDims, int NumTensorDims>
struct preserve_inner_most_dims<ReducedDims, NumTensorDims, ColMajor>{
  static const bool value = indices_statically_known_to_increase<ReducedDims>()() &&
                            index_statically_gt<ReducedDims>()(0, 0);
};
template <typename ReducedDims, int NumTensorDims>
struct preserve_inner_most_dims<ReducedDims, NumTensorDims, RowMajor>{
  static const bool value = indices_statically_known_to_increase<ReducedDims>()() &&
                            index_statically_lt<ReducedDims>()(array_size<ReducedDims>::value - 1, NumTensorDims - 1);
};
#endif


template <int DimIndex, typename Self, typename Op>
struct GenericDimReducer {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const Self& self, typename Self::Index firstIndex, Op& reducer, typename Self::CoeffReturnType* accum) {
    EIGEN_STATIC_ASSERT(DimIndex > 0, YOU_MADE_A_PROGRAMMING_MISTAKE);
    for (int j = 0; j < self.m_reducedDims[DimIndex]; ++j) {
      const typename Self::Index input = firstIndex + j * self.m_reducedStrides[DimIndex];
      GenericDimReducer<DimIndex-1, Self, Op>::reduce(self, input, reducer, accum);
    }
  }
};
template <typename Self, typename Op>
struct GenericDimReducer<0, Self, Op> {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const Self& self, typename Self::Index firstIndex, Op& reducer, typename Self::CoeffReturnType* accum) {
    for (int j = 0; j < self.m_reducedDims[0]; ++j) {
      const typename Self::Index input = firstIndex + j * self.m_reducedStrides[0];
      reducer.reduce(self.m_impl.coeff(input), accum);
    }
  }
};

template <typename Self, typename Op, bool Vectorizable = (Self::InputPacketAccess & Op::PacketAccess)>
struct InnerMostDimReducer {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE typename Self::CoeffReturnType reduce(const Self& self, typename Self::Index firstIndex, typename Self::Index numValuesToReduce, Op& reducer) {
    typename Self::CoeffReturnType accum = reducer.initialize();
    for (typename Self::Index j = 0; j < numValuesToReduce; ++j) {
      reducer.reduce(self.m_impl.coeff(firstIndex + j), &accum);
    }
    return reducer.finalize(accum);
  }
};

template <typename Self, typename Op>
struct InnerMostDimReducer<Self, Op, true> {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE typename Self::CoeffReturnType reduce(const Self& self, typename Self::Index firstIndex, typename Self::Index numValuesToReduce, Op& reducer) {
    const int packetSize = internal::unpacket_traits<typename Self::PacketReturnType>::size;
    const typename Self::Index VectorizedSize = (numValuesToReduce / packetSize) * packetSize;
    typename Self::PacketReturnType p = reducer.template initializePacket<typename Self::PacketReturnType>();
    for (typename Self::Index j = 0; j < VectorizedSize; j += packetSize) {
      reducer.reducePacket(self.m_impl.template packet<Unaligned>(firstIndex + j), &p);
    }
    typename Self::CoeffReturnType accum = reducer.initialize();
    for (typename Self::Index j = VectorizedSize; j < numValuesToReduce; ++j) {
      reducer.reduce(self.m_impl.coeff(firstIndex + j), &accum);
    }
    return reducer.finalizeBoth(accum, p);
  }
};

template <int DimIndex, typename Self, typename Op, bool vectorizable = (Self::InputPacketAccess & Op::PacketAccess)>
struct InnerMostDimPreserver {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const Self&, typename Self::Index, Op&, typename Self::PacketReturnType*) {
    eigen_assert(false && "should never be called");
  }
};

template <int DimIndex, typename Self, typename Op>
struct InnerMostDimPreserver<DimIndex, Self, Op, true> {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const Self& self, typename Self::Index firstIndex, Op& reducer, typename Self::PacketReturnType* accum) {
    EIGEN_STATIC_ASSERT(DimIndex > 0, YOU_MADE_A_PROGRAMMING_MISTAKE);
    for (int j = 0; j < self.m_reducedDims[DimIndex]; ++j) {
      const typename Self::Index input = firstIndex + j * self.m_reducedStrides[DimIndex];
      InnerMostDimPreserver<DimIndex-1, Self, Op>::reduce(self, input, reducer, accum);
    }
  }
};

template <typename Self, typename Op>
struct InnerMostDimPreserver<0, Self, Op, true> {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const Self& self, typename Self::Index firstIndex, Op& reducer, typename Self::PacketReturnType* accum) {
    for (int j = 0; j < self.m_reducedDims[0]; ++j) {
      const typename Self::Index input = firstIndex + j * self.m_reducedStrides[0];
      reducer.reducePacket(self.m_impl.template packet<Unaligned>(input), accum);
    }
  }
};

}  // end namespace internal


template <typename Op, typename Dims, typename XprType>
class TensorReductionOp : public TensorBase<TensorReductionOp<Op, Dims, XprType>, ReadOnlyAccessors> {
  public:
    typedef typename Eigen::internal::traits<TensorReductionOp>::Scalar Scalar;
    typedef typename Eigen::internal::traits<TensorReductionOp>::Packet Packet;
    typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
    typedef typename internal::remove_const<typename XprType::CoeffReturnType>::type CoeffReturnType;
    typedef typename internal::remove_const<typename XprType::PacketReturnType>::type PacketReturnType;
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
  static const int NumOutputDims = (NumInputDims==NumReducedDims) ? 1 : NumInputDims - NumReducedDims;
  typedef DSizes<Index, NumOutputDims> Dimensions;
  typedef typename XprType::Scalar Scalar;
  typedef TensorEvaluator<const TensorReductionOp<Op, Dims, ArgType>, Device> Self;
  static const bool InputPacketAccess = TensorEvaluator<ArgType, Device>::PacketAccess;

  enum {
    IsAligned = false,
    PacketAccess = Self::InputPacketAccess && Op::PacketAccess,
    Layout = TensorEvaluator<ArgType, Device>::Layout,
    CoordAccess = false,  // to be implemented
  };

  static const bool ReducingInnerMostDims = internal::are_inner_most_dims<Dims, NumInputDims, Layout>::value;
  static const bool PreservingInnerMostDims = internal::preserve_inner_most_dims<Dims, NumInputDims, Layout>::value;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
      : m_impl(op.expression(), device), m_reducer(op.reducer())
  {
    EIGEN_STATIC_ASSERT(NumInputDims >= NumReducedDims, YOU_MADE_A_PROGRAMMING_MISTAKE);
    EIGEN_STATIC_ASSERT((!ReducingInnerMostDims | !PreservingInnerMostDims | (NumReducedDims == NumInputDims)),
                        YOU_MADE_A_PROGRAMMING_MISTAKE);

    // Bitmap indicating if an input dimension is reduced or not.
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

    // Precompute output strides.
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      m_outputStrides[0] = 1;
      for (int i = 1; i < NumOutputDims; ++i) {
        m_outputStrides[i] = m_outputStrides[i - 1] * m_dimensions[i - 1];
      }
    } else {
      m_outputStrides[NumOutputDims - 1] = 1;
      for (int i = NumOutputDims - 2; i >= 0; --i) {
        m_outputStrides[i] = m_outputStrides[i + 1] * m_dimensions[i + 1];
      }
    }

    // Precompute input strides.
    array<Index, NumInputDims> input_strides;
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      input_strides[0] = 1;
      for (int i = 1; i < NumInputDims; ++i) {
        input_strides[i] = input_strides[i-1] * input_dims[i-1];
      }
    } else {
      input_strides[NumInputDims - 1] = 1;
      for (int i = NumInputDims - 2; i >= 0; --i) {
        input_strides[i] = input_strides[i + 1] * input_dims[i + 1];
      }
    }

    outputIndex = 0;
    reduceIndex = 0;
    for (int i = 0; i < NumInputDims; ++i) {
      if (reduced[i]) {
        m_reducedStrides[reduceIndex] = input_strides[i];
        ++reduceIndex;
      } else {
        m_preservedStrides[outputIndex] = input_strides[i];
        ++outputIndex;
      }
    }

    // Special case for full reductions
    if (NumInputDims == NumReducedDims) {
      m_dimensions[0] = 1;
      m_preservedStrides[0] = internal::array_prod(input_dims);
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

  typedef typename internal::remove_const<typename XprType::CoeffReturnType>::type CoeffReturnType;
  typedef typename internal::remove_const<typename XprType::PacketReturnType>::type PacketReturnType;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const
  {
    Op reducer(m_reducer);
    if (ReducingInnerMostDims) {
      const Index num_values_to_reduce =
	(static_cast<int>(Layout) == static_cast<int>(ColMajor)) ? m_preservedStrides[0] : m_preservedStrides[NumOutputDims - 1];
      return internal::InnerMostDimReducer<Self, Op>::reduce(*this, firstInput(index),
                                                             num_values_to_reduce, reducer);
    } else {
      typename Self::CoeffReturnType accum = reducer.initialize();
      internal::GenericDimReducer<NumReducedDims-1, Self, Op>::reduce(*this, firstInput(index), reducer, &accum);
      return reducer.finalize(accum);
    }
  }

  // TODO(bsteiner): provide a more efficient implementation.
  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const
  {
    const int packetSize = internal::unpacket_traits<PacketReturnType>::size;
    EIGEN_STATIC_ASSERT(packetSize > 1, YOU_MADE_A_PROGRAMMING_MISTAKE)
    eigen_assert(index + packetSize - 1 < dimensions().TotalSize());

    EIGEN_ALIGN_DEFAULT typename internal::remove_const<CoeffReturnType>::type values[packetSize];
    if (ReducingInnerMostDims) {
      const Index num_values_to_reduce =
	(static_cast<int>(Layout) == static_cast<int>(ColMajor)) ? m_preservedStrides[0] : m_preservedStrides[NumOutputDims - 1];
      const Index firstIndex = firstInput(index);
      for (Index i = 0; i < packetSize; ++i) {
        Op reducer(m_reducer);
        values[i] = internal::InnerMostDimReducer<Self, Op>::reduce(*this, firstIndex + i * num_values_to_reduce,
                                                                    num_values_to_reduce, reducer);
      }
    } else if (PreservingInnerMostDims) {
      const Index firstIndex = firstInput(index);
      const int innermost_dim = (static_cast<int>(Layout) == static_cast<int>(ColMajor)) ? 0 : NumOutputDims - 1;
      // TBD: extend this the the n innermost dimensions that we preserve.
      if (((firstIndex % m_dimensions[innermost_dim]) + packetSize - 1) < m_dimensions[innermost_dim]) {
        Op reducer(m_reducer);
        typename Self::PacketReturnType accum = reducer.template initializePacket<typename Self::PacketReturnType>();
        internal::InnerMostDimPreserver<NumReducedDims-1, Self, Op>::reduce(*this, firstIndex, reducer, &accum);
        return reducer.finalizePacket(accum);
      } else {
        for (int i = 0; i < packetSize; ++i) {
          values[i] = coeff(index + i);
        }
      }
    } else {
      for (int i = 0; i < packetSize; ++i) {
        values[i] = coeff(index + i);
      }
    }
    PacketReturnType rslt = internal::pload<PacketReturnType>(values);
    return rslt;
  }

  Scalar* data() const { return NULL; }

  private:
  template <int, typename, typename> friend struct internal::GenericDimReducer;
  template <typename, typename, bool> friend struct internal::InnerMostDimReducer;
  template <int, typename, typename, bool> friend struct internal::InnerMostDimPreserver;

  // Returns the Index in the input tensor of the first value that needs to be
  // used to compute the reduction at output index "index".
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index firstInput(Index index) const {
    if (ReducingInnerMostDims) {
      if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
        return index * m_preservedStrides[0];
      } else {
        return index * m_preservedStrides[NumOutputDims - 1];
      }
    }
    // TBD: optimize the case where we preserve the innermost dimensions.
    Index startInput = 0;
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      for (int i = NumOutputDims - 1; i > 0; --i) {
        // This is index_i in the output tensor.
        const Index idx = index / m_outputStrides[i];
        startInput += idx * m_preservedStrides[i];
        index -= idx * m_outputStrides[i];
      }
      startInput += index * m_preservedStrides[0];
    } else {
      for (int i = 0; i < NumOutputDims - 1; ++i) {
        // This is index_i in the output tensor.
        const Index idx = index / m_outputStrides[i];
        startInput += idx * m_preservedStrides[i];
        index -= idx * m_outputStrides[i];
      }
      startInput += index * m_preservedStrides[NumOutputDims - 1];
    }
    return startInput;
  }

  // Dimensions of the output of the operation.
  Dimensions m_dimensions;
  // Precomputed strides for the output tensor.
  array<Index, NumOutputDims> m_outputStrides;
  // Subset of strides of the input tensor for the non-reduced dimensions.
  // Indexed by output dimensions.
  array<Index, NumOutputDims> m_preservedStrides;

  // Subset of strides of the input tensor for the reduced dimensions.
  // Indexed by reduced dimensions.
  array<Index, NumReducedDims> m_reducedStrides;
  // Size of the input dimensions that are reduced.
  // Indexed by reduced dimensions.
  array<Index, NumReducedDims> m_reducedDims;

  // Evaluator for the input expression.
  TensorEvaluator<ArgType, Device> m_impl;

  // Operation to apply for computing the reduction.
  Op m_reducer;
};

} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_H
