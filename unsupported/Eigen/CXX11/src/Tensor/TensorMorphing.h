// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_MORPHING_H
#define EIGEN_CXX11_TENSOR_TENSOR_MORPHING_H

namespace Eigen {

/** \class TensorReshaping
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor reshaping class.
  *
  *
  */
namespace internal {
template<typename NewDimensions, typename XprType>
struct traits<TensorReshapingOp<NewDimensions, XprType> > : public traits<XprType>
{
  typedef typename XprType::Scalar Scalar;
  typedef typename internal::packet_traits<Scalar>::type Packet;
  typedef typename traits<XprType>::StorageKind StorageKind;
  typedef typename traits<XprType>::Index Index;
  typedef typename XprType::Nested Nested;
  typedef typename remove_reference<Nested>::type _Nested;
};

template<typename NewDimensions, typename XprType>
struct eval<TensorReshapingOp<NewDimensions, XprType>, Eigen::Dense>
{
  typedef const TensorReshapingOp<NewDimensions, XprType>& type;
};

template<typename NewDimensions, typename XprType>
struct nested<TensorReshapingOp<NewDimensions, XprType>, 1, typename eval<TensorReshapingOp<NewDimensions, XprType> >::type>
{
  typedef TensorReshapingOp<NewDimensions, XprType> type;
};

}  // end namespace internal



template<typename NewDimensions, typename XprType>
class TensorReshapingOp : public TensorBase<TensorReshapingOp<NewDimensions, XprType>, WriteAccessors>
{
  public:
  typedef typename Eigen::internal::traits<TensorReshapingOp>::Scalar Scalar;
  typedef typename Eigen::internal::traits<TensorReshapingOp>::Packet Packet;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;
  typedef typename Eigen::internal::nested<TensorReshapingOp>::type Nested;
  typedef typename Eigen::internal::traits<TensorReshapingOp>::StorageKind StorageKind;
  typedef typename Eigen::internal::traits<TensorReshapingOp>::Index Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorReshapingOp(const XprType& expr, const NewDimensions& dims)
      : m_xpr(expr), m_dims(dims) {}

    EIGEN_DEVICE_FUNC
    const NewDimensions& dimensions() const { return m_dims; }

    EIGEN_DEVICE_FUNC
    const typename internal::remove_all<typename XprType::Nested>::type&
    expression() const { return m_xpr; }

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE TensorReshapingOp& operator = (const OtherDerived& other)
    {
      typedef TensorAssignOp<TensorReshapingOp, const OtherDerived> Assign;
      Assign assign(*this, other);
      internal::TensorExecutor<const Assign, DefaultDevice, false>::run(assign, DefaultDevice());
      return *this;
    }

  protected:
    typename XprType::Nested m_xpr;
    const NewDimensions m_dims;
};


// Eval as rvalue
template<typename NewDimensions, typename ArgType, typename Device>
struct TensorEvaluator<const TensorReshapingOp<NewDimensions, ArgType>, Device>
{
  typedef TensorReshapingOp<NewDimensions, ArgType> XprType;
  typedef NewDimensions Dimensions;

  enum {
    IsAligned = TensorEvaluator<ArgType, Device>::IsAligned,
    PacketAccess = TensorEvaluator<ArgType, Device>::PacketAccess,
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
      : m_impl(op.expression(), device), m_dimensions(op.dimensions())
  { }

  typedef typename XprType::Index Index;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dimensions; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void evalSubExprsIfNeeded() {
    m_impl.evalSubExprsIfNeeded();
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_impl.cleanup();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const
  {
    return m_impl.coeff(index);
  }

  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const
  {
    return m_impl.template packet<LoadMode>(index);
  }

 protected:
  NewDimensions m_dimensions;
  TensorEvaluator<ArgType, Device> m_impl;
};


// Eval as lvalue
// TODO(bsteiner): share the code with the evaluator for rvalue reshapes.
template<typename NewDimensions, typename ArgType, typename Device>
struct TensorEvaluator<TensorReshapingOp<NewDimensions, ArgType>, Device>
{
  typedef TensorReshapingOp<NewDimensions, ArgType> XprType;
  typedef NewDimensions Dimensions;

  enum {
    IsAligned = TensorEvaluator<ArgType, Device>::IsAligned,
    PacketAccess = TensorEvaluator<ArgType, Device>::PacketAccess,
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
      : m_impl(op.expression(), device), m_dimensions(op.dimensions())
  { }

  typedef typename XprType::Index Index;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dimensions; }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void evalSubExprsIfNeeded() {
    m_impl.evalSubExprsIfNeeded();
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_impl.cleanup();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const
  {
    return m_impl.coeff(index);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType& coeffRef(Index index)
  {
    return m_impl.coeffRef(index);
  }
  template <int StoreMode> EIGEN_STRONG_INLINE
  void writePacket(Index index, const PacketReturnType& x)
  {
    m_impl.template writePacket<StoreMode>(index, x);
  }
  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const
  {
    return m_impl.template packet<LoadMode>(index);
  }

 private:
  NewDimensions m_dimensions;
  TensorEvaluator<ArgType, Device> m_impl;
};


/** \class TensorSlicing
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor slicing class.
  *
  *
  */
namespace internal {
template<typename StartIndices, typename Sizes, typename XprType>
struct traits<TensorSlicingOp<StartIndices, Sizes, XprType> > : public traits<XprType>
{
  typedef typename XprType::Scalar Scalar;
  typedef typename internal::packet_traits<Scalar>::type Packet;
  typedef typename traits<XprType>::StorageKind StorageKind;
  typedef typename traits<XprType>::Index Index;
  typedef typename XprType::Nested Nested;
  typedef typename remove_reference<Nested>::type _Nested;
};

template<typename StartIndices, typename Sizes, typename XprType>
struct eval<TensorSlicingOp<StartIndices, Sizes, XprType>, Eigen::Dense>
{
  typedef const TensorSlicingOp<StartIndices, Sizes, XprType>& type;
};

template<typename StartIndices, typename Sizes, typename XprType>
struct nested<TensorSlicingOp<StartIndices, Sizes, XprType>, 1, typename eval<TensorSlicingOp<StartIndices, Sizes, XprType> >::type>
{
  typedef TensorSlicingOp<StartIndices, Sizes, XprType> type;
};

}  // end namespace internal



template<typename StartIndices, typename Sizes, typename XprType>
class TensorSlicingOp : public TensorBase<TensorSlicingOp<StartIndices, Sizes, XprType> >
{
  public:
  typedef typename Eigen::internal::traits<TensorSlicingOp>::Scalar Scalar;
  typedef typename Eigen::internal::traits<TensorSlicingOp>::Packet Packet;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;
  typedef typename Eigen::internal::nested<TensorSlicingOp>::type Nested;
  typedef typename Eigen::internal::traits<TensorSlicingOp>::StorageKind StorageKind;
  typedef typename Eigen::internal::traits<TensorSlicingOp>::Index Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorSlicingOp(const XprType& expr, const StartIndices& indices, const Sizes& sizes)
      : m_xpr(expr), m_indices(indices), m_sizes(sizes) {}

    EIGEN_DEVICE_FUNC
    const StartIndices& startIndices() const { return m_indices; }
    EIGEN_DEVICE_FUNC
    const Sizes& sizes() const { return m_sizes; }

    EIGEN_DEVICE_FUNC
    const typename internal::remove_all<typename XprType::Nested>::type&
    expression() const { return m_xpr; }

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE TensorSlicingOp& operator = (const OtherDerived& other)
    {
      typedef TensorAssignOp<TensorSlicingOp, const OtherDerived> Assign;
      Assign assign(*this, other);
      internal::TensorExecutor<const Assign, DefaultDevice, false>::run(assign, DefaultDevice());
      return *this;
    }

  protected:
    typename XprType::Nested m_xpr;
    const StartIndices m_indices;
    const Sizes m_sizes;
};


// Eval as rvalue
template<typename StartIndices, typename Sizes, typename ArgType, typename Device>
struct TensorEvaluator<const TensorSlicingOp<StartIndices, Sizes, ArgType>, Device>
{
  typedef TensorSlicingOp<StartIndices, Sizes, ArgType> XprType;
  static const int NumDims = internal::array_size<Sizes>::value;

  enum {
    IsAligned = TensorEvaluator<ArgType, Device>::IsAligned,
    PacketAccess = /*TensorEvaluator<ArgType, Device>::PacketAccess*/false,
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
      : m_impl(op.expression(), device), m_dimensions(op.sizes()), m_offsets(op.startIndices())
  {
    for (int i = 0; i < internal::array_size<Dimensions>::value; ++i) {
      eigen_assert(m_impl.dimensions()[i] >= op.sizes()[i] + op.startIndices()[i]);
    }

    const typename TensorEvaluator<ArgType, Device>::Dimensions& input_dims = m_impl.dimensions();
    for (int i = 0; i < NumDims; ++i) {
      if (i > 0) {
        m_inputStrides[i] = m_inputStrides[i-1] * input_dims[i-1];
      } else {
        m_inputStrides[0] = 1;
      }
    }

    const Sizes& output_dims = op.sizes();
    for (int i = 0; i < NumDims; ++i) {
      if (i > 0) {
        m_outputStrides[i] = m_outputStrides[i-1] * output_dims[i-1];
      } else {
        m_outputStrides[0] = 1;
      }
    }
  }

  typedef typename XprType::Index Index;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;
  typedef Sizes Dimensions;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dimensions; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void evalSubExprsIfNeeded() {
    m_impl.evalSubExprsIfNeeded();
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_impl.cleanup();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const
  {
    Index inputIndex = 0;
    for (int i = NumDims - 1; i > 0; --i) {
      const Index idx = index / m_outputStrides[i];
      inputIndex += (idx + m_offsets[i]) * m_inputStrides[i];
      index -= idx * m_outputStrides[i];
    }
    inputIndex += (index + m_offsets[0]);
    return m_impl.coeff(inputIndex);
  }

  /*  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const
  {
    return m_impl.template packet<LoadMode>(index);
    }*/

 private:
  Dimensions m_dimensions;
  array<Index, NumDims> m_outputStrides;
  array<Index, NumDims> m_inputStrides;
  const StartIndices m_offsets;
  TensorEvaluator<ArgType, Device> m_impl;
};


// Eval as lvalue
// TODO(bsteiner): share the code with the evaluator for rvalue slices.
template<typename StartIndices, typename Sizes, typename ArgType, typename Device>
struct TensorEvaluator<TensorSlicingOp<StartIndices, Sizes, ArgType>, Device>
{
  typedef TensorSlicingOp<StartIndices, Sizes, ArgType> XprType;
  static const int NumDims = internal::array_size<Sizes>::value;

  enum {
    IsAligned = TensorEvaluator<ArgType, Device>::IsAligned,
    PacketAccess = /*TensorEvaluator<ArgType, Device>::PacketAccess*/false,
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
      : m_impl(op.expression(), device), m_dimensions(op.sizes()), m_offsets(op.startIndices())
  {
    for (int i = 0; i < internal::array_size<Dimensions>::value; ++i) {
      eigen_assert(m_impl.dimensions()[i] >= op.sizes()[i] + op.startIndices()[i]);
    }

    const typename TensorEvaluator<ArgType, Device>::Dimensions& input_dims = m_impl.dimensions();
    for (int i = 0; i < NumDims; ++i) {
      if (i > 0) {
        m_inputStrides[i] = m_inputStrides[i-1] * input_dims[i-1];
      } else {
        m_inputStrides[0] = 1;
      }
    }

    const Sizes& output_dims = op.sizes();
    for (int i = 0; i < NumDims; ++i) {
      if (i > 0) {
        m_outputStrides[i] = m_outputStrides[i-1] * output_dims[i-1];
      } else {
        m_outputStrides[0] = 1;
      }
    }
  }

  typedef typename XprType::Index Index;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;
  typedef Sizes Dimensions;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dimensions; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void evalSubExprsIfNeeded() {
    m_impl.evalSubExprsIfNeeded();
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_impl.cleanup();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const
  {
    Index inputIndex = 0;
    for (int i = NumDims - 1; i > 0; --i) {
      const Index idx = index / m_outputStrides[i];
      inputIndex += (idx + m_offsets[i]) * m_inputStrides[i];
      index -= idx * m_outputStrides[i];
    }
    inputIndex += (index + m_offsets[0]);
    return m_impl.coeff(inputIndex);
  }

  /*  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const
  {
    return m_impl.template packet<LoadMode>(index);
    }*/

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType& coeffRef(Index index)
  {
    Index inputIndex = 0;
    for (int i = NumDims - 1; i > 0; --i) {
      const Index idx = index / m_outputStrides[i];
      inputIndex += (idx + m_offsets[i]) * m_inputStrides[i];
      index -= idx * m_outputStrides[i];
    }
    inputIndex += (index + m_offsets[0]);
    return m_impl.coeffRef(inputIndex);
  }

 private:
  Dimensions m_dimensions;
  array<Index, NumDims> m_outputStrides;
  array<Index, NumDims> m_inputStrides;
  const StartIndices m_offsets;
  TensorEvaluator<ArgType, Device> m_impl;
};


} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_MORPHING_H
