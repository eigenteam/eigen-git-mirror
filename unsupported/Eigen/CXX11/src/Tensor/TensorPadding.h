// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_PADDING_H
#define EIGEN_CXX11_TENSOR_TENSOR_PADDING_H

namespace Eigen {

/** \class TensorPadding
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor padding class.
  * At the moment only 0-padding is supported.
  *
  */
namespace internal {
template<typename PaddingDimensions, typename XprType>
struct traits<TensorPaddingOp<PaddingDimensions, XprType> > : public traits<XprType>
{
  typedef typename XprType::Scalar Scalar;
  typedef typename internal::packet_traits<Scalar>::type Packet;
  typedef typename traits<XprType>::StorageKind StorageKind;
  typedef typename traits<XprType>::Index Index;
  typedef typename XprType::Nested Nested;
  typedef typename remove_reference<Nested>::type _Nested;
};

template<typename PaddingDimensions, typename XprType>
struct eval<TensorPaddingOp<PaddingDimensions, XprType>, Eigen::Dense>
{
  typedef const TensorPaddingOp<PaddingDimensions, XprType>& type;
};

template<typename PaddingDimensions, typename XprType>
struct nested<TensorPaddingOp<PaddingDimensions, XprType>, 1, typename eval<TensorPaddingOp<PaddingDimensions, XprType> >::type>
{
  typedef TensorPaddingOp<PaddingDimensions, XprType> type;
};

}  // end namespace internal



template<typename PaddingDimensions, typename XprType>
class TensorPaddingOp : public TensorBase<TensorPaddingOp<PaddingDimensions, XprType>, ReadOnlyAccessors>
{
  public:
  typedef typename Eigen::internal::traits<TensorPaddingOp>::Scalar Scalar;
  typedef typename Eigen::internal::traits<TensorPaddingOp>::Packet Packet;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;
  typedef typename Eigen::internal::nested<TensorPaddingOp>::type Nested;
  typedef typename Eigen::internal::traits<TensorPaddingOp>::StorageKind StorageKind;
  typedef typename Eigen::internal::traits<TensorPaddingOp>::Index Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorPaddingOp(const XprType& expr, const PaddingDimensions& padding_dims)
      : m_xpr(expr), m_padding_dims(padding_dims) {}

    EIGEN_DEVICE_FUNC
    const PaddingDimensions& padding() const { return m_padding_dims; }

    EIGEN_DEVICE_FUNC
    const typename internal::remove_all<typename XprType::Nested>::type&
    expression() const { return m_xpr; }

  protected:
    typename XprType::Nested m_xpr;
    const PaddingDimensions m_padding_dims;
};


// Eval as rvalue
template<typename PaddingDimensions, typename ArgType, typename Device>
struct TensorEvaluator<const TensorPaddingOp<PaddingDimensions, ArgType>, Device>
{
  typedef TensorPaddingOp<PaddingDimensions, ArgType> XprType;
  typedef typename XprType::Index Index;
  static const int NumDims = internal::array_size<PaddingDimensions>::value;
  typedef DSizes<Index, NumDims> Dimensions;

  enum {
    IsAligned = false,
    PacketAccess = TensorEvaluator<ArgType, Device>::PacketAccess,
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
      : m_impl(op.expression(), device), m_padding(op.padding())
  {
    // Compute dimensions
    m_dimensions = m_impl.dimensions();
    for (int i = 0; i < NumDims; ++i) {
      m_dimensions[i] += m_padding[i].first + m_padding[i].second;
    }

    const typename TensorEvaluator<ArgType, Device>::Dimensions& input_dims = m_impl.dimensions();
    m_inputStrides[0] = 1;
    m_outputStrides[0] = 1;
    for (int i = 1; i < NumDims; ++i) {
      m_inputStrides[i] = m_inputStrides[i-1] * input_dims[i-1];
      m_outputStrides[i] = m_outputStrides[i-1] * m_dimensions[i-1];
    }
    m_outputStrides[NumDims] = m_outputStrides[NumDims-1] * m_dimensions[NumDims-1];
  }

  typedef typename XprType::Scalar Scalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dimensions; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(Scalar*) {
    m_impl.evalSubExprsIfNeeded(NULL);
    return true;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_impl.cleanup();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const
  {
    Index inputIndex = 0;
    for (int i = NumDims - 1; i > 0; --i) {
      const Index idx = index / m_outputStrides[i];
      if (idx < m_padding[i].first || idx >= m_dimensions[i] - m_padding[i].second) {
        return Scalar(0);
      }
      inputIndex += (idx - m_padding[i].first) * m_inputStrides[i];
      index -= idx * m_outputStrides[i];
    }
    if (index < m_padding[0].first || index >= m_dimensions[0] - m_padding[0].second) {
      return Scalar(0);
    }
    inputIndex += (index - m_padding[0].first);
    return m_impl.coeff(inputIndex);
  }

  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const
  {
    static const int packetSize = internal::unpacket_traits<PacketReturnType>::size;
    EIGEN_STATIC_ASSERT(packetSize > 1, YOU_MADE_A_PROGRAMMING_MISTAKE)
    eigen_assert(index+packetSize-1 < dimensions().TotalSize());

    const Index initialIndex = index;
    Index inputIndex = 0;
    for (int i = NumDims - 1; i > 0; --i) {
      const int first = index;
      const int last = index + packetSize - 1;
      const int lastPaddedLeft = m_padding[i].first * m_outputStrides[i];
      const int firstPaddedRight = (m_dimensions[i] - m_padding[i].second) * m_outputStrides[i];
      const int lastPaddedRight = m_outputStrides[i+1];

      if (last < lastPaddedLeft) {
        // all the coefficient are in the padding zone.
        return internal::pset1<PacketReturnType>(Scalar(0));
      }
      else if (first >= firstPaddedRight && last < lastPaddedRight) {
        // all the coefficient are in the padding zone.
        return internal::pset1<PacketReturnType>(Scalar(0));
      }
      else if (first >= lastPaddedLeft && last < firstPaddedRight) {
        // all the coefficient are between the 2 padding zones.
        const Index idx = index / m_outputStrides[i];
        inputIndex += (idx - m_padding[i].first) * m_inputStrides[i];
        index -= idx * m_outputStrides[i];
      }
      else {
        // Every other case
        return packetWithPossibleZero(initialIndex);
      }
    }

    const Index last = index + packetSize - 1;
    const Index first = index;
    const int lastPaddedLeft = m_padding[0].first;
    const int firstPaddedRight = (m_dimensions[0] - m_padding[0].second);
    const int lastPaddedRight = m_outputStrides[1];

    if (last < lastPaddedLeft) {
      // all the coefficient are in the padding zone.
      return internal::pset1<PacketReturnType>(Scalar(0));
    }
    else if (first >= firstPaddedRight && last < lastPaddedRight) {
      // all the coefficient are in the padding zone.
      return internal::pset1<PacketReturnType>(Scalar(0));
    }
    else if (first >= lastPaddedLeft && last < firstPaddedRight) {
      // all the coefficient are between the 2 padding zones.
      inputIndex += (index - m_padding[0].first);
      return m_impl.template packet<Unaligned>(inputIndex);
    }
    // Every other case
    return packetWithPossibleZero(initialIndex);
  }

  Scalar* data() const { return NULL; }

 protected:

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packetWithPossibleZero(Index index) const
  {
    static const int packetSize = internal::unpacket_traits<PacketReturnType>::size;
    EIGEN_ALIGN_DEFAULT typename internal::remove_const<CoeffReturnType>::type values[packetSize];
    for (int i = 0; i < packetSize; ++i) {
      values[i] = coeff(index+i);
    }
    PacketReturnType rslt = internal::pload<PacketReturnType>(values);
    return rslt;
  }

  PaddingDimensions m_padding;
  Dimensions m_dimensions;
  array<Index, NumDims+1> m_outputStrides;
  array<Index, NumDims> m_inputStrides;
  TensorEvaluator<ArgType, Device> m_impl;
};




} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_PADDING_H
