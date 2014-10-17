// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_CHIPPING_H
#define EIGEN_CXX11_TENSOR_TENSOR_CHIPPING_H

namespace Eigen {

/** \class TensorKChippingReshaping
  * \ingroup CXX11_Tensor_Module
  *
  * \brief A chip is a thin slice, corresponding to a column or a row in a 2-d tensor.
  *
  *
  */

namespace internal {
template<std::size_t DimId, typename XprType>
struct traits<TensorChippingOp<DimId, XprType> > : public traits<XprType>
{
  typedef typename XprType::Scalar Scalar;
  typedef typename internal::packet_traits<Scalar>::type Packet;
  typedef typename traits<XprType>::StorageKind StorageKind;
  typedef typename traits<XprType>::Index Index;
  typedef typename XprType::Nested Nested;
  typedef typename remove_reference<Nested>::type _Nested;
};

template<std::size_t DimId, typename XprType>
struct eval<TensorChippingOp<DimId, XprType>, Eigen::Dense>
{
  typedef const TensorChippingOp<DimId, XprType>& type;
};

template<std::size_t DimId, typename XprType>
struct nested<TensorChippingOp<DimId, XprType>, 1, typename eval<TensorChippingOp<DimId, XprType> >::type>
{
  typedef TensorChippingOp<DimId, XprType> type;
};

}  // end namespace internal



template<std::size_t DimId, typename XprType>
class TensorChippingOp : public TensorBase<TensorChippingOp<DimId, XprType> >
{
  public:
  typedef typename Eigen::internal::traits<TensorChippingOp>::Scalar Scalar;
  typedef typename Eigen::internal::traits<TensorChippingOp>::Packet Packet;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;
  typedef typename Eigen::internal::nested<TensorChippingOp>::type Nested;
  typedef typename Eigen::internal::traits<TensorChippingOp>::StorageKind StorageKind;
  typedef typename Eigen::internal::traits<TensorChippingOp>::Index Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorChippingOp(const XprType& expr, const Index offset)
      : m_xpr(expr), m_offset(offset) {}

    EIGEN_DEVICE_FUNC
    const Index offset() const { return m_offset; }

    EIGEN_DEVICE_FUNC
    const typename internal::remove_all<typename XprType::Nested>::type&
    expression() const { return m_xpr; }

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE TensorChippingOp& operator = (const OtherDerived& other)
    {
      typedef TensorAssignOp<TensorChippingOp, const OtherDerived> Assign;
      Assign assign(*this, other);
      internal::TensorExecutor<const Assign, DefaultDevice, false>::run(assign, DefaultDevice());
      return *this;
    }

  protected:
    typename XprType::Nested m_xpr;
    const Index m_offset;
};


// Eval as rvalue
template<std::size_t DimId, typename ArgType, typename Device>
struct TensorEvaluator<const TensorChippingOp<DimId, ArgType>, Device>
{
  typedef TensorChippingOp<DimId, ArgType> XprType;
  static const int NumInputDims = internal::array_size<typename TensorEvaluator<ArgType, Device>::Dimensions>::value;
  static const int NumDims = NumInputDims-1;
  typedef typename XprType::Index Index;
  typedef DSizes<Index, NumDims> Dimensions;

  enum {
    // Alignment can't be guaranteed at compile time since it depends on the
    // slice offsets.
    IsAligned = false,
    PacketAccess = false,  // not yet implemented
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
      : m_impl(op.expression(), device), m_device(device)
  {
    // We could also support the case where NumInputDims==1 if needed.
    EIGEN_STATIC_ASSERT(NumInputDims >= 2, YOU_MADE_A_PROGRAMMING_MISTAKE);
    EIGEN_STATIC_ASSERT(NumInputDims > DimId, YOU_MADE_A_PROGRAMMING_MISTAKE);

    const typename TensorEvaluator<ArgType, Device>::Dimensions& input_dims = m_impl.dimensions();
    int j = 0;
    for (int i = 0; i < NumInputDims; ++i) {
      if (i != DimId) {
        m_dimensions[j] = input_dims[i];
        ++j;
      }
    }

     m_stride = 1;
     m_inputStride = 1;
     for (int i = 0; i < DimId; ++i) {
       m_stride *= input_dims[i];
       m_inputStride *= input_dims[i];
     }
     m_inputStride *= input_dims[DimId];
     m_inputOffset = m_stride * op.offset();
  }

  typedef typename XprType::Scalar Scalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dimensions; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(Scalar* /*data*/) {
    m_impl.evalSubExprsIfNeeded(NULL);
    return true;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_impl.cleanup();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const
  {
    return m_impl.coeff(srcCoeff(index));
  }

  /* to be done
  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const
  {

  }*/

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar* data() const {
    Scalar* result = m_impl.data();
    if (DimId == NumDims && result) {
      return result + m_inputOffset;
    } else {
      return NULL;
    }
  }

 protected:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index srcCoeff(Index index) const
  {
    Index inputIndex;
    if (DimId == 0) {
      // m_stride is equal to 1, so let's avoid the integer division.
      eigen_assert(m_stride == 1);
      inputIndex = index * m_inputStride + m_inputOffset;
    } else if (DimId == NumInputDims-1) {
      // m_stride is aways greater than index, so let's avoid the integer division.
      eigen_assert(m_stride > index);
      inputIndex = index + m_inputOffset;
    } else {
      const Index idx = index / m_stride;
      inputIndex = idx * m_inputStride + m_inputOffset;
      index -= idx * m_stride;
      inputIndex += index;
    }
    return inputIndex;
  }

  Dimensions m_dimensions;
  Index m_stride;
  Index m_inputOffset;
  Index m_inputStride;
  TensorEvaluator<ArgType, Device> m_impl;
  const Device& m_device;
};


// Eval as lvalue
template<std::size_t DimId, typename ArgType, typename Device>
struct TensorEvaluator<TensorChippingOp<DimId, ArgType>, Device>
  : public TensorEvaluator<const TensorChippingOp<DimId, ArgType>, Device>
{
  typedef TensorEvaluator<const TensorChippingOp<DimId, ArgType>, Device> Base;
  typedef TensorChippingOp<DimId, ArgType> XprType;
  static const int NumInputDims = internal::array_size<typename TensorEvaluator<ArgType, Device>::Dimensions>::value;
  static const int NumDims = NumInputDims-1;
  typedef typename XprType::Index Index;
  typedef DSizes<Index, NumDims> Dimensions;

  enum {
    IsAligned = false,
    PacketAccess = false,
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
    : Base(op, device)
    { }

  typedef typename XprType::Scalar Scalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType& coeffRef(Index index)
  {
    return this->m_impl.coeffRef(this->srcCoeff(index));
  }

  /* to be done
  template <int StoreMode> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  void writePacket(Index index, const PacketReturnType& x)
  {
  } */
};


} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_CHIPPING_H
