// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_STRIDING_H
#define EIGEN_CXX11_TENSOR_TENSOR_STRIDING_H

namespace Eigen {

/** \class TensorStriding
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor striding class.
  *
  *
  */
namespace internal {
template<typename Strides, typename XprType>
struct traits<TensorStridingOp<Strides, XprType> > : public traits<XprType>
{
  typedef typename XprType::Scalar Scalar;
  typedef typename internal::packet_traits<Scalar>::type Packet;
  typedef typename traits<XprType>::StorageKind StorageKind;
  typedef typename traits<XprType>::Index Index;
  typedef typename XprType::Nested Nested;
  typedef typename remove_reference<Nested>::type _Nested;
};

template<typename Strides, typename XprType>
struct eval<TensorStridingOp<Strides, XprType>, Eigen::Dense>
{
  typedef const TensorStridingOp<Strides, XprType>& type;
};

template<typename Strides, typename XprType>
struct nested<TensorStridingOp<Strides, XprType>, 1, typename eval<TensorStridingOp<Strides, XprType> >::type>
{
  typedef TensorStridingOp<Strides, XprType> type;
};

}  // end namespace internal



template<typename Strides, typename XprType>
class TensorStridingOp : public TensorBase<TensorStridingOp<Strides, XprType>, WriteAccessors>
{
  public:
  typedef typename Eigen::internal::traits<TensorStridingOp>::Scalar Scalar;
  typedef typename Eigen::internal::traits<TensorStridingOp>::Packet Packet;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;
  typedef typename Eigen::internal::nested<TensorStridingOp>::type Nested;
  typedef typename Eigen::internal::traits<TensorStridingOp>::StorageKind StorageKind;
  typedef typename Eigen::internal::traits<TensorStridingOp>::Index Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorStridingOp(const XprType& expr, const Strides& dims)
      : m_xpr(expr), m_dims(dims) {}

    EIGEN_DEVICE_FUNC
    const Strides& strides() const { return m_dims; }

    EIGEN_DEVICE_FUNC
    const typename internal::remove_all<typename XprType::Nested>::type&
    expression() const { return m_xpr; }

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE TensorStridingOp& operator = (const OtherDerived& other)
    {
      typedef TensorAssignOp<TensorStridingOp, const OtherDerived> Assign;
      Assign assign(*this, other);
      internal::TensorExecutor<const Assign, DefaultDevice, false>::run(assign, DefaultDevice());
      return *this;
    }

  protected:
    typename XprType::Nested m_xpr;
    const Strides m_dims;
};


// Eval as rvalue
template<typename Strides, typename ArgType, typename Device>
struct TensorEvaluator<const TensorStridingOp<Strides, ArgType>, Device>
{
  typedef TensorStridingOp<Strides, ArgType> XprType;
  typedef typename XprType::Index Index;
  static const int NumDims = internal::array_size<typename TensorEvaluator<ArgType, Device>::Dimensions>::value;
  typedef DSizes<Index, NumDims> Dimensions;

  enum {
    IsAligned = /*TensorEvaluator<ArgType, Device>::IsAligned*/false,
    PacketAccess = /*TensorEvaluator<ArgType, Device>::PacketAccess*/false,
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
      : m_impl(op.expression(), device)
  {
    m_dimensions = m_impl.dimensions();
    for (int i = 0; i < NumDims; ++i) {
      m_dimensions[i] = ceilf(static_cast<float>(m_dimensions[i]) / op.strides()[i]);
    }

    const typename TensorEvaluator<ArgType, Device>::Dimensions& input_dims = m_impl.dimensions();
    for (int i = 0; i < NumDims; ++i) {
      if (i > 0) {
        m_inputStrides[i] = m_inputStrides[i-1] * input_dims[i-1];
        m_outputStrides[i] = m_outputStrides[i-1] * m_dimensions[i-1];
      } else {
        m_inputStrides[0] = 1;
        m_outputStrides[0] = 1;
      }
    }
    for (int i = 0; i < NumDims; ++i) {
        m_inputStrides[i] *= op.strides()[i];
    }
  }

  //  typedef typename XprType::Index Index;
  typedef typename XprType::Scalar Scalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dimensions; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(Scalar* data) {
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
      inputIndex += idx * m_inputStrides[i];
      index -= idx * m_outputStrides[i];
    }
    inputIndex += index * m_inputStrides[0];
    return m_impl.coeff(inputIndex);
  }

  /*  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const
  {
    return m_impl.template packet<LoadMode>(index);
    }*/

  Scalar* data() const { return NULL; }

 protected:
    //  Strides m_strides;
  Dimensions m_dimensions;
  array<Index, NumDims> m_outputStrides;
  array<Index, NumDims> m_inputStrides;
  TensorEvaluator<ArgType, Device> m_impl;
};


} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_STRIDING_H
