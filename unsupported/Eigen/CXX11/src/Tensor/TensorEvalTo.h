// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_EVAL_TO_H
#define EIGEN_CXX11_TENSOR_TENSOR_EVAL_TO_H

namespace Eigen {

/** \class TensorForcedEval
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor reshaping class.
  *
  *
  */
namespace internal {
template<typename XprType>
struct traits<TensorEvalToOp<XprType> >
{
  // Type promotion to handle the case where the types of the lhs and the rhs are different.
  typedef typename XprType::Scalar Scalar;
  typedef typename internal::packet_traits<Scalar>::type Packet;
  typedef typename traits<XprType>::StorageKind StorageKind;
  typedef typename traits<XprType>::Index Index;
  typedef typename XprType::Nested Nested;
  typedef typename remove_reference<Nested>::type _Nested;

  enum {
    Flags = 0,
  };
};

template<typename XprType>
struct eval<TensorEvalToOp<XprType>, Eigen::Dense>
{
  typedef const TensorEvalToOp<XprType>& type;
};

template<typename XprType>
struct nested<TensorEvalToOp<XprType>, 1, typename eval<TensorEvalToOp<XprType> >::type>
{
  typedef TensorEvalToOp<XprType> type;
};

}  // end namespace internal




template<typename XprType>
class TensorEvalToOp : public TensorBase<TensorEvalToOp<XprType> >
{
  public:
  typedef typename Eigen::internal::traits<TensorEvalToOp>::Scalar Scalar;
  typedef typename Eigen::internal::traits<TensorEvalToOp>::Packet Packet;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;
  typedef typename Eigen::internal::nested<TensorEvalToOp>::type Nested;
  typedef typename Eigen::internal::traits<TensorEvalToOp>::StorageKind StorageKind;
  typedef typename Eigen::internal::traits<TensorEvalToOp>::Index Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvalToOp(Scalar* buffer, const XprType& expr)
      : m_xpr(expr), m_buffer(buffer) {}

    EIGEN_DEVICE_FUNC
    const typename internal::remove_all<typename XprType::Nested>::type&
    expression() const { return m_xpr; }

    EIGEN_DEVICE_FUNC Scalar* buffer() const { return m_buffer; }

  protected:
    typename XprType::Nested m_xpr;
    Scalar* m_buffer;
};



template<typename ArgType, typename Device>
struct TensorEvaluator<const TensorEvalToOp<ArgType>, Device>
{
  typedef TensorEvalToOp<ArgType> XprType;
  typedef typename ArgType::Scalar Scalar;
  typedef typename ArgType::Packet Packet;
  typedef typename TensorEvaluator<ArgType, Device>::Dimensions Dimensions;

  enum {
    IsAligned = true,
    PacketAccess = true,
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
      : m_impl(op.expression(), device), m_device(device), m_buffer(op.buffer())
  { }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE ~TensorEvaluator() {
  }

  typedef typename XprType::Index Index;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;

  EIGEN_DEVICE_FUNC const Dimensions& dimensions() const { return m_impl.dimensions(); }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void evalSubExprsIfNeeded() {
    m_impl.evalSubExprsIfNeeded();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void evalScalar(Index i) {
    m_buffer[i] = m_impl.coeff(i);
  }
  EIGEN_STRONG_INLINE void evalPacket(Index i) {
    internal::pstoret<Scalar, Packet, Aligned>(m_buffer + i, m_impl.template packet<TensorEvaluator<ArgType, Device>::IsAligned ? Aligned : Unaligned>(i));
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_impl.cleanup();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const
  {
    return m_buffer[index];
  }

  template<int LoadMode>
  EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const
  {
    return internal::ploadt<Packet, LoadMode>(m_buffer + index);
  }

 private:
  TensorEvaluator<ArgType, Device> m_impl;
  const Device& m_device;
  Scalar* m_buffer;
};


} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_EVAL_TO_H
