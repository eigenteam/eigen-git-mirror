// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_EVALUATOR_H
#define EIGEN_CXX11_TENSOR_TENSOR_EVALUATOR_H

namespace Eigen {

/** \class TensorEvaluator
  * \ingroup CXX11_Tensor_Module
  *
  * \brief The tensor evaluator classes.
  *
  * These classes are responsible for the evaluation of the tensor expression.
  *
  * TODO: add support for more types of expressions, in particular expressions
  * leading to lvalues (slicing, reshaping, etc...)
  * TODO: add support for vectorization
  */

template<typename Derived>
struct TensorEvaluator
{
  typedef typename Derived::Index Index;
  typedef typename Derived::Scalar Scalar;
  typedef typename Derived::Packet Packet;
  typedef typename Derived::Scalar CoeffReturnType;
  typedef typename Derived::Packet PacketReturnType;

  enum {
    IsAligned = Derived::IsAligned,
    PacketAccess = Derived::PacketAccess,
  };

  TensorEvaluator(Derived& m)
      : m_data(const_cast<Scalar*>(m.data()))
  { }

  EIGEN_DEVICE_FUNC CoeffReturnType coeff(Index index) const {
    return m_data[index];
  }

  EIGEN_DEVICE_FUNC Scalar& coeffRef(Index index) {
    return m_data[index];
  }

  template<int LoadMode>
  PacketReturnType packet(Index index) const
  {
    return internal::ploadt<Packet, LoadMode>(m_data + index);
  }

  template <int StoreMode>
  void writePacket(Index index, const Packet& x)
  {
    return internal::pstoret<Scalar, Packet, StoreMode>(m_data + index, x);
  }

 protected:
  Scalar* m_data;
};




// -------------------- CwiseUnaryOp --------------------

template<typename UnaryOp, typename ArgType>
struct TensorEvaluator<const TensorCwiseUnaryOp<UnaryOp, ArgType> >
{
  typedef TensorCwiseUnaryOp<UnaryOp, ArgType> XprType;

  enum {
    IsAligned = TensorEvaluator<ArgType>::IsAligned,
    PacketAccess = TensorEvaluator<ArgType>::PacketAccess & internal::functor_traits<UnaryOp>::PacketAccess,
  };

  TensorEvaluator(const XprType& op)
    : m_functor(op.functor()),
      m_argImpl(op.nestedExpression())
  { }

  typedef typename XprType::Index Index;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;

  EIGEN_DEVICE_FUNC CoeffReturnType coeff(Index index) const
  {
    return m_functor(m_argImpl.coeff(index));
  }

  template<int LoadMode>
  EIGEN_DEVICE_FUNC PacketReturnType packet(Index index) const
  {
    return m_functor.packetOp(m_argImpl.template packet<LoadMode>(index));
  }

 private:
  const UnaryOp m_functor;
  TensorEvaluator<ArgType> m_argImpl;
};


// -------------------- CwiseBinaryOp --------------------

template<typename BinaryOp, typename LeftArgType, typename RightArgType>
struct TensorEvaluator<const TensorCwiseBinaryOp<BinaryOp, LeftArgType, RightArgType> >
{
  typedef TensorCwiseBinaryOp<BinaryOp, LeftArgType, RightArgType> XprType;

  enum {
    IsAligned = TensorEvaluator<LeftArgType>::IsAligned & TensorEvaluator<RightArgType>::IsAligned,
    PacketAccess = TensorEvaluator<LeftArgType>::PacketAccess & TensorEvaluator<RightArgType>::PacketAccess &
                   internal::functor_traits<BinaryOp>::PacketAccess,
  };

  TensorEvaluator(const XprType& op)
    : m_functor(op.functor()),
      m_leftImpl(op.lhsExpression()),
      m_rightImpl(op.rhsExpression())
  { }

  typedef typename XprType::Index Index;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;

  EIGEN_DEVICE_FUNC CoeffReturnType coeff(Index index) const
  {
    return m_functor(m_leftImpl.coeff(index), m_rightImpl.coeff(index));
  }
  template<int LoadMode>
  EIGEN_DEVICE_FUNC PacketReturnType packet(Index index) const
  {
    return m_functor.packetOp(m_leftImpl.template packet<LoadMode>(index), m_rightImpl.template packet<LoadMode>(index));
  }

 private:
  const BinaryOp m_functor;
  TensorEvaluator<LeftArgType> m_leftImpl;
  TensorEvaluator<RightArgType> m_rightImpl;
};

} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_EVALUATOR_H
