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
  typedef typename Derived::Scalar& CoeffReturnType;
  //typedef typename Derived::PacketScalar PacketScalar;
  typedef TensorEvaluator<Derived> nestedType;

  TensorEvaluator(Derived& m)
      : m_data(const_cast<Scalar*>(m.data()))
  { }

  CoeffReturnType coeff(Index index) const {
    return m_data[index];
  }

  Scalar& coeffRef(Index index) {
    return m_data[index];
  }

  // to do: vectorized evaluation.
  /*  template<int LoadMode>
  PacketReturnType packet(Index index) const
  {
    return ploadt<PacketScalar, LoadMode>(m_data + index);
  }

  template<int StoreMode>
  void writePacket(Index index, const PacketScalar& x)
  {
  return pstoret<Scalar, PacketScalar, StoreMode>(const_cast<Scalar*>(m_data) + index, x);
  }*/

 protected:
  Scalar* m_data;
};




// -------------------- CwiseUnaryOp --------------------

template<typename UnaryOp, typename ArgType>
struct TensorEvaluator<const TensorCwiseUnaryOp<UnaryOp, ArgType> >
{
  typedef TensorCwiseUnaryOp<UnaryOp, ArgType> XprType;
  typedef TensorEvaluator<ArgType> nestedType;

  TensorEvaluator(const XprType& op)
    : m_functor(op.functor()),
      m_argImpl(op.nestedExpression())
  { }

  typedef typename XprType::Index Index;
  typedef typename XprType::CoeffReturnType CoeffReturnType;

  CoeffReturnType coeff(Index index) const
  {
    return m_functor(m_argImpl.coeff(index));
  }

 private:
  const UnaryOp m_functor;
  typename TensorEvaluator<ArgType>::nestedType m_argImpl;
};


// -------------------- CwiseBinaryOp --------------------

template<typename BinaryOp, typename LeftArgType, typename RightArgType>
struct TensorEvaluator<const TensorCwiseBinaryOp<BinaryOp, LeftArgType, RightArgType> >
{
  typedef TensorCwiseBinaryOp<BinaryOp, LeftArgType, RightArgType> XprType;
  typedef TensorEvaluator<LeftArgType> leftType;
  typedef TensorEvaluator<RightArgType> rightType;

  TensorEvaluator(const XprType& op)
    : m_functor(op.functor()),
      m_leftImpl(op.lhsExpression()),
      m_rightImpl(op.rhsExpression())
  { }

  typedef typename XprType::Index Index;
  typedef typename XprType::CoeffReturnType CoeffReturnType;

  CoeffReturnType coeff(Index index) const
  {
    return m_functor(m_leftImpl.coeff(index), m_rightImpl.coeff(index));
  }

 private:
  const BinaryOp m_functor;
  typename TensorEvaluator<LeftArgType>::nestedType m_leftImpl;
  typename TensorEvaluator<RightArgType>::nestedType m_rightImpl;
};

} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_EVALUATOR_H
