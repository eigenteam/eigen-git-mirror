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
  */

// Generic evaluator
template<typename Derived, typename Device>
struct TensorEvaluator
{
  typedef typename Derived::Index Index;
  typedef typename Derived::Scalar Scalar;
  typedef typename Derived::Packet Packet;
  typedef typename Derived::Scalar CoeffReturnType;
  typedef typename Derived::Packet PacketReturnType;
  typedef typename Derived::Dimensions Dimensions;

  enum {
    IsAligned = Derived::IsAligned,
    PacketAccess = Derived::PacketAccess,
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const Derived& m, const Device&)
      : m_data(const_cast<Scalar*>(m.data())), m_dims(m.dimensions())
  { }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dims; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void evalSubExprsIfNeeded() { }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() { }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const {
    eigen_assert(m_data);
    return m_data[index];
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar& coeffRef(Index index) {
    eigen_assert(m_data);
    return m_data[index];
  }

  template<int LoadMode> EIGEN_STRONG_INLINE
  PacketReturnType packet(Index index) const
  {
    return internal::ploadt<Packet, LoadMode>(m_data + index);
  }

  template <int StoreMode> EIGEN_STRONG_INLINE
  void writePacket(Index index, const Packet& x)
  {
    return internal::pstoret<Scalar, Packet, StoreMode>(m_data + index, x);
  }

 protected:
  Scalar* m_data;
  Dimensions m_dims;
};


// Default evaluator for rvalues
template<typename Derived, typename Device>
struct TensorEvaluator<const Derived, Device>
{
  typedef typename Derived::Index Index;
  typedef typename Derived::Scalar Scalar;
  typedef typename Derived::Packet Packet;
  typedef typename Derived::Scalar CoeffReturnType;
  typedef typename Derived::Packet PacketReturnType;
  typedef typename Derived::Dimensions Dimensions;

  enum {
    IsAligned = Derived::IsAligned,
    PacketAccess = Derived::PacketAccess,
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const Derived& m, const Device&)
      : m_data(m.data()), m_dims(m.dimensions())
  { }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dims; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void evalSubExprsIfNeeded() { }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() { }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const {
    eigen_assert(m_data);
    return m_data[index];
  }

  template<int LoadMode> EIGEN_STRONG_INLINE
  PacketReturnType packet(Index index) const
  {
    return internal::ploadt<Packet, LoadMode>(m_data + index);
  }

 protected:
  const Scalar* m_data;
  Dimensions m_dims;
};




// -------------------- CwiseNullaryOp --------------------

template<typename NullaryOp, typename ArgType, typename Device>
struct TensorEvaluator<const TensorCwiseNullaryOp<NullaryOp, ArgType>, Device>
{
  typedef TensorCwiseNullaryOp<NullaryOp, ArgType> XprType;

  enum {
    IsAligned = true,
    PacketAccess = internal::functor_traits<NullaryOp>::PacketAccess,
  };

  EIGEN_DEVICE_FUNC
  TensorEvaluator(const XprType& op, const Device& device)
      : m_functor(op.functor()), m_argImpl(op.nestedExpression(), device)
  { }

  typedef typename XprType::Index Index;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;
  typedef typename TensorEvaluator<ArgType, Device>::Dimensions Dimensions;

  EIGEN_DEVICE_FUNC const Dimensions& dimensions() const { return m_argImpl.dimensions(); }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void evalSubExprsIfNeeded() { }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() { }

  EIGEN_DEVICE_FUNC CoeffReturnType coeff(Index index) const
  {
    return m_functor(index);
  }

  template<int LoadMode>
  EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const
  {
    return m_functor.packetOp(index);
  }

 private:
  const NullaryOp m_functor;
  TensorEvaluator<ArgType, Device> m_argImpl;
};



// -------------------- CwiseUnaryOp --------------------

template<typename UnaryOp, typename ArgType, typename Device>
struct TensorEvaluator<const TensorCwiseUnaryOp<UnaryOp, ArgType>, Device>
{
  typedef TensorCwiseUnaryOp<UnaryOp, ArgType> XprType;

  enum {
    IsAligned = TensorEvaluator<ArgType, Device>::IsAligned,
    PacketAccess = TensorEvaluator<ArgType, Device>::PacketAccess & internal::functor_traits<UnaryOp>::PacketAccess,
  };

  EIGEN_DEVICE_FUNC TensorEvaluator(const XprType& op, const Device& device)
    : m_functor(op.functor()),
      m_argImpl(op.nestedExpression(), device)
  { }

  typedef typename XprType::Index Index;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;
  typedef typename TensorEvaluator<ArgType, Device>::Dimensions Dimensions;

  EIGEN_DEVICE_FUNC const Dimensions& dimensions() const { return m_argImpl.dimensions(); }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void evalSubExprsIfNeeded() {
    m_argImpl.evalSubExprsIfNeeded();
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_argImpl.cleanup();
  }

  EIGEN_DEVICE_FUNC CoeffReturnType coeff(Index index) const
  {
    return m_functor(m_argImpl.coeff(index));
  }

  template<int LoadMode>
  EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const
  {
    return m_functor.packetOp(m_argImpl.template packet<LoadMode>(index));
  }

 private:
  const UnaryOp m_functor;
  TensorEvaluator<ArgType, Device> m_argImpl;
};


// -------------------- CwiseBinaryOp --------------------

template<typename BinaryOp, typename LeftArgType, typename RightArgType, typename Device>
struct TensorEvaluator<const TensorCwiseBinaryOp<BinaryOp, LeftArgType, RightArgType>, Device>
{
  typedef TensorCwiseBinaryOp<BinaryOp, LeftArgType, RightArgType> XprType;

  enum {
    IsAligned = TensorEvaluator<LeftArgType, Device>::IsAligned & TensorEvaluator<RightArgType, Device>::IsAligned,
    PacketAccess = TensorEvaluator<LeftArgType, Device>::PacketAccess & TensorEvaluator<RightArgType, Device>::PacketAccess &
                   internal::functor_traits<BinaryOp>::PacketAccess,
  };

  EIGEN_DEVICE_FUNC TensorEvaluator(const XprType& op, const Device& device)
    : m_functor(op.functor()),
      m_leftImpl(op.lhsExpression(), device),
      m_rightImpl(op.rhsExpression(), device)
  { }

  typedef typename XprType::Index Index;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;
  typedef typename TensorEvaluator<LeftArgType, Device>::Dimensions Dimensions;

  EIGEN_DEVICE_FUNC const Dimensions& dimensions() const
  {
    // TODO: use right impl instead if right impl dimensions are known at compile time.
    return m_leftImpl.dimensions();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void evalSubExprsIfNeeded() {
    m_leftImpl.evalSubExprsIfNeeded();
    m_rightImpl.evalSubExprsIfNeeded();
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_leftImpl.cleanup();
    m_rightImpl.cleanup();
  }

  EIGEN_DEVICE_FUNC CoeffReturnType coeff(Index index) const
  {
    return m_functor(m_leftImpl.coeff(index), m_rightImpl.coeff(index));
  }
  template<int LoadMode>
  EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const
  {
    return m_functor.packetOp(m_leftImpl.template packet<LoadMode>(index), m_rightImpl.template packet<LoadMode>(index));
  }

 private:
  const BinaryOp m_functor;
  TensorEvaluator<LeftArgType, Device> m_leftImpl;
  TensorEvaluator<RightArgType, Device> m_rightImpl;
};


// -------------------- SelectOp --------------------

template<typename IfArgType, typename ThenArgType, typename ElseArgType, typename Device>
struct TensorEvaluator<const TensorSelectOp<IfArgType, ThenArgType, ElseArgType>, Device>
{
  typedef TensorSelectOp<IfArgType, ThenArgType, ElseArgType> XprType;

  enum {
    IsAligned = TensorEvaluator<ThenArgType, Device>::IsAligned & TensorEvaluator<ElseArgType, Device>::IsAligned,
    PacketAccess = TensorEvaluator<ThenArgType, Device>::PacketAccess & TensorEvaluator<ElseArgType, Device>::PacketAccess/* &
                                                                                                             TensorEvaluator<IfArgType>::PacketAccess*/,
  };

  EIGEN_DEVICE_FUNC TensorEvaluator(const XprType& op, const Device& device)
    : m_condImpl(op.ifExpression(), device),
      m_thenImpl(op.thenExpression(), device),
      m_elseImpl(op.elseExpression(), device)
  { }

  typedef typename XprType::Index Index;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;
  typedef typename TensorEvaluator<IfArgType, Device>::Dimensions Dimensions;

  EIGEN_DEVICE_FUNC const Dimensions& dimensions() const
  {
    // TODO: use then or else impl instead if they happen to be known at compile time.
    return m_condImpl.dimensions();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void evalSubExprsIfNeeded() {
    m_condImpl.evalSubExprsIfNeeded();
    m_thenImpl.evalSubExprsIfNeeded();
    m_elseImpl.evalSubExprsIfNeeded();
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_condImpl.cleanup();
    m_thenImpl.cleanup();
    m_elseImpl.cleanup();
  }

  EIGEN_DEVICE_FUNC CoeffReturnType coeff(Index index) const
  {
    return m_condImpl.coeff(index) ? m_thenImpl.coeff(index) : m_elseImpl.coeff(index);
  }
  template<int LoadMode>
  PacketReturnType packet(Index index) const
  {
    static const int PacketSize = internal::unpacket_traits<PacketReturnType>::size;
    internal::Selector<PacketSize> select;
    for (Index i = 0; i < PacketSize; ++i) {
      select.select[i] = m_condImpl.coeff(index+i);
    }
    return internal::pblend(select,
                            m_thenImpl.template packet<LoadMode>(index),
                            m_elseImpl.template packet<LoadMode>(index));
  }

 private:
  TensorEvaluator<IfArgType, Device> m_condImpl;
  TensorEvaluator<ThenArgType, Device> m_thenImpl;
  TensorEvaluator<ElseArgType, Device> m_elseImpl;
};


} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_EVALUATOR_H
