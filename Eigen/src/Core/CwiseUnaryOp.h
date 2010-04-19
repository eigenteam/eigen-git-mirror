// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// Eigen is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// Alternatively, you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of
// the License, or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License or the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License and a copy of the GNU General Public License along with
// Eigen. If not, see <http://www.gnu.org/licenses/>.

#ifndef EIGEN_CWISE_UNARY_OP_H
#define EIGEN_CWISE_UNARY_OP_H

/** \class CwiseUnaryOp
  *
  * \brief Generic expression where a coefficient-wise unary operator is applied to an expression
  *
  * \param UnaryOp template functor implementing the operator
  * \param XprType the type of the expression to which we are applying the unary operator
  *
  * This class represents an expression where a unary operator is applied to an expression.
  * It is the return type of all operations taking exactly 1 input expression, regardless of the
  * presence of other inputs such as scalars. For example, the operator* in the expression 3*matrix
  * is considered unary, because only the right-hand side is an expression, and its
  * return type is a specialization of CwiseUnaryOp.
  *
  * Most of the time, this is the only way that it is used, so you typically don't have to name
  * CwiseUnaryOp types explicitly.
  *
  * \sa MatrixBase::unaryExpr(const CustomUnaryOp &) const, class CwiseBinaryOp, class CwiseNullaryOp
  */
template<typename UnaryOp, typename XprType>
struct ei_traits<CwiseUnaryOp<UnaryOp, XprType> >
 : ei_traits<XprType>
{
  typedef typename ei_result_of<
                     UnaryOp(typename XprType::Scalar)
                   >::type Scalar;
  typedef typename XprType::Nested XprTypeNested;
  typedef typename ei_unref<XprTypeNested>::type _XprTypeNested;
  enum {
    Flags = _XprTypeNested::Flags & (
      HereditaryBits | LinearAccessBit | AlignedBit
      | (ei_functor_traits<UnaryOp>::PacketAccess ? PacketAccessBit : 0)),
    CoeffReadCost = _XprTypeNested::CoeffReadCost + ei_functor_traits<UnaryOp>::Cost
  };
};

template<typename UnaryOp, typename XprType, typename StorageKind>
class CwiseUnaryOpImpl;

template<typename UnaryOp, typename XprType>
class CwiseUnaryOp : ei_no_assignment_operator,
  public CwiseUnaryOpImpl<UnaryOp, XprType, typename ei_traits<XprType>::StorageKind>
{
  public:

    typedef typename CwiseUnaryOpImpl<UnaryOp, XprType,typename ei_traits<XprType>::StorageKind>::Base Base;
    EIGEN_GENERIC_PUBLIC_INTERFACE_NEW(CwiseUnaryOp)

    inline CwiseUnaryOp(const XprType& xpr, const UnaryOp& func = UnaryOp())
      : m_xpr(xpr), m_functor(func) {}

    EIGEN_STRONG_INLINE int rows() const { return m_xpr.rows(); }
    EIGEN_STRONG_INLINE int cols() const { return m_xpr.cols(); }

    /** \returns the functor representing the unary operation */
    const UnaryOp& functor() const { return m_functor; }

    /** \returns the nested expression */
    const typename ei_cleantype<typename XprType::Nested>::type&
    nestedExpression() const { return m_xpr; }

    /** \returns the nested expression */
    typename ei_cleantype<typename XprType::Nested>::type&
    nestedExpression() { return m_xpr.const_cast_derived(); }

  protected:
    const typename XprType::Nested m_xpr;
    const UnaryOp m_functor;
};

// This is the generic implementation for dense storage.
// It can be used for any expression types implementing the dense concept.
template<typename UnaryOp, typename XprType>
class CwiseUnaryOpImpl<UnaryOp,XprType,Dense>
  : public XprType::template MakeBase< CwiseUnaryOp<UnaryOp, XprType> >::Type
 {
    typedef CwiseUnaryOp<UnaryOp, XprType> Derived;

  public:

    typedef typename XprType::template MakeBase< CwiseUnaryOp<UnaryOp, XprType> >::Type Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(Derived)

    EIGEN_STRONG_INLINE const Scalar coeff(int row, int col) const
    {
      return derived().functor()(derived().nestedExpression().coeff(row, col));
    }

    template<int LoadMode>
    EIGEN_STRONG_INLINE PacketScalar packet(int row, int col) const
    {
      return derived().functor().packetOp(derived().nestedExpression().template packet<LoadMode>(row, col));
    }

    EIGEN_STRONG_INLINE const Scalar coeff(int index) const
    {
      return derived().functor()(derived().nestedExpression().coeff(index));
    }

    template<int LoadMode>
    EIGEN_STRONG_INLINE PacketScalar packet(int index) const
    {
      return derived().functor().packetOp(derived().nestedExpression().template packet<LoadMode>(index));
    }
};

#endif // EIGEN_CWISE_UNARY_OP_H
