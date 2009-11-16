// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
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
  * \brief Generic expression of a coefficient-wise unary operator of a matrix or a vector
  *
  * \param UnaryOp template functor implementing the operator
  * \param MatrixType the type of the matrix we are applying the unary operator
  *
  * This class represents an expression of a generic unary operator of a matrix or a vector.
  * It is the return type of the unary operator-, of a matrix or a vector, and most
  * of the time this is the only way it is used.
  *
  * \sa MatrixBase::unaryExpr(const CustomUnaryOp &) const, class CwiseBinaryOp, class CwiseNullaryOp
  */
template<typename UnaryOp, typename MatrixType>
struct ei_traits<CwiseUnaryOp<UnaryOp, MatrixType> >
 : ei_traits<MatrixType>
{
  typedef typename ei_result_of<
                     UnaryOp(typename MatrixType::Scalar)
                   >::type Scalar;
  typedef typename MatrixType::Nested MatrixTypeNested;
  typedef typename ei_unref<MatrixTypeNested>::type _MatrixTypeNested;
  enum {
    Flags = (_MatrixTypeNested::Flags & (
      HereditaryBits | LinearAccessBit | AlignedBit
      | (ei_functor_traits<UnaryOp>::PacketAccess ? PacketAccessBit : 0))),
    CoeffReadCost = _MatrixTypeNested::CoeffReadCost + ei_functor_traits<UnaryOp>::Cost
  };
};

template<typename UnaryOp, typename MatrixType, typename StorageType>
class CwiseUnaryOpImpl;

template<typename UnaryOp, typename MatrixType>
class CwiseUnaryOp : ei_no_assignment_operator,
  public CwiseUnaryOpImpl<UnaryOp, MatrixType, typename ei_traits<MatrixType>::StorageType>
{
  public:

    typedef typename CwiseUnaryOpImpl<UnaryOp, MatrixType,typename ei_traits<MatrixType>::StorageType>::Base Base;
    EIGEN_GENERIC_PUBLIC_INTERFACE_NEW(CwiseUnaryOp)

    inline CwiseUnaryOp(const MatrixType& mat, const UnaryOp& func = UnaryOp())
      : m_matrix(mat), m_functor(func) {}

    EIGEN_STRONG_INLINE int rows() const { return m_matrix.rows(); }
    EIGEN_STRONG_INLINE int cols() const { return m_matrix.cols(); }

    /** \internal used for introspection */
    const UnaryOp& _functor() const { return m_functor; }

    /** \internal used for introspection */
    const typename ei_cleantype<typename MatrixType::Nested>::type&
    _expression() const { return m_matrix; }

    const typename ei_cleantype<typename MatrixType::Nested>::type&
    nestedExpression() const { return m_matrix; }

    typename ei_cleantype<typename MatrixType::Nested>::type&
    nestedExpression() { return m_matrix.const_cast_derived(); }

  protected:
    const typename MatrixType::Nested m_matrix;
    const UnaryOp m_functor;
};

template<typename UnaryOp, typename MatrixType>
class CwiseUnaryOpImpl<UnaryOp,MatrixType,Dense> : public MatrixBase<CwiseUnaryOp<UnaryOp, MatrixType> >
{
    const typename ei_cleantype<typename MatrixType::Nested>::type& matrix() const
    { return derived().nestedExpression(); }
    typename ei_cleantype<typename MatrixType::Nested>::type& matrix()
    { return derived().nestedExpression(); }

  public:

    typedef CwiseUnaryOp<UnaryOp, MatrixType> Derived;
    EIGEN_DENSE_PUBLIC_INTERFACE( Derived )

    EIGEN_STRONG_INLINE const Scalar coeff(int row, int col) const
    {
      return derived()._functor()(matrix().coeff(row, col));
    }

    template<int LoadMode>
    EIGEN_STRONG_INLINE PacketScalar packet(int row, int col) const
    {
      return derived()._functor().packetOp(matrix().template packet<LoadMode>(row, col));
    }

    EIGEN_STRONG_INLINE const Scalar coeff(int index) const
    {
      return derived()._functor()(matrix().coeff(index));
    }

    template<int LoadMode>
    EIGEN_STRONG_INLINE PacketScalar packet(int index) const
    {
      return derived()._functor().packetOp(matrix().template packet<LoadMode>(index));
    }
};

/** \returns an expression of the coefficient-wise absolute value of \c *this
  *
  * Example: \include Cwise_abs.cpp
  * Output: \verbinclude Cwise_abs.out
  *
  * \sa abs2()
  */
template<typename ExpressionType>
EIGEN_STRONG_INLINE const EIGEN_CWISE_UNOP_RETURN_TYPE(ei_scalar_abs_op)
Cwise<ExpressionType>::abs() const
{
  return _expression();
}

/** \returns an expression of the coefficient-wise squared absolute value of \c *this
  *
  * Example: \include Cwise_abs2.cpp
  * Output: \verbinclude Cwise_abs2.out
  *
  * \sa abs(), square()
  */
template<typename ExpressionType>
EIGEN_STRONG_INLINE const EIGEN_CWISE_UNOP_RETURN_TYPE(ei_scalar_abs2_op)
Cwise<ExpressionType>::abs2() const
{
  return _expression();
}

/** \returns an expression of the coefficient-wise exponential of *this.
  *
  * Example: \include Cwise_exp.cpp
  * Output: \verbinclude Cwise_exp.out
  *
  * \sa pow(), log(), sin(), cos()
  */
template<typename ExpressionType>
inline const EIGEN_CWISE_UNOP_RETURN_TYPE(ei_scalar_exp_op)
Cwise<ExpressionType>::exp() const
{
  return _expression();
}

/** \returns an expression of the coefficient-wise logarithm of *this.
  *
  * Example: \include Cwise_log.cpp
  * Output: \verbinclude Cwise_log.out
  *
  * \sa exp()
  */
template<typename ExpressionType>
inline const EIGEN_CWISE_UNOP_RETURN_TYPE(ei_scalar_log_op)
Cwise<ExpressionType>::log() const
{
  return _expression();
}

#endif // EIGEN_CWISE_UNARY_OP_H
