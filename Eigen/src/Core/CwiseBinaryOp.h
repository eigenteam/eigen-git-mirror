// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
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

#ifndef EIGEN_CWISE_BINARY_OP_H
#define EIGEN_CWISE_BINARY_OP_H

/** \class CwiseBinaryOp
  *
  * \brief Generic expression of a coefficient-wise operator between two matrices or vectors
  *
  * \param BinaryOp template functor implementing the operator
  * \param Lhs the type of the left-hand side
  * \param Rhs the type of the right-hand side
  *
  * This class represents an expression of a generic binary operator of two matrices or vectors.
  * It is the return type of the operator+, operator-, and the Cwise methods, and most
  * of the time this is the only way it is used.
  *
  * However, if you want to write a function returning such an expression, you
  * will need to use this class.
  *
  * \sa MatrixBase::binaryExpr(const MatrixBase<OtherDerived> &,const CustomBinaryOp &) const, class CwiseUnaryOp, class CwiseNullaryOp
  */
template<typename BinaryOp, typename Lhs, typename Rhs>
struct ei_traits<CwiseBinaryOp<BinaryOp, Lhs, Rhs> >
{
  typedef typename ei_result_of<
                     BinaryOp(
                       typename Lhs::Scalar,
                       typename Rhs::Scalar
                     )
                   >::type Scalar;
  typedef typename Lhs::Nested LhsNested;
  typedef typename Rhs::Nested RhsNested;
  typedef typename ei_unref<LhsNested>::type _LhsNested;
  typedef typename ei_unref<RhsNested>::type _RhsNested;
  enum {
    LhsCoeffReadCost = _LhsNested::CoeffReadCost,
    RhsCoeffReadCost = _RhsNested::CoeffReadCost,
    LhsFlags = _LhsNested::Flags,
    RhsFlags = _RhsNested::Flags,
    RowsAtCompileTime = Lhs::RowsAtCompileTime,
    ColsAtCompileTime = Lhs::ColsAtCompileTime,
    MaxRowsAtCompileTime = Lhs::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = Lhs::MaxColsAtCompileTime,
    Flags = (int(LhsFlags) | int(RhsFlags)) & (
        HereditaryBits
      | (int(LhsFlags) & int(RhsFlags) & (LinearAccessBit | AlignedBit))
      | (ei_functor_traits<BinaryOp>::PacketAccess && ((int(LhsFlags) & RowMajorBit)==(int(RhsFlags) & RowMajorBit))
        ? (int(LhsFlags) & int(RhsFlags) & PacketAccessBit) : 0)),
    CoeffReadCost = LhsCoeffReadCost + RhsCoeffReadCost + ei_functor_traits<BinaryOp>::Cost
  };
};

template<typename BinaryOp, typename Lhs, typename Rhs>
class CwiseBinaryOp : ei_no_assignment_operator,
  public MatrixBase<CwiseBinaryOp<BinaryOp, Lhs, Rhs> >
{
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(CwiseBinaryOp)
    typedef typename ei_traits<CwiseBinaryOp>::LhsNested LhsNested;
    typedef typename ei_traits<CwiseBinaryOp>::RhsNested RhsNested;

    class InnerIterator;

    inline CwiseBinaryOp(const Lhs& lhs, const Rhs& rhs, const BinaryOp& func = BinaryOp())
      : m_lhs(lhs), m_rhs(rhs), m_functor(func)
    {
      ei_assert(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols());
    }

    inline int rows() const { return m_lhs.rows(); }
    inline int cols() const { return m_lhs.cols(); }

    inline const Scalar coeff(int row, int col) const
    {
      return m_functor(m_lhs.coeff(row, col), m_rhs.coeff(row, col));
    }

    template<int LoadMode>
    inline PacketScalar packet(int row, int col) const
    {
      return m_functor.packetOp(m_lhs.template packet<LoadMode>(row, col), m_rhs.template packet<LoadMode>(row, col));
    }

    inline const Scalar coeff(int index) const
    {
      return m_functor(m_lhs.coeff(index), m_rhs.coeff(index));
    }

    template<int LoadMode>
    inline PacketScalar packet(int index) const
    {
      return m_functor.packetOp(m_lhs.template packet<LoadMode>(index), m_rhs.template packet<LoadMode>(index));
    }

  protected:
    const LhsNested m_lhs;
    const RhsNested m_rhs;
    const BinaryOp m_functor;
};

/**\returns an expression of the difference of \c *this and \a other
  *
  * \note If you want to substract a given scalar from all coefficients, see Cwise::operator-().
  *
  * \sa class CwiseBinaryOp, MatrixBase::operator-=(), Cwise::operator-()
  */
template<typename Derived>
template<typename OtherDerived>
inline const CwiseBinaryOp<ei_scalar_difference_op<typename ei_traits<Derived>::Scalar>,
                                 Derived, OtherDerived>
MatrixBase<Derived>::operator-(const MatrixBase<OtherDerived> &other) const
{
  return CwiseBinaryOp<ei_scalar_difference_op<Scalar>,
                       Derived, OtherDerived>(derived(), other.derived());
}

/** replaces \c *this by \c *this - \a other.
  *
  * \returns a reference to \c *this
  */
template<typename Derived>
template<typename OtherDerived>
inline Derived &
MatrixBase<Derived>::operator-=(const MatrixBase<OtherDerived> &other)
{
  return *this = *this - other;
}

/** \relates MatrixBase
  *
  * \returns an expression of the sum of \c *this and \a other
  *
  * \note If you want to add a given scalar to all coefficients, see Cwise::operator+().
  *
  * \sa class CwiseBinaryOp, MatrixBase::operator+=(), Cwise::operator+()
  */
template<typename Derived>
template<typename OtherDerived>
inline const CwiseBinaryOp<ei_scalar_sum_op<typename ei_traits<Derived>::Scalar>, Derived, OtherDerived>
MatrixBase<Derived>::operator+(const MatrixBase<OtherDerived> &other) const
{
  return CwiseBinaryOp<ei_scalar_sum_op<Scalar>, Derived, OtherDerived>(derived(), other.derived());
}

/** replaces \c *this by \c *this + \a other.
  *
  * \returns a reference to \c *this
  */
template<typename Derived>
template<typename OtherDerived>
inline Derived &
MatrixBase<Derived>::operator+=(const MatrixBase<OtherDerived>& other)
{
  return *this = *this + other;
}

/** \returns an expression of the Schur product (coefficient wise product) of *this and \a other
  *
  * Example: \include Cwise_product.cpp
  * Output: \verbinclude Cwise_product.out
  *
  * \sa class CwiseBinaryOp, operator/(), square()
  */
template<typename ExpressionType>
template<typename OtherDerived>
inline const EIGEN_CWISE_BINOP_RETURN_TYPE(ei_scalar_product_op)
Cwise<ExpressionType>::operator*(const MatrixBase<OtherDerived> &other) const
{
  return EIGEN_CWISE_BINOP_RETURN_TYPE(ei_scalar_product_op)(_expression(), other.derived());
}

/** \returns an expression of the coefficient-wise quotient of *this and \a other
  *
  * Example: \include Cwise_quotient.cpp
  * Output: \verbinclude Cwise_quotient.out
  *
  * \sa class CwiseBinaryOp, operator*(), inverse()
  */
template<typename ExpressionType>
template<typename OtherDerived>
inline const EIGEN_CWISE_BINOP_RETURN_TYPE(ei_scalar_quotient_op)
Cwise<ExpressionType>::operator/(const MatrixBase<OtherDerived> &other) const
{
  return EIGEN_CWISE_BINOP_RETURN_TYPE(ei_scalar_quotient_op)(_expression(), other.derived());
}

/** \returns an expression of the coefficient-wise min of *this and \a other
  *
  * Example: \include Cwise_min.cpp
  * Output: \verbinclude Cwise_min.out
  *
  * \sa class CwiseBinaryOp
  */
template<typename ExpressionType>
template<typename OtherDerived>
inline const EIGEN_CWISE_BINOP_RETURN_TYPE(ei_scalar_min_op)
Cwise<ExpressionType>::min(const MatrixBase<OtherDerived> &other) const
{
  return EIGEN_CWISE_BINOP_RETURN_TYPE(ei_scalar_min_op)(_expression(), other.derived());
}

/** \returns an expression of the coefficient-wise max of *this and \a other
  *
  * Example: \include Cwise_max.cpp
  * Output: \verbinclude Cwise_max.out
  *
  * \sa class CwiseBinaryOp
  */
template<typename ExpressionType>
template<typename OtherDerived>
inline const EIGEN_CWISE_BINOP_RETURN_TYPE(ei_scalar_max_op)
Cwise<ExpressionType>::max(const MatrixBase<OtherDerived> &other) const
{
  return EIGEN_CWISE_BINOP_RETURN_TYPE(ei_scalar_max_op)(_expression(), other.derived());
}

/** \returns an expression of a custom coefficient-wise operator \a func of *this and \a other
  *
  * The template parameter \a CustomBinaryOp is the type of the functor
  * of the custom operator (see class CwiseBinaryOp for an example)
  *
  * \addexample CustomCwiseBinaryFunctors \label How to use custom coeff wise binary functors
  *
  * Here is an example illustrating the use of custom functors:
  * \include class_CwiseBinaryOp.cpp
  * Output: \verbinclude class_CwiseBinaryOp.out
  *
  * \sa class CwiseBinaryOp, MatrixBase::operator+, MatrixBase::operator-, Cwise::operator*, Cwise::operator/
  */
template<typename Derived>
template<typename CustomBinaryOp, typename OtherDerived>
inline const CwiseBinaryOp<CustomBinaryOp, Derived, OtherDerived>
MatrixBase<Derived>::binaryExpr(const MatrixBase<OtherDerived> &other, const CustomBinaryOp& func) const
{
  return CwiseBinaryOp<CustomBinaryOp, Derived, OtherDerived>(derived(), other.derived(), func);
}

#endif // EIGEN_CWISE_BINARY_OP_H
