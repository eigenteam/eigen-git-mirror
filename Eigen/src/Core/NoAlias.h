// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_NOALIAS_H
#define EIGEN_NOALIAS_H

/** \class NoAlias
  *
  * \brief Pseudo expression providing an operator = assuming no aliasing
  *
  * \param ExpressionType the type of the object on which to do the lazy assignment
  *
  * This class represents an expression with special assignment operators
  * assuming no aliasing between the target expression and the source expression.
  * More precisely it alloas to bypass the EvalBeforeAssignBit flag of the source expression.
  * It is the return type of MatrixBase::noalias()
  * and most of the time this is the only way it is used.
  *
  * \sa MatrixBase::noalias()
  */
template<typename ExpressionType>
class NoAlias
{
  public:
    NoAlias(ExpressionType& expression) : m_expression(expression) {}

    /** Behaves like MatrixBase::lazyAssign(other)
      * \sa MatrixBase::lazyAssign() */
    template<typename OtherDerived>
    EIGEN_STRONG_INLINE ExpressionType& operator=(const MatrixBase<OtherDerived>& other)
    { return m_expression.lazyAssign(other.derived()); }

    /** \sa MatrixBase::operator+= */
    template<typename OtherDerived>
    EIGEN_STRONG_INLINE ExpressionType& operator+=(const MatrixBase<OtherDerived>& other)
    { return m_expression.lazyAssign(m_expression + other.derived()); }

    /** \sa MatrixBase::operator-= */
    template<typename OtherDerived>
    EIGEN_STRONG_INLINE ExpressionType& operator-=(const MatrixBase<OtherDerived>& other)
    { return m_expression.lazyAssign(m_expression - other.derived()); }

#ifndef EIGEN_PARSED_BY_DOXYGEN
    template<typename ProductDerived, typename Lhs, typename Rhs>
    EIGEN_STRONG_INLINE ExpressionType& operator+=(const ProductBase<ProductDerived, Lhs,Rhs>& other)
    { other.derived().addTo(m_expression); return m_expression; }

    template<typename ProductDerived, typename Lhs, typename Rhs>
    EIGEN_STRONG_INLINE ExpressionType& operator-=(const ProductBase<ProductDerived, Lhs,Rhs>& other)
    { other.derived().subTo(m_expression); return m_expression; }
#endif

  protected:
    ExpressionType& m_expression;
};

/** \returns a pseudo expression of \c *this with an operator= assuming
  * no aliasing between \c *this and the source expression.
  * 
  * More precisely, noalias() allows to bypass the EvalBeforeAssignBit flag.
  * Currently, even though several expressions may alias, only product
  * expressions have this flag. Therefore, noalias() is only usefull when
  * the source expression contains a matrix product.
  * 
  * Here are some examples where noalias is usefull:
  * \code
  * D.noalias()  = A * B;
  * D.noalias() += A.transpose() * B;
  * D.noalias() -= 2 * A * B.adjoint();
  * \endcode
  *
  * On the other hand the following example will lead to a \b wrong result:
  * \code
  * A.noalias() = A * B;
  * \endcode
  * because the result matrix A is also an operand of the matrix product. Therefore,
  * there is no alternative than evaluating A * B in a temporary, that is the default
  * behavior when you write:
  * \code
  * A = A * B;
  * \endcode
  *
  * \sa class NoAlias
  */
template<typename Derived>
NoAlias<Derived> MatrixBase<Derived>::noalias()
{
  return derived();
}

#endif // EIGEN_NOALIAS_H
