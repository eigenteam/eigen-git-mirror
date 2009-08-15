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
  * This class represents an expression with a special assignment operator (operator=)
  * assuming no aliasing between the target expression and the source expression.
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

    /** Behaves like MatrixBase::lazyAssign() */
    template<typename OtherDerived>
    ExpressionType& operator=(const MatrixBase<OtherDerived>& other)
    {
      return m_expression.lazyAssign(other.derived());
    }

    // TODO could be removed if we decide that += is noalias by default
    template<typename OtherDerived>
    ExpressionType& operator+=(const MatrixBase<OtherDerived>& other)
    {
      return m_expression.lazyAssign(m_expression + other.derived());
    }

    // TODO could be removed if we decide that += is noalias by default
    template<typename OtherDerived>
    ExpressionType& operator-=(const MatrixBase<OtherDerived>& other)
    {
      return m_expression.lazyAssign(m_expression - other.derived());
    }

    // TODO could be removed if we decide that += is noalias by default
    template<typename ProductDerived, typename Lhs, typename Rhs>
    ExpressionType& operator+=(const ProductBase<ProductDerived, Lhs,Rhs>& other)
    { other.derived().addTo(m_expression); return m_expression; }

    // TODO could be removed if we decide that += is noalias by default
    template<typename ProductDerived, typename Lhs, typename Rhs>
    ExpressionType& operator-=(const ProductBase<ProductDerived, Lhs,Rhs>& other)
    { other.derived().subTo(m_expression); return m_expression; }

  protected:
    ExpressionType& m_expression;
};


/** \returns a pseudo expression of \c *this with an operator= assuming
  * no aliasing between \c *this and the source expression
  *
  * \sa class NoAlias
  */
template<typename Derived>
NoAlias<Derived> MatrixBase<Derived>::noalias()
{
  return derived();
}

#endif // EIGEN_NOALIAS_H
