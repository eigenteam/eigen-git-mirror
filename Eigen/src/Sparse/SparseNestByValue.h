// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_SPARSENESTBYVALUE_H
#define EIGEN_SPARSENESTBYVALUE_H

/** \class SparseNestByValue
  *
  * \brief Expression which must be nested by value
  *
  * \param ExpressionType the type of the object of which we are requiring nesting-by-value
  *
  * This class is the return type of MatrixBase::nestByValue()
  * and most of the time this is the only way it is used.
  *
  * \sa SparseMatrixBase::nestByValue(), class NestByValue
  */
template<typename ExpressionType>
struct ei_traits<SparseNestByValue<ExpressionType> > : public ei_traits<ExpressionType>
{};

template<typename ExpressionType> class SparseNestByValue
  : public SparseMatrixBase<SparseNestByValue<ExpressionType> >
{
  public:

    typedef typename ExpressionType::InnerIterator InnerIterator;

    EIGEN_SPARSE_GENERIC_PUBLIC_INTERFACE(SparseNestByValue)

    inline SparseNestByValue(const ExpressionType& matrix) : m_expression(matrix) {}

    EIGEN_STRONG_INLINE int rows() const { return m_expression.rows(); }
    EIGEN_STRONG_INLINE int cols() const { return m_expression.cols(); }

    operator const ExpressionType&() const { return m_expression; }

  protected:
    const ExpressionType m_expression;
};

/** \returns an expression of the temporary version of *this.
  */
template<typename Derived>
inline const SparseNestByValue<Derived>
SparseMatrixBase<Derived>::nestByValue() const
{
  return SparseNestByValue<Derived>(derived());
}

// template<typename MatrixType>
// class SparseNestByValue<MatrixType>::InnerIterator : public MatrixType::InnerIterator
// {
//     typedef typename MatrixType::InnerIterator Base;
//   public:
// 
//     EIGEN_STRONG_INLINE InnerIterator(const SparseNestByValue& expr, int outer)
//       : Base(expr.m_expression, outer)
//     {}
// };

#endif // EIGEN_SPARSENESTBYVALUE_H
