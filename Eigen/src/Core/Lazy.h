// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
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

#ifndef EIGEN_LAZY_H
#define EIGEN_LAZY_H

/** \class Lazy
  *
  * \brief Expression with the lazy flag set
  *
  * \param ExpressionType the type of the object of which we are taking the lazy version
  *
  * This class represents the lazy version of an expression.
  * It is the return type of MatrixBase::lazy()
  * and most of the time this is the only way it is used.
  *
  * \sa MatrixBase::lazy()
  */
template<typename ExpressionType>
struct ei_traits<Lazy<ExpressionType> >
{
  typedef typename ExpressionType::Scalar Scalar;
  enum {
    RowsAtCompileTime = ExpressionType::RowsAtCompileTime,
    ColsAtCompileTime = ExpressionType::ColsAtCompileTime,
    MaxRowsAtCompileTime = ExpressionType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = ExpressionType::MaxColsAtCompileTime,
    Flags = ExpressionType::Flags & ~(EvalBeforeNestingBit | EvalBeforeAssigningBit),
    CoeffReadCost = ExpressionType::CoeffReadCost
  };
};

template<typename ExpressionType> class Lazy
  : public MatrixBase<Lazy<ExpressionType> >
{
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(Lazy)

    Lazy(const ExpressionType& matrix) : m_expression(matrix) {}

    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Lazy)

  private:

    int _rows() const { return m_expression.rows(); }
    int _cols() const { return m_expression.cols(); }

    const Scalar _coeff(int row, int col) const
    {
      return m_expression.coeff(row, col);
    }

  protected:
    const typename ExpressionType::XprCopy m_expression;
};

/** \returns an expression of the lazy version of *this.
  *
  * Example: \include MatrixBase_lazy.cpp
  * Output: \verbinclude MatrixBase_lazy.out
  */
template<typename Derived>
const Lazy<Derived>
MatrixBase<Derived>::lazy() const
{
  return Lazy<Derived>(derived());
}

#endif // EIGEN_LAZY_H
