// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2007-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
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

#ifndef EIGEN_DIAGONALPRODUCT_H
#define EIGEN_DIAGONALPRODUCT_H

template<typename MatrixType, typename DiagonalType, int Order>
struct ei_traits<DiagonalProduct<MatrixType, DiagonalType, Order> >
{
  typedef typename MatrixType::Scalar Scalar;
  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime,
    Flags = (unsigned int)(MatrixType::Flags) & HereditaryBits,
    CoeffReadCost = NumTraits<Scalar>::MulCost + MatrixType::CoeffReadCost + DiagonalType::DiagonalVectorType::CoeffReadCost
  };
};

template<typename MatrixType, typename DiagonalType, int Order>
class DiagonalProduct : ei_no_assignment_operator,
                        public MatrixBase<DiagonalProduct<MatrixType, DiagonalType, Order> >
{
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(DiagonalProduct)

    inline DiagonalProduct(const MatrixType& matrix, const DiagonalType& diagonal)
      : m_matrix(matrix), m_diagonal(diagonal)
    {
      ei_assert(diagonal.diagonal().size() == (Order == DiagonalOnTheLeft ? matrix.rows() : matrix.cols()));
    }

    inline int rows() const { return m_matrix.rows(); }
    inline int cols() const { return m_matrix.cols(); }

    const Scalar coeff(int row, int col) const
    {
      return m_diagonal.diagonal().coeff(Order == DiagonalOnTheLeft ? row : col) * m_matrix.coeff(row, col);
    }

  protected:
    const typename MatrixType::Nested m_matrix;
    const typename DiagonalType::Nested m_diagonal;
};

/** \returns the diagonal matrix product of \c *this by the diagonal matrix \a diagonal.
  */
template<typename Derived>
template<typename DiagonalDerived>
inline const DiagonalProduct<Derived, DiagonalDerived, DiagonalOnTheRight>
MatrixBase<Derived>::operator*(const DiagonalBase<DiagonalDerived> &diagonal) const
{
  return DiagonalProduct<Derived, DiagonalDerived, DiagonalOnTheRight>(derived(), diagonal.derived());
}

/** \returns the diagonal matrix product of \c *this by the matrix \a matrix.
  */
template<typename DiagonalDerived>
template<typename MatrixDerived>
inline const DiagonalProduct<MatrixDerived, DiagonalDerived, DiagonalOnTheLeft>
DiagonalBase<DiagonalDerived>::operator*(const MatrixBase<MatrixDerived> &matrix) const
{
  return DiagonalProduct<MatrixDerived, DiagonalDerived, DiagonalOnTheLeft>(matrix.derived(), derived());
}


#endif // EIGEN_DIAGONALPRODUCT_H
