// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2007 Benoit Jacob <jacob@math.jussieu.fr>
//
// Eigen is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation; either version 2 or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along
// with Eigen; if not, write to the Free Software Foundation, Inc., 51
// Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
//
// As a special exception, if other files instantiate templates or use macros
// or functions from this file, or you compile this file and link it
// with other works to produce a work based on this file, this file does not
// by itself cause the resulting work to be covered by the GNU General Public
// License. This exception does not invalidate any other reasons why a work
// based on this file might be covered by the GNU General Public License.

#ifndef EIGEN_ROWANDCOL_H
#define EIGEN_ROWANDCOL_H

namespace Eigen {

template<typename MatrixType> class MatrixRow
{
  public:
    typedef typename MatrixType::Scalar Scalar;

    MatrixRow(const MatrixType& matrix, int row)
      : m_matrix(matrix), m_row(row)
    {
      EIGEN_CHECK_ROW_RANGE(matrix, row);
    }
    
    MatrixRow(const MatrixRow& other)
      : m_matrix(other.m_matrix), m_row(other.m_row) {}
    
    int rows() const { return m_matrix.cols(); }
    int cols() const { return 1; }
    
    Scalar& operator()(int row, int col=0)
    {
      EIGEN_UNUSED(col);
      EIGEN_CHECK_ROW_RANGE(*this, row);
      return m_matrix(m_row, row);
    }
    
    Scalar operator()(int row, int col=0) const
    {
      EIGEN_UNUSED(col);
      EIGEN_CHECK_ROW_RANGE(*this, row);
      return m_matrix(m_row, row);
    }
    
  protected:
    MatrixType m_matrix;
    const int m_row;
};

template<typename MatrixType> class MatrixCol
{
  public:
    typedef typename MatrixType::Scalar Scalar;

    MatrixCol(const MatrixType& matrix, int col)
      : m_matrix(matrix), m_col(col)
    {
      EIGEN_CHECK_COL_RANGE(matrix, col);
    }
    
    MatrixCol(const MatrixCol& other)
      : m_matrix(other.m_matrix), m_col(other.m_col) {}
    
    int rows() const { return m_matrix.rows(); }
    int cols() const { return 1; }
    
    Scalar& operator()(int row, int col=0)
    {
      EIGEN_UNUSED(col);
      EIGEN_CHECK_ROW_RANGE(*this, row);
      return m_matrix(row, m_col);
    }
    
    Scalar operator()(int row, int col=0) const
    {
      EIGEN_UNUSED(col);
      EIGEN_CHECK_ROW_RANGE(*this, row);
      return m_matrix(row, m_col);
    }
    
  protected:
    MatrixType m_matrix;
    const int m_col;
};

#define EIGEN_MAKE_ROW_COL_FUNCTIONS(func, Func) \
template<typename Derived> \
MatrixConstXpr< \
  Matrix##Func< \
    const MatrixConstRef< \
      MatrixBase<Derived> \
    > \
  > \
> \
MatrixBase<Derived>::func(int i) const\
{ \
  typedef Matrix##Func<const ConstRef> ProductType; \
  typedef MatrixConstXpr<ProductType> XprType; \
  return XprType(ProductType(constRef(), i)); \
} \
\
template<typename Content> \
MatrixConstXpr< \
  Matrix##Func< \
    const MatrixConstXpr<Content> \
  > \
> \
MatrixConstXpr<Content>::func(int i) const\
{ \
  typedef Matrix##Func< \
            const MatrixConstXpr<Content> \
          > ProductType; \
  typedef MatrixConstXpr<ProductType> XprType; \
  return XprType(ProductType(*this, i)); \
} \
\
template<typename Content> \
MatrixXpr< \
  Matrix##Func< \
    MatrixXpr<Content> \
  > \
> \
MatrixXpr<Content>::func(int i) \
{ \
  typedef Matrix##Func< \
            MatrixXpr<Content> \
          > ProductType; \
  typedef MatrixXpr<ProductType> XprType; \
  return XprType(ProductType(*this, i)); \
}

EIGEN_MAKE_ROW_COL_FUNCTIONS(row, Row)
EIGEN_MAKE_ROW_COL_FUNCTIONS(col, Col)

#undef EIGEN_MAKE_ROW_COL_FUNCTIONS

}

#endif // EIGEN_ROWANDCOL_H
