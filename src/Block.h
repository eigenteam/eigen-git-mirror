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

#ifndef EIGEN_BLOCK_H
#define EIGEN_BLOCK_H

namespace Eigen {

template<typename MatrixType> class MatrixBlock
{
  public:
    typedef typename MatrixType::Scalar Scalar;

    MatrixBlock(const MatrixType& matrix, int startRow, int endRow,
                                          int startCol = 0, int endCol = 0)
      : m_matrix(matrix), m_startRow(startRow), m_endRow(endRow),
                          m_startCol(startCol), m_endCol(endCol)
    {
      assert(startRow >= 0 && startRow <= endRow && endRow < matrix.rows()
          && startCol >= 0 && startCol <= endCol && endCol < matrix.cols());
    }
    
    MatrixBlock(const MatrixBlock& other)
      : m_matrix(other.m_matrix), m_startRow(other.m_startRow), m_endRow(other.m_endRow),
                                  m_startCol(other.m_startCol), m_endCol(other.m_endCol) {}
    
    int rows() const { return m_endRow - m_startRow + 1; }
    int cols() const { return m_endCol - m_startCol + 1; }
    
    Scalar& write(int row, int col=0)
    {
      return m_matrix.write(row + m_startRow, col + m_startCol);
    }
    
    Scalar read(int row, int col=0) const
    {
      return m_matrix.read(row + m_startRow, col + m_startCol);
    }
    
  protected:
    MatrixType m_matrix;
    const int m_startRow, m_endRow, m_startCol, m_endCol;
};

template<typename Derived>
MatrixXpr<
  MatrixBlock<
    MatrixRef<
      MatrixBase<Derived>
    >
  >
>
MatrixBase<Derived>::block(int startRow, int endRow, int startCol, int endCol)
{
  typedef MatrixBlock<Ref> ProductType;
  typedef MatrixXpr<ProductType> XprType;
  return XprType(ProductType(ref(), startRow, endRow, startCol, endCol));
}

template<typename Content>
MatrixXpr<
  MatrixBlock<
    MatrixXpr<Content>
  >
>
MatrixXpr<Content>::block(int startRow, int endRow, int startCol, int endCol)
{
  typedef MatrixBlock<
            MatrixXpr<Content>
          > ProductType;
  typedef MatrixXpr<ProductType> XprType;
  return XprType(ProductType(*this, startRow, endRow, startCol, endCol));
}

}

#endif // EIGEN_BLOCK_H
