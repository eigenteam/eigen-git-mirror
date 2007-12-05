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

template<typename MatrixType> class DynBlock
  : public MatrixBase<typename MatrixType::Scalar, DynBlock<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Ref MatRef;
    friend class MatrixBase<Scalar, DynBlock<MatrixType> >;
    
    static const int RowsAtCompileTime = Dynamic,
                     ColsAtCompileTime = Dynamic;

    DynBlock(const MatRef& matrix,
          int startRow, int startCol,
          int blockRows, int blockCols)
      : m_matrix(matrix), m_startRow(startRow), m_startCol(startCol),
                          m_blockRows(blockRows), m_blockCols(blockCols)
    {
      assert(startRow >= 0 && blockRows >= 1 && startRow + blockRows <= matrix.rows()
          && startCol >= 0 && blockCols >= 1 && startCol + blockCols <= matrix.rows());
    }
    
    DynBlock(const DynBlock& other)
      : m_matrix(other.m_matrix),
        m_startRow(other.m_startRow), m_startCol(other.m_startCol),
        m_blockRows(other.m_blockRows), m_blockCols(other.m_blockCols) {}
    
    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(DynBlock)
    
  private:
    const DynBlock& _ref() const { return *this; }
    int _rows() const { return m_blockRows; }
    int _cols() const { return m_blockCols; }
    
    Scalar& _write(int row, int col=0)
    {
      return m_matrix.write(row + m_startRow, col + m_startCol);
    }
    
    Scalar _read(int row, int col=0) const
    {
      return m_matrix.read(row + m_startRow, col + m_startCol);
    }
    
  protected:
    MatRef m_matrix;
    const int m_startRow, m_startCol, m_blockRows, m_blockCols;
};

template<typename Scalar, typename Derived>
DynBlock<Derived> MatrixBase<Scalar, Derived>
  ::dynBlock(int startRow, int startCol, int blockRows, int blockCols) const
{
  return DynBlock<Derived>(static_cast<Derived*>(const_cast<MatrixBase*>(this))->ref(),
                        startRow, startCol, blockRows, blockCols);
}

#endif // EIGEN_BLOCK_H
