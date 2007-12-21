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

#ifndef EIGEN_DYNBLOCK_H
#define EIGEN_DYNBLOCK_H

/** \class DynBlock
  *
  * \brief Expression of a dynamic-size block
  *
  * This class represents an expression of a dynamic-size block. It is the return
  * type of MatrixBase::dynBlock() and most of the time this is the only way this
  * class is used.
  *
  * However, if you want to directly maniputate dynamic-size block expressions,
  * for instance if you want to write a function returning such an expression, you
  * will need to use this class.
  *
  * Here is an example illustrating this:
  * \include class_DynBlock.cpp
  * Output:
  * \verbinclude class_DynBlock.out
  *
  * \sa MatrixBase::dynBlock()
  */
template<typename MatrixType> class DynBlock
  : public MatrixBase<typename MatrixType::Scalar, DynBlock<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Ref MatRef;
    friend class MatrixBase<Scalar, DynBlock<MatrixType> >;
    
    DynBlock(const MatRef& matrix,
          int startRow, int startCol,
          int blockRows, int blockCols)
      : m_matrix(matrix), m_startRow(startRow), m_startCol(startCol),
                          m_blockRows(blockRows), m_blockCols(blockCols)
    {
      assert(startRow >= 0 && blockRows >= 1 && startRow + blockRows <= matrix.rows()
          && startCol >= 0 && blockCols >= 1 && startCol + blockCols <= matrix.cols());
    }
    
    DynBlock(const DynBlock& other)
      : m_matrix(other.m_matrix),
        m_startRow(other.m_startRow), m_startCol(other.m_startCol),
        m_blockRows(other.m_blockRows), m_blockCols(other.m_blockCols) {}
    
    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(DynBlock)
    
  private:
    static const int
      _RowsAtCompileTime = MatrixType::RowsAtCompileTime == 1 ? 1 : Dynamic,
      _ColsAtCompileTime = MatrixType::ColsAtCompileTime == 1 ? 1 : Dynamic;

    const DynBlock& _ref() const { return *this; }
    int _rows() const { return m_blockRows; }
    int _cols() const { return m_blockCols; }
    
    Scalar& _coeffRef(int row, int col)
    {
      return m_matrix.coeffRef(row + m_startRow, col + m_startCol);
    }
    
    Scalar _coeff(int row, int col) const
    {
      return m_matrix.coeff(row + m_startRow, col + m_startCol);
    }
    
  protected:
    MatRef m_matrix;
    const int m_startRow, m_startCol, m_blockRows, m_blockCols;
};

/** \returns a dynamic-size expression of a block in *this.
  *
  * \param startRow the first row in the block
  * \param startCol the first column in the block
  * \param blockRows the number of rows in the block
  * \param blockCols the number of columns in the block
  *
  * Example:
  * \include MatrixBase_dynBlock.cpp
  * Output:
  * \verbinclude MatrixBase_dynBlock.out
  *
  * \sa class DynBlock
  */
template<typename Scalar, typename Derived>
DynBlock<Derived> MatrixBase<Scalar, Derived>
  ::dynBlock(int startRow, int startCol, int blockRows, int blockCols) const
{
  return DynBlock<Derived>(static_cast<Derived*>(const_cast<MatrixBase*>(this))->ref(),
                        startRow, startCol, blockRows, blockCols);
}

#endif // EIGEN_DYNBLOCK_H
