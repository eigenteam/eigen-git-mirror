// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
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

#ifndef EIGEN_FIXEDBLOCK_H
#define EIGEN_FIXEDBLOCK_H

/** \class FixedBlock
  *
  * \brief Expression of a fixed-size block
  *
  * \param MatrixType the type of the object in which we are taking a block
  * \param BlockRows the number of rows of the block we are taking
  * \param BlockCols the number of columns of the block we are taking
  *
  * This class represents an expression of a fixed-size block. It is the return
  * type of MatrixBase::fixedBlock() and most of the time this is the only way it
  * is used.
  *
  * However, if you want to directly maniputate fixed-size block expressions,
  * for instance if you want to write a function returning such an expression, you
  * will need to use this class.
  *
  * Here is an example illustrating this:
  * \include class_FixedBlock.cpp
  * Output: \verbinclude class_FixedBlock.out
  *
  * \sa MatrixBase::fixedBlock(int,int), class Block
  */
template<typename MatrixType, int BlockRows, int BlockCols> class FixedBlock
  : public MatrixBase<typename MatrixType::Scalar,
                      FixedBlock<MatrixType, BlockRows, BlockCols> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Ref MatRef;
    friend class MatrixBase<Scalar, FixedBlock>;
    typedef MatrixBase<Scalar, FixedBlock> Base;

    FixedBlock(const MatRef& matrix, int startRow, int startCol)
      : m_matrix(matrix), m_startRow(startRow), m_startCol(startCol)
    {
      assert(startRow >= 0 && BlockRows >= 1 && startRow + BlockRows <= matrix.rows()
          && startCol >= 0 && BlockCols >= 1 && startCol + BlockCols <= matrix.cols());
    }
    
    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(FixedBlock)
    
  private:
    enum{
      RowsAtCompileTime = BlockRows,
      ColsAtCompileTime = BlockCols,
      MaxRowsAtCompileTime = BlockRows,
      MaxColsAtCompileTime = BlockCols
    };

    const FixedBlock& _ref() const { return *this; }
    static int _rows() { return BlockRows; }
    static int _cols() { return BlockCols; }
    
    Scalar& _coeffRef(int row, int col)
    {
      return m_matrix.coeffRef(row + m_startRow.value(), col + m_startCol.value());
    }
    
    Scalar _coeff(int row, int col) const
    {
      return m_matrix.coeff(row + m_startRow.value(), col + m_startCol.value());
    }
    
  protected:
    MatRef m_matrix;
    IntAtRunTimeIfDynamic<MatrixType::Traits::RowsAtCompileTime == 1 ? 0 : Dynamic>
      m_startRow;
    IntAtRunTimeIfDynamic<MatrixType::Traits::ColsAtCompileTime == 1 ? 0 : Dynamic>
      m_startCol;
};

/** \returns a fixed-size expression of a block in *this.
  *
  * The template parameters \a BlockRows and \a BlockCols are the number of
  * rows and columns in the block.
  *
  * \param startRow the first row in the block
  * \param startCol the first column in the block
  *
  * Example: \include MatrixBase_fixedBlock_int_int.cpp
  * Output: \verbinclude MatrixBase_fixedBlock_int_int.out
  *
  * \sa class FixedBlock, block(int,int,int,int)
  */
template<typename Scalar, typename Derived>
template<int BlockRows, int BlockCols>
FixedBlock<Derived, BlockRows, BlockCols> MatrixBase<Scalar, Derived>
  ::fixedBlock(int startRow, int startCol)
{
  return FixedBlock<Derived, BlockRows, BlockCols>(ref(), startRow, startCol);
}

/** This is the const version of fixedBlock(int, int). */
template<typename Scalar, typename Derived>
template<int BlockRows, int BlockCols>
const FixedBlock<Derived, BlockRows, BlockCols> MatrixBase<Scalar, Derived>
  ::fixedBlock(int startRow, int startCol) const
{
  return FixedBlock<Derived, BlockRows, BlockCols>(ref(), startRow, startCol);
}

#endif // EIGEN_FIXEDBLOCK_H
