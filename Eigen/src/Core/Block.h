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

#ifndef EIGEN_BLOCK_H
#define EIGEN_BLOCK_H

/** \class Block
  *
  * \brief Expression of a dynamic-size block
  *
  * \param MatrixType the type of the object in which we are taking a block
  *
  * This class represents an expression of a dynamic-size block. It is the return
  * type of MatrixBase::block(int,int,int,int) and most of the time this is the only way it
  * is used.
  *
  * However, if you want to directly maniputate dynamic-size block expressions,
  * for instance if you want to write a function returning such an expression, you
  * will need to use this class.
  *
  * Here is an example illustrating this:
  * \include class_Block.cpp
  * Output: \verbinclude class_Block.out
  *
  * \note Even though this expression has dynamic size, in the case where \a MatrixType
  * has fixed size, this expression inherits a fixed maximal size which means that evaluating
  * it does not cause a dynamic memory allocation.
  *
  * \sa MatrixBase::block(int,int,int,int), class VectorBlock
  */
template<typename MatrixType> class Block
  : public MatrixBase<typename MatrixType::Scalar, Block<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Ref MatRef;
    friend class MatrixBase<Scalar, Block<MatrixType> >;
    
    Block(const MatRef& matrix,
          int startRow, int startCol,
          int blockRows, int blockCols)
      : m_matrix(matrix), m_startRow(startRow), m_startCol(startCol),
                          m_blockRows(blockRows), m_blockCols(blockCols)
    {
      assert(startRow >= 0 && blockRows >= 1 && startRow + blockRows <= matrix.rows()
          && startCol >= 0 && blockCols >= 1 && startCol + blockCols <= matrix.cols());
    }
    
    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Block)
    
  private:
    enum {
      RowsAtCompileTime = MatrixType::Traits::RowsAtCompileTime == 1 ? 1 : Dynamic,
      ColsAtCompileTime = MatrixType::Traits::ColsAtCompileTime == 1 ? 1 : Dynamic,
      MaxRowsAtCompileTime = RowsAtCompileTime == 1 ? 1 : MatrixType::Traits::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = ColsAtCompileTime == 1 ? 1 : MatrixType::Traits::MaxColsAtCompileTime
    };

    const Block& _ref() const { return *this; }
    int _rows() const { return m_blockRows.value(); }
    int _cols() const { return m_blockCols.value(); }
    
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
    IntAtRunTimeIfDynamic<RowsAtCompileTime> m_blockRows;
    IntAtRunTimeIfDynamic<ColsAtCompileTime> m_blockCols;
};

/** \returns a dynamic-size expression of a block in *this.
  *
  * \param startRow the first row in the block
  * \param startCol the first column in the block
  * \param blockRows the number of rows in the block
  * \param blockCols the number of columns in the block
  *
  * Example: \include MatrixBase_block_int_int_int_int.cpp
  * Output: \verbinclude MatrixBase_block_int_int_int_int.out
  *
  * \note Even though the returned expression has dynamic size, in the case
  * when it is applied to a fixed-size matrix, it inherits a fixed maximal size,
  * which means that evaluating it does not cause a dynamic memory allocation.
  *
  * \sa class Block, fixedBlock(int,int)
  */
template<typename Scalar, typename Derived>
Block<Derived> MatrixBase<Scalar, Derived>
  ::block(int startRow, int startCol, int blockRows, int blockCols)
{
  return Block<Derived>(ref(), startRow, startCol, blockRows, blockCols);
}

/** This is the const version of block(int,int,int,int). */
template<typename Scalar, typename Derived>
const Block<Derived> MatrixBase<Scalar, Derived>
  ::block(int startRow, int startCol, int blockRows, int blockCols) const
{
  return Block<Derived>(ref(), startRow, startCol, blockRows, blockCols);
}

/** \returns a dynamic-size expression of a block in *this.
  *
  * \only_for_vectors
  *
  * \param start the first coefficient in the block
  * \param size the number of coefficients in the block
  *
  * Example: \include MatrixBase_block_int_int.cpp
  * Output: \verbinclude MatrixBase_block_int_int.out
  *
  * \note Even though the returned expression has dynamic size, in the case
  * when it is applied to a fixed-size vector, it inherits a fixed maximal size,
  * which means that evaluating it does not cause a dynamic memory allocation.
  *
  * \sa class Block, fixedBlock(int)
  */
template<typename Scalar, typename Derived>
Block<Derived> MatrixBase<Scalar, Derived>
  ::block(int start, int size)
{
  assert(Traits::IsVectorAtCompileTime);
  return Block<Derived>(ref(), Traits::RowsAtCompileTime == 1 ? 0 : start,
                               Traits::ColsAtCompileTime == 1 ? 0 : start,
                               Traits::RowsAtCompileTime == 1 ? 1 : size,
                               Traits::ColsAtCompileTime == 1 ? 1 : size);
}

/** This is the const version of block(int,int).*/
template<typename Scalar, typename Derived>
const Block<Derived> MatrixBase<Scalar, Derived>
  ::block(int start, int size) const
{
  assert(Traits::IsVectorAtCompileTime);
  return Block<Derived>(ref(), Traits::RowsAtCompileTime == 1 ? 0 : start,
                               Traits::ColsAtCompileTime == 1 ? 0 : start,
                               Traits::RowsAtCompileTime == 1 ? 1 : size,
                               Traits::ColsAtCompileTime == 1 ? 1 : size);
}

/** \returns a dynamic-size expression of the first coefficients of *this.
  *
  * \only_for_vectors
  *
  * \param size the number of coefficients in the block
  *
  * Example: \include MatrixBase_start_int.cpp
  * Output: \verbinclude MatrixBase_start_int.out
  *
  * \note Even though the returned expression has dynamic size, in the case
  * when it is applied to a fixed-size vector, it inherits a fixed maximal size,
  * which means that evaluating it does not cause a dynamic memory allocation.
  *
  * \sa class Block, block(int,int)
  */
template<typename Scalar, typename Derived>
Block<Derived> MatrixBase<Scalar, Derived>
  ::start(int size)
{
  assert(Traits::IsVectorAtCompileTime);
  return Block<Derived>(ref(), 0, 0,
                        Traits::RowsAtCompileTime == 1 ? 1 : size,
                        Traits::ColsAtCompileTime == 1 ? 1 : size);
}

/** This is the const version of start(int).*/
template<typename Scalar, typename Derived>
const Block<Derived> MatrixBase<Scalar, Derived>
  ::start(int size) const
{
  assert(Traits::IsVectorAtCompileTime);
  return Block<Derived>(ref(), 0, 0,
                        Traits::RowsAtCompileTime == 1 ? 1 : size,
                        Traits::ColsAtCompileTime == 1 ? 1 : size);
}

/** \returns a dynamic-size expression of the last coefficients of *this.
  *
  * \only_for_vectors
  *
  * \param size the number of coefficients in the block
  *
  * Example: \include MatrixBase_end_int.cpp
  * Output: \verbinclude MatrixBase_end_int.out
  *
  * \note Even though the returned expression has dynamic size, in the case
  * when it is applied to a fixed-size vector, it inherits a fixed maximal size,
  * which means that evaluating it does not cause a dynamic memory allocation.
  *
  * \sa class Block, block(int,int)
  */
template<typename Scalar, typename Derived>
Block<Derived> MatrixBase<Scalar, Derived>
  ::end(int size)
{
  assert(Traits::IsVectorAtCompileTime);
  return Block<Derived>(ref(),
                        Traits::RowsAtCompileTime == 1 ? 0 : rows() - size,
                        Traits::ColsAtCompileTime == 1 ? 0 : cols() - size,
                        Traits::RowsAtCompileTime == 1 ? 1 : size,
                        Traits::ColsAtCompileTime == 1 ? 1 : size);
}

/** This is the const version of end(int).*/
template<typename Scalar, typename Derived>
const Block<Derived> MatrixBase<Scalar, Derived>
  ::end(int size) const
{
  assert(Traits::IsVectorAtCompileTime);
  return Block<Derived>(ref(),
                        Traits::RowsAtCompileTime == 1 ? 0 : rows() - size,
                        Traits::ColsAtCompileTime == 1 ? 0 : cols() - size,
                        Traits::RowsAtCompileTime == 1 ? 1 : size,
                        Traits::ColsAtCompileTime == 1 ? 1 : size);
}

/** \returns a dynamic-size expression of a corner of *this.
  *
  * \param type the type of corner. Can be \a Eigen::TopLeft, \a Eigen::TopRight,
  * \a Eigen::BottomLeft, \a Eigen::BottomRight.
  * \param cRows the number of rows in the corner
  * \param cCols the number of columns in the corner
  *
  * Example: \include MatrixBase_corner_enum_int_int.cpp
  * Output: \verbinclude MatrixBase_corner_enum_int_int.out
  *
  * \note Even though the returned expression has dynamic size, in the case
  * when it is applied to a fixed-size matrix, it inherits a fixed maximal size,
  * which means that evaluating it does not cause a dynamic memory allocation.
  *
  * \sa class Block, block(int,int,int,int)
  */
template<typename Scalar, typename Derived>
Block<Derived> MatrixBase<Scalar, Derived>
  ::corner(CornerType type, int cRows, int cCols)
{
  if(type == TopLeft) return Block<Derived>(ref(), 0, 0, cRows, cCols);
  else if(type == TopRight) return Block<Derived>(ref(), 0, cols() - cCols, cRows, cCols);
  else if(type == BottomLeft) return Block<Derived>(ref(), rows() - cRows, 0, cRows, cCols);
  else if(type == BottomRight)
    return Block<Derived>(ref(), rows() - cRows, cols() - cCols, cRows, cCols);
}

/** This is the const version of corner(CornerType, int, int).*/
template<typename Scalar, typename Derived>
const Block<Derived> MatrixBase<Scalar, Derived>
  ::corner(CornerType type, int cRows, int cCols) const
{
  if(type == TopLeft) return Block<Derived>(ref(), 0, 0, cRows, cCols);
  else if(type == TopRight) return Block<Derived>(ref(), 0, cols() - cCols, cRows, cCols);
  else if(type == BottomLeft) return Block<Derived>(ref(), rows() - cRows, 0, cRows, cCols);
  else if(type == BottomRight)
    return Block<Derived>(ref(), rows() - cRows, cols() - cCols, cRows, cCols);
}

#endif // EIGEN_BLOCK_H
