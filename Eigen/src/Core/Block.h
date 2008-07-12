// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_BLOCK_H
#define EIGEN_BLOCK_H

/** \class Block
  *
  * \brief Expression of a fixed-size or dynamic-size block
  *
  * \param MatrixType the type of the object in which we are taking a block
  * \param BlockRows the number of rows of the block we are taking at compile time (optional)
  * \param BlockCols the number of columns of the block we are taking at compile time (optional)
  *
  * This class represents an expression of either a fixed-size or dynamic-size block. It is the return
  * type of MatrixBase::block(int,int,int,int) and MatrixBase::block<int,int>(int,int) and
  * most of the time this is the only way it is used.
  *
  * However, if you want to directly maniputate block expressions,
  * for instance if you want to write a function returning such an expression, you
  * will need to use this class.
  *
  * Here is an example illustrating the dynamic case:
  * \include class_Block.cpp
  * Output: \verbinclude class_Block.out
  *
  * \note Even though this expression has dynamic size, in the case where \a MatrixType
  * has fixed size, this expression inherits a fixed maximal size which means that evaluating
  * it does not cause a dynamic memory allocation.
  *
  * Here is an example illustrating the fixed-size case:
  * \include class_FixedBlock.cpp
  * Output: \verbinclude class_FixedBlock.out
  *
  * \sa MatrixBase::block(int,int,int,int), MatrixBase::block(int,int), class VectorBlock
  */
template<typename MatrixType, int BlockRows, int BlockCols, int DirectAccesStatus>
struct ei_traits<Block<MatrixType, BlockRows, BlockCols, DirectAccesStatus> >
{
  typedef typename MatrixType::Scalar Scalar;
  enum{
    RowsAtCompileTime = MatrixType::RowsAtCompileTime == 1 ? 1 : BlockRows,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime == 1 ? 1 : BlockCols,
    MaxRowsAtCompileTime = RowsAtCompileTime == 1 ? 1
      : (BlockRows==Dynamic ? MatrixType::MaxRowsAtCompileTime : BlockRows),
    MaxColsAtCompileTime = ColsAtCompileTime == 1 ? 1
      : (BlockCols==Dynamic ? MatrixType::MaxColsAtCompileTime : BlockCols),
    MaskLargeBit = ((RowsAtCompileTime != Dynamic && MatrixType::RowsAtCompileTime == Dynamic)
                  || (ColsAtCompileTime != Dynamic && MatrixType::ColsAtCompileTime == Dynamic))
                   ? ~LargeBit
                   : ~(unsigned int)0,
    RowMajor = int(MatrixType::Flags)&RowMajorBit,
    InnerSize = RowMajor ? ColsAtCompileTime : RowsAtCompileTime,
    InnerMaxSize = RowMajor ? MaxColsAtCompileTime : MaxRowsAtCompileTime,
    MaskPacketAccessBit = (InnerMaxSize == Dynamic || (InnerSize % ei_packet_traits<Scalar>::size) == 0)
                        ? PacketAccessBit : 0,
    FlagsLinearAccessBit = (RowsAtCompileTime == 1 || ColsAtCompileTime == 1) ? LinearAccessBit : 0,
    Flags = (MatrixType::Flags & (HereditaryBits | MaskPacketAccessBit | DirectAccessBit) & MaskLargeBit)
          | FlagsLinearAccessBit,
    CoeffReadCost = MatrixType::CoeffReadCost
  };
};

template<typename MatrixType, int BlockRows, int BlockCols, int DirectAccesStatus> class Block
  : public MatrixBase<Block<MatrixType, BlockRows, BlockCols, DirectAccesStatus> >
{
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(Block)

    /** Column or Row constructor
      */
    inline Block(const MatrixType& matrix, int i)
      : m_matrix(matrix),
        // It is a row if and only if BlockRows==1 and BlockCols==MatrixType::ColsAtCompileTime,
        // and it is a column if and only if BlockRows==MatrixType::RowsAtCompileTime and BlockCols==1,
        // all other cases are invalid.
        // The case a 1x1 matrix seems ambiguous, but the result is the same anyway.
        m_startRow( (BlockRows==1) && (BlockCols==MatrixType::ColsAtCompileTime) ? i : 0),
        m_startCol( (BlockRows==MatrixType::RowsAtCompileTime) && (BlockCols==1) ? i : 0),
        m_blockRows(matrix.rows()), // if it is a row, then m_blockRows has a fixed-size of 1, so no pb to try to overwrite it
        m_blockCols(matrix.cols())  // same for m_blockCols
    {
      ei_assert( (i>=0) && (
          ((BlockRows==1) && (BlockCols==MatrixType::ColsAtCompileTime) && i<matrix.rows())
        ||((BlockRows==MatrixType::RowsAtCompileTime) && (BlockCols==1) && i<matrix.cols())));
    }

    /** Fixed-size constructor
      */
    inline Block(const MatrixType& matrix, int startRow, int startCol)
      : m_matrix(matrix), m_startRow(startRow), m_startCol(startCol)
    {
      ei_assert(RowsAtCompileTime!=Dynamic && RowsAtCompileTime!=Dynamic);
      ei_assert(startRow >= 0 && BlockRows >= 1 && startRow + BlockRows <= matrix.rows()
          && startCol >= 0 && BlockCols >= 1 && startCol + BlockCols <= matrix.cols());
    }

    /** Dynamic-size constructor
      */
    inline Block(const MatrixType& matrix,
          int startRow, int startCol,
          int blockRows, int blockCols)
      : m_matrix(matrix), m_startRow(startRow), m_startCol(startCol),
                          m_blockRows(blockRows), m_blockCols(blockCols)
    {
      ei_assert((RowsAtCompileTime==Dynamic || RowsAtCompileTime==blockRows)
          && (ColsAtCompileTime==Dynamic || ColsAtCompileTime==blockCols));
      ei_assert(startRow >= 0 && blockRows >= 1 && startRow + blockRows <= matrix.rows()
          && startCol >= 0 && blockCols >= 1 && startCol + blockCols <= matrix.cols());
    }

    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Block)

    inline int rows() const { return m_blockRows.value(); }
    inline int cols() const { return m_blockCols.value(); }

    inline int stride(void) const { return m_matrix.stride(); }

    inline Scalar& coeffRef(int row, int col)
    {
      return m_matrix.const_cast_derived()
               .coeffRef(row + m_startRow.value(), col + m_startCol.value());
    }

    inline const Scalar coeff(int row, int col) const
    {
      return m_matrix.coeff(row + m_startRow.value(), col + m_startCol.value());
    }

    inline Scalar& coeffRef(int index)
    {
      return m_matrix.const_cast_derived()
             .coeffRef(m_startRow.value() + (RowsAtCompileTime == 1 ? 0 : index),
                       m_startCol.value() + (RowsAtCompileTime == 1 ? index : 0));
    }

    inline const Scalar coeff(int index) const
    {
      return m_matrix
             .coeff(m_startRow.value() + (RowsAtCompileTime == 1 ? 0 : index),
                    m_startCol.value() + (RowsAtCompileTime == 1 ? index : 0));
    }

    template<int LoadMode>
    inline PacketScalar packet(int row, int col) const
    {
      return m_matrix.template packet<Unaligned>
              (row + m_startRow.value(), col + m_startCol.value());
    }

    template<int LoadMode>
    inline void writePacket(int row, int col, const PacketScalar& x)
    {
      m_matrix.const_cast_derived().template writePacket<Unaligned>
              (row + m_startRow.value(), col + m_startCol.value(), x);
    }

    template<int LoadMode>
    inline PacketScalar packet(int index) const
    {
      return m_matrix.template packet<Unaligned>
              (m_startRow.value() + (RowsAtCompileTime == 1 ? 0 : index),
               m_startCol.value() + (RowsAtCompileTime == 1 ? index : 0));
    }

    template<int LoadMode>
    inline void writePacket(int index, const PacketScalar& x)
    {
      m_matrix.const_cast_derived().template writePacket<Unaligned>
         (m_startRow.value() + (RowsAtCompileTime == 1 ? 0 : index),
          m_startCol.value() + (RowsAtCompileTime == 1 ? index : 0), x);
    }

  protected:

    const typename MatrixType::Nested m_matrix;
    const ei_int_if_dynamic<MatrixType::RowsAtCompileTime == 1 ? 0 : Dynamic> m_startRow;
    const ei_int_if_dynamic<MatrixType::ColsAtCompileTime == 1 ? 0 : Dynamic> m_startCol;
    const ei_int_if_dynamic<RowsAtCompileTime> m_blockRows;
    const ei_int_if_dynamic<ColsAtCompileTime> m_blockCols;
};

/** \internal */
template<typename MatrixType, int BlockRows, int BlockCols> class Block<MatrixType,BlockRows,BlockCols,HasDirectAccess>
  : public MatrixBase<Block<MatrixType, BlockRows, BlockCols,HasDirectAccess> >
{
    enum {
      IsRowMajor = int(ei_traits<MatrixType>::Flags)&RowMajorBit ? 1 : 0
    };

  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(Block)

    /** Column or Row constructor
      */
    inline Block(const MatrixType& matrix, int i)
      : m_matrix(matrix),
        m_data_ptr(&matrix.const_cast_derived().coeffRef(
          (BlockRows==1) && (BlockCols==MatrixType::ColsAtCompileTime) ? i : 0,
          (BlockRows==MatrixType::RowsAtCompileTime) && (BlockCols==1) ? i : 0)),
        m_blockRows(matrix.rows()),
        m_blockCols(matrix.cols())
    {
      ei_assert( (i>=0) && (
          ((BlockRows==1) && (BlockCols==MatrixType::ColsAtCompileTime) && i<matrix.rows())
        ||((BlockRows==MatrixType::RowsAtCompileTime) && (BlockCols==1) && i<matrix.cols())));
    }

    /** Fixed-size constructor
      */
    inline Block(const MatrixType& matrix, int startRow, int startCol)
      : m_matrix(matrix), m_data_ptr(&matrix.const_cast_derived().coeffRef(startRow,startCol))
    {
      ei_assert(RowsAtCompileTime!=Dynamic && RowsAtCompileTime!=Dynamic);
      ei_assert(startRow >= 0 && BlockRows >= 1 && startRow + BlockRows <= matrix.rows()
          && startCol >= 0 && BlockCols >= 1 && startCol + BlockCols <= matrix.cols());
    }

    /** Dynamic-size constructor
      */
    inline Block(const MatrixType& matrix,
          int startRow, int startCol,
          int blockRows, int blockCols)
      : m_matrix(matrix), m_data_ptr(&matrix.const_cast_derived().coeffRef(startRow,startCol)),
        m_blockRows(blockRows), m_blockCols(blockCols)
    {
      ei_assert((RowsAtCompileTime==Dynamic || RowsAtCompileTime==blockRows)
          && (ColsAtCompileTime==Dynamic || ColsAtCompileTime==blockCols));
      ei_assert(startRow >= 0 && blockRows >= 1 && startRow + blockRows <= matrix.rows()
          && startCol >= 0 && blockCols >= 1 && startCol + blockCols <= matrix.cols());
    }

    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Block)

    inline int rows() const { return m_blockRows.value(); }
    inline int cols() const { return m_blockCols.value(); }

    inline int stride(void) const { return m_matrix.stride(); }

    inline Scalar& coeffRef(int row, int col)
    {
      if (IsRowMajor)
        return m_data_ptr[col + row * stride()];
      else
        return m_data_ptr[row + col * stride()];
    }

    inline const Scalar coeff(int row, int col) const
    {
//       std::cerr << "coeff(int row, int col)\n";
      if (IsRowMajor)
        return m_data_ptr[col + row * stride()];
      else
        return m_data_ptr[row + col * stride()];
    }

    inline Scalar& coeffRef(int index)
    {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(Block);
      return m_data_ptr[index];
    }

    inline const Scalar coeff(int index) const
    {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(Block);
      if ( (RowsAtCompileTime == 1) == IsRowMajor )
        return m_data_ptr[index];
      else
        return m_data_ptr[index*stride()];
    }

    template<int LoadMode>
    inline PacketScalar packet(int row, int col) const
    {
      if (IsRowMajor)
        return ei_ploadu(&m_data_ptr[col + row * stride()]);
      else
        return ei_ploadu(&m_data_ptr[row + col * stride()]);
    }

    template<int LoadMode>
    inline void writePacket(int row, int col, const PacketScalar& x)
    {
      if (IsRowMajor)
        ei_pstoreu(&m_data_ptr[col + row * stride()], x);
      else
        ei_pstoreu(&m_data_ptr[row + col * stride()], x);
    }

    template<int LoadMode>
    inline PacketScalar packet(int index) const
    {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(Block);
      return ei_ploadu(&m_data_ptr[index]);
    }

    template<int LoadMode>
    inline void writePacket(int index, const PacketScalar& x)
    {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(Block);
      ei_pstoreu(&m_data_ptr[index], x);
    }

  protected:

    const typename MatrixType::Nested m_matrix;
    Scalar* m_data_ptr;
    const ei_int_if_dynamic<RowsAtCompileTime> m_blockRows;
    const ei_int_if_dynamic<ColsAtCompileTime> m_blockCols;
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
  * \sa class Block, block(int,int)
  */
template<typename Derived>
inline Block<Derived> MatrixBase<Derived>
  ::block(int startRow, int startCol, int blockRows, int blockCols)
{
  return Block<Derived>(derived(), startRow, startCol, blockRows, blockCols);
}

/** This is the const version of block(int,int,int,int). */
template<typename Derived>
inline const Block<Derived> MatrixBase<Derived>
  ::block(int startRow, int startCol, int blockRows, int blockCols) const
{
  return Block<Derived>(derived(), startRow, startCol, blockRows, blockCols);
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
  * \sa class Block, block(int)
  */
template<typename Derived>
inline Block<Derived> MatrixBase<Derived>
  ::block(int start, int size)
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
  return Block<Derived>(derived(), RowsAtCompileTime == 1 ? 0 : start,
                                   ColsAtCompileTime == 1 ? 0 : start,
                                   RowsAtCompileTime == 1 ? 1 : size,
                                   ColsAtCompileTime == 1 ? 1 : size);
}

/** This is the const version of block(int,int).*/
template<typename Derived>
inline const Block<Derived> MatrixBase<Derived>
  ::block(int start, int size) const
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
  return Block<Derived>(derived(), RowsAtCompileTime == 1 ? 0 : start,
                                   ColsAtCompileTime == 1 ? 0 : start,
                                   RowsAtCompileTime == 1 ? 1 : size,
                                   ColsAtCompileTime == 1 ? 1 : size);
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
template<typename Derived>
inline typename MatrixBase<Derived>::template SubVectorReturnType<Dynamic>::Type
MatrixBase<Derived>::start(int size)
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
  return Block<Derived,
               RowsAtCompileTime == 1 ? 1 : Dynamic,
               ColsAtCompileTime == 1 ? 1 : Dynamic>
              (derived(), 0, 0,
               RowsAtCompileTime == 1 ? 1 : size,
               ColsAtCompileTime == 1 ? 1 : size);
}

/** This is the const version of start(int).*/
template<typename Derived>
inline const typename MatrixBase<Derived>::template SubVectorReturnType<Dynamic>::Type
MatrixBase<Derived>::start(int size) const
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
  return Block<Derived,
               RowsAtCompileTime == 1 ? 1 : Dynamic,
               ColsAtCompileTime == 1 ? 1 : Dynamic>
              (derived(), 0, 0,
               RowsAtCompileTime == 1 ? 1 : size,
               ColsAtCompileTime == 1 ? 1 : size);
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
template<typename Derived>
inline typename MatrixBase<Derived>::template SubVectorReturnType<Dynamic>::Type
MatrixBase<Derived>::end(int size)
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
  return Block<Derived,
               RowsAtCompileTime == 1 ? 1 : Dynamic,
               ColsAtCompileTime == 1 ? 1 : Dynamic>
              (derived(),
               RowsAtCompileTime == 1 ? 0 : rows() - size,
               ColsAtCompileTime == 1 ? 0 : cols() - size,
               RowsAtCompileTime == 1 ? 1 : size,
               ColsAtCompileTime == 1 ? 1 : size);
}

/** This is the const version of end(int).*/
template<typename Derived>
inline const typename MatrixBase<Derived>::template SubVectorReturnType<Dynamic>::Type
MatrixBase<Derived>::end(int size) const
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
  return Block<Derived,
               RowsAtCompileTime == 1 ? 1 : Dynamic,
               ColsAtCompileTime == 1 ? 1 : Dynamic>
              (derived(),
               RowsAtCompileTime == 1 ? 0 : rows() - size,
               ColsAtCompileTime == 1 ? 0 : cols() - size,
               RowsAtCompileTime == 1 ? 1 : size,
               ColsAtCompileTime == 1 ? 1 : size);
}

/** \returns a fixed-size expression of the first coefficients of *this.
  *
  * \only_for_vectors
  *
  * The template parameter \a Size is the number of coefficients in the block
  *
  * Example: \include MatrixBase_template_int_start.cpp
  * Output: \verbinclude MatrixBase_template_int_start.out
  *
  * \sa class Block
  */
template<typename Derived>
template<int Size>
inline typename MatrixBase<Derived>::template SubVectorReturnType<Size>::Type
MatrixBase<Derived>::start()
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
  return Block<Derived, (RowsAtCompileTime == 1 ? 1 : Size),
                        (ColsAtCompileTime == 1 ? 1 : Size)>(derived(), 0, 0);
}

/** This is the const version of start<int>().*/
template<typename Derived>
template<int Size>
inline const typename MatrixBase<Derived>::template SubVectorReturnType<Size>::Type
MatrixBase<Derived>::start() const
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
  return Block<Derived, (RowsAtCompileTime == 1 ? 1 : Size),
                        (ColsAtCompileTime == 1 ? 1 : Size)>(derived(), 0, 0);
}

/** \returns a fixed-size expression of the last coefficients of *this.
  *
  * \only_for_vectors
  *
  * The template parameter \a Size is the number of coefficients in the block
  *
  * Example: \include MatrixBase_template_int_end.cpp
  * Output: \verbinclude MatrixBase_template_int_end.out
  *
  * \sa class Block
  */
template<typename Derived>
template<int Size>
inline typename MatrixBase<Derived>::template SubVectorReturnType<Size>::Type
MatrixBase<Derived>::end()
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
  return Block<Derived, RowsAtCompileTime == 1 ? 1 : Size,
                        ColsAtCompileTime == 1 ? 1 : Size>
           (derived(),
            RowsAtCompileTime == 1 ? 0 : rows() - Size,
            ColsAtCompileTime == 1 ? 0 : cols() - Size);
}

/** This is the const version of end<int>.*/
template<typename Derived>
template<int Size>
inline const typename MatrixBase<Derived>::template SubVectorReturnType<Size>::Type
MatrixBase<Derived>::end() const
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
  return Block<Derived, RowsAtCompileTime == 1 ? 1 : Size,
                        ColsAtCompileTime == 1 ? 1 : Size>
           (derived(),
            RowsAtCompileTime == 1 ? 0 : rows() - Size,
            ColsAtCompileTime == 1 ? 0 : cols() - Size);
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
template<typename Derived>
inline Block<Derived> MatrixBase<Derived>
  ::corner(CornerType type, int cRows, int cCols)
{
  switch(type)
  {
    default:
      ei_assert(false && "Bad corner type.");
    case TopLeft:
      return Block<Derived>(derived(), 0, 0, cRows, cCols);
    case TopRight:
      return Block<Derived>(derived(), 0, cols() - cCols, cRows, cCols);
    case BottomLeft:
      return Block<Derived>(derived(), rows() - cRows, 0, cRows, cCols);
    case BottomRight:
      return Block<Derived>(derived(), rows() - cRows, cols() - cCols, cRows, cCols);
  }
}

/** This is the const version of corner(CornerType, int, int).*/
template<typename Derived>
inline const Block<Derived> MatrixBase<Derived>
  ::corner(CornerType type, int cRows, int cCols) const
{
  switch(type)
  {
    default:
      ei_assert(false && "Bad corner type.");
    case TopLeft:
      return Block<Derived>(derived(), 0, 0, cRows, cCols);
    case TopRight:
      return Block<Derived>(derived(), 0, cols() - cCols, cRows, cCols);
    case BottomLeft:
      return Block<Derived>(derived(), rows() - cRows, 0, cRows, cCols);
    case BottomRight:
      return Block<Derived>(derived(), rows() - cRows, cols() - cCols, cRows, cCols);
  }
}

/** \returns a fixed-size expression of a corner of *this.
  *
  * \param type the type of corner. Can be \a Eigen::TopLeft, \a Eigen::TopRight,
  * \a Eigen::BottomLeft, \a Eigen::BottomRight.
  *
  * The template parameters CRows and CCols arethe number of rows and columns in the corner.
  *
  * Example: \include MatrixBase_template_int_int_corner_enum.cpp
  * Output: \verbinclude MatrixBase_template_int_int_corner_enum.out
  *
  * \sa class Block, block(int,int,int,int)
  */
template<typename Derived>
template<int CRows, int CCols>
inline Block<Derived, CRows, CCols> MatrixBase<Derived>
  ::corner(CornerType type)
{
  switch(type)
  {
    default:
      ei_assert(false && "Bad corner type.");
    case TopLeft:
      return Block<Derived, CRows, CCols>(derived(), 0, 0);
    case TopRight:
      return Block<Derived, CRows, CCols>(derived(), 0, cols() - CCols);
    case BottomLeft:
      return Block<Derived, CRows, CCols>(derived(), rows() - CRows, 0);
    case BottomRight:
      return Block<Derived, CRows, CCols>(derived(), rows() - CRows, cols() - CCols);
  }
}

/** This is the const version of corner<int, int>(CornerType).*/
template<typename Derived>
template<int CRows, int CCols>
inline const Block<Derived, CRows, CCols> MatrixBase<Derived>
  ::corner(CornerType type) const
{
  switch(type)
  {
    default:
      ei_assert(false && "Bad corner type.");
    case TopLeft:
      return Block<Derived, CRows, CCols>(derived(), 0, 0);
    case TopRight:
      return Block<Derived, CRows, CCols>(derived(), 0, cols() - CCols);
    case BottomLeft:
      return Block<Derived, CRows, CCols>(derived(), rows() - CRows, 0);
    case BottomRight:
      return Block<Derived, CRows, CCols>(derived(), rows() - CRows, cols() - CCols);
  }
}

/** \returns a fixed-size expression of a block in *this.
  *
  * The template parameters \a BlockRows and \a BlockCols are the number of
  * rows and columns in the block.
  *
  * \param startRow the first row in the block
  * \param startCol the first column in the block
  *
  * Example: \include MatrixBase_block_int_int.cpp
  * Output: \verbinclude MatrixBase_block_int_int.out
  *
  * \note since block is a templated member, the keyword template has to be used
  * if the matrix type is also a template parameter: \code m.template block<3,3>(1,1); \endcode
  *
  * \sa class Block, block(int,int,int,int)
  */
template<typename Derived>
template<int BlockRows, int BlockCols>
inline Block<Derived, BlockRows, BlockCols> MatrixBase<Derived>
  ::block(int startRow, int startCol)
{
  return Block<Derived, BlockRows, BlockCols>(derived(), startRow, startCol);
}

/** This is the const version of block<>(int, int). */
template<typename Derived>
template<int BlockRows, int BlockCols>
inline const Block<Derived, BlockRows, BlockCols> MatrixBase<Derived>
  ::block(int startRow, int startCol) const
{
  return Block<Derived, BlockRows, BlockCols>(derived(), startRow, startCol);
}

/** \returns an expression of the \a i-th column of *this. Note that the numbering starts at 0.
  *
  * Example: \include MatrixBase_col.cpp
  * Output: \verbinclude MatrixBase_col.out
  *
  * \sa row(), class Block */
template<typename Derived>
inline Block<Derived, ei_traits<Derived>::RowsAtCompileTime, 1>
MatrixBase<Derived>::col(int i)
{
  return Block<Derived, ei_traits<Derived>::RowsAtCompileTime, 1>(derived(), i);
}

/** This is the const version of col(). */
template<typename Derived>
inline const Block<Derived, ei_traits<Derived>::RowsAtCompileTime, 1>
MatrixBase<Derived>::col(int i) const
{
  return Block<Derived, ei_traits<Derived>::RowsAtCompileTime, 1>(derived(), i);
}

/** \returns an expression of the \a i-th row of *this. Note that the numbering starts at 0.
  *
  * Example: \include MatrixBase_row.cpp
  * Output: \verbinclude MatrixBase_row.out
  *
  * \sa col(), class Block */
template<typename Derived>
inline Block<Derived, 1, ei_traits<Derived>::ColsAtCompileTime>
MatrixBase<Derived>::row(int i)
{
  return Block<Derived, 1, ei_traits<Derived>::ColsAtCompileTime>(derived(), i);
}

/** This is the const version of row(). */
template<typename Derived>
inline const Block<Derived, 1, ei_traits<Derived>::ColsAtCompileTime>
MatrixBase<Derived>::row(int i) const
{
  return Block<Derived, 1, ei_traits<Derived>::ColsAtCompileTime>(derived(), i);
}

#endif // EIGEN_BLOCK_H
