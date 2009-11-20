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

#ifndef EIGEN_VECTORBLOCK_H
#define EIGEN_VECTORBLOCK_H

/** \class VectorBlock
  *
  * \brief Expression of a fixed-size or dynamic-size sub-vector
  *
  * \param VectorType the type of the object in which we are taking a sub-vector
  * \param Size size of the sub-vector we are taking at compile time (optional)
  *
  * This class represents an expression of either a fixed-size or dynamic-size sub-vector.
  * It is the return type of MatrixBase::segment(int,int) and MatrixBase::segment<int>(int) and
  * most of the time this is the only way it is used.
  *
  * However, if you want to directly maniputate sub-vector expressions,
  * for instance if you want to write a function returning such an expression, you
  * will need to use this class.
  *
  * Here is an example illustrating the dynamic case:
  * \include class_VectorBlock.cpp
  * Output: \verbinclude class_VectorBlock.out
  *
  * \note Even though this expression has dynamic size, in the case where \a VectorType
  * has fixed size, this expression inherits a fixed maximal size which means that evaluating
  * it does not cause a dynamic memory allocation.
  *
  * Here is an example illustrating the fixed-size case:
  * \include class_FixedVectorBlock.cpp
  * Output: \verbinclude class_FixedVectorBlock.out
  *
  * \sa class Block, MatrixBase::segment(int,int,int,int), MatrixBase::segment(int,int)
  */
template<typename VectorType, int Size>
struct ei_traits<VectorBlock<VectorType, Size> >
  : public ei_traits<Block<VectorType,
                 ei_traits<VectorType>::RowsAtCompileTime==1 ? 1 : Size,
                 ei_traits<VectorType>::ColsAtCompileTime==1 ? 1 : Size> >
{
};

template<typename VectorType, int Size> class VectorBlock
  : public Block<VectorType,
                 ei_traits<VectorType>::RowsAtCompileTime==1 ? 1 : Size,
                 ei_traits<VectorType>::ColsAtCompileTime==1 ? 1 : Size>
{
    typedef Block<VectorType,
                  ei_traits<VectorType>::RowsAtCompileTime==1 ? 1 : Size,
                  ei_traits<VectorType>::ColsAtCompileTime==1 ? 1 : Size> _Base;
    enum {
      IsColVector = ei_traits<VectorType>::ColsAtCompileTime==1
    };
  public:
    _EIGEN_GENERIC_PUBLIC_INTERFACE(VectorBlock, _Base)

    using Base::operator=;
    using Base::operator+=;
    using Base::operator-=;
    using Base::operator*=;
    using Base::operator/=;

    /** Dynamic-size constructor
      */
    inline VectorBlock(const VectorType& vector, int start, int size)
      : Base(vector,
             IsColVector ? start : 0, IsColVector ? 0 : start,
             IsColVector ? size  : 1, IsColVector ? 1 : size)
    {

      EIGEN_STATIC_ASSERT_VECTOR_ONLY(VectorBlock);
    }

    /** Fixed-size constructor
      */
    inline VectorBlock(const VectorType& vector, int start)
      : Base(vector, IsColVector ? start : 0, IsColVector ? 0 : start)
    {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(VectorBlock);
    }
};


/** \returns a dynamic-size expression of a segment (i.e. a vector block) in *this.
  *
  * \only_for_vectors
  *
  * \param start the first coefficient in the segment
  * \param size the number of coefficients in the segment
  *
  * Example: \include MatrixBase_segment_int_int.cpp
  * Output: \verbinclude MatrixBase_segment_int_int.out
  *
  * \note Even though the returned expression has dynamic size, in the case
  * when it is applied to a fixed-size vector, it inherits a fixed maximal size,
  * which means that evaluating it does not cause a dynamic memory allocation.
  *
  * \sa class Block, segment(int)
  */
template<typename Derived>
inline VectorBlock<Derived> MatrixBase<Derived>
  ::segment(int start, int size)
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  return VectorBlock<Derived>(derived(), start, size);
}

/** This is the const version of segment(int,int).*/
template<typename Derived>
inline const VectorBlock<Derived>
MatrixBase<Derived>::segment(int start, int size) const
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  return VectorBlock<Derived>(derived(), start, size);
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
inline VectorBlock<Derived>
MatrixBase<Derived>::start(int size)
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  return VectorBlock<Derived>(derived(), 0, size);
}

/** This is the const version of start(int).*/
template<typename Derived>
inline const VectorBlock<Derived>
MatrixBase<Derived>::start(int size) const
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  return VectorBlock<Derived>(derived(), 0, size);
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
inline VectorBlock<Derived>
MatrixBase<Derived>::end(int size)
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  return VectorBlock<Derived>(derived(), this->size() - size, size);
}

/** This is the const version of end(int).*/
template<typename Derived>
inline const VectorBlock<Derived>
MatrixBase<Derived>::end(int size) const
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  return VectorBlock<Derived>(derived(), this->size() - size, size);
}

/** \returns a fixed-size expression of a segment (i.e. a vector block) in \c *this
  *
  * \only_for_vectors
  *
  * The template parameter \a Size is the number of coefficients in the block
  *
  * \param start the index of the first element of the sub-vector
  *
  * Example: \include MatrixBase_template_int_segment.cpp
  * Output: \verbinclude MatrixBase_template_int_segment.out
  *
  * \sa class Block
  */
template<typename Derived>
template<int Size>
inline VectorBlock<Derived,Size>
MatrixBase<Derived>::segment(int start)
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  return VectorBlock<Derived,Size>(derived(), start);
}

/** This is the const version of segment<int>(int).*/
template<typename Derived>
template<int Size>
inline const VectorBlock<Derived,Size>
MatrixBase<Derived>::segment(int start) const
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  return VectorBlock<Derived,Size>(derived(), start);
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
inline VectorBlock<Derived,Size>
MatrixBase<Derived>::start()
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  return VectorBlock<Derived,Size>(derived(), 0);
}

/** This is the const version of start<int>().*/
template<typename Derived>
template<int Size>
inline const VectorBlock<Derived,Size>
MatrixBase<Derived>::start() const
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  return VectorBlock<Derived,Size>(derived(), 0);
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
inline VectorBlock<Derived,Size>
MatrixBase<Derived>::end()
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  return VectorBlock<Derived, Size>(derived(), size() - Size);
}

/** This is the const version of end<int>.*/
template<typename Derived>
template<int Size>
inline const VectorBlock<Derived,Size>
MatrixBase<Derived>::end() const
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  return VectorBlock<Derived, Size>(derived(), size() - Size);
}


#endif // EIGEN_VECTORBLOCK_H
