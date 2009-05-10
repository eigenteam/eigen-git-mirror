// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
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

#ifndef EIGEN_DIAGONAL_H
#define EIGEN_DIAGONAL_H

/** \class Diagonal
  *
  * \brief Expression of a diagonal/subdiagonal/superdiagonal in a matrix
  *
  * \param MatrixType the type of the object in which we are taking a sub/main/super diagonal
  * \param Index the index of the sub/super diagonal. The default is 0 and it means the main diagonal.
  *              A positive value means a superdiagonal, a negative value means a subdiagonal.
  *              You can also use Dynamic so the index can be set at runtime.
  *
  * The matrix is not required to be square.
  *
  * This class represents an expression of the main diagonal, or any sub/super diagonal
  * of a square matrix. It is the return type of MatrixBase::diagonal() and MatrixBase::diagonal(int) and most of the
  * time this is the only way it is used.
  *
  * \sa MatrixBase::diagonal(), MatrixBase::diagonal(int)
  */
template<typename MatrixType, int Index>
struct ei_traits<Diagonal<MatrixType,Index> >
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename ei_nested<MatrixType>::type MatrixTypeNested;
  typedef typename ei_unref<MatrixTypeNested>::type _MatrixTypeNested;
  enum {
    AbsIndex = Index<0 ? -Index : Index, // only used if Index != Dynamic
    RowsAtCompileTime = (int(Index) == Dynamic || int(MatrixType::SizeAtCompileTime) == Dynamic) ? Dynamic
                      : (EIGEN_ENUM_MIN(MatrixType::RowsAtCompileTime,
                                        MatrixType::ColsAtCompileTime) - AbsIndex),
    ColsAtCompileTime = 1,
    MaxRowsAtCompileTime = int(MatrixType::MaxSizeAtCompileTime) == Dynamic ? Dynamic
                         : (EIGEN_ENUM_MIN(MatrixType::MaxRowsAtCompileTime,
                                          MatrixType::MaxColsAtCompileTime) - AbsIndex),
    MaxColsAtCompileTime = 1,
    Flags = (unsigned int)_MatrixTypeNested::Flags & (HereditaryBits | LinearAccessBit),
    CoeffReadCost = _MatrixTypeNested::CoeffReadCost
  };
};

template<typename MatrixType, int Index> class Diagonal
   : public MatrixBase<Diagonal<MatrixType, Index> >
{
    // some compilers may fail to optimize std::max etc in case of compile-time constants...
    EIGEN_STRONG_INLINE int absIndex() const { return m_index.value()>0 ? m_index.value() : -m_index.value(); }
    EIGEN_STRONG_INLINE int rowOffset() const { return m_index.value()>0 ? 0 : -m_index.value(); }
    EIGEN_STRONG_INLINE int colOffset() const { return m_index.value()>0 ? m_index.value() : 0; }
    
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(Diagonal)

    inline Diagonal(const MatrixType& matrix, int index = Index) : m_matrix(matrix), m_index(index) {}

    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Diagonal)

    inline int rows() const{ return m_matrix.diagonalSize() - absIndex(); }
    inline int cols() const { return 1; }

    inline Scalar& coeffRef(int row, int)
    {
      return m_matrix.const_cast_derived().coeffRef(row+rowOffset(), row+colOffset());
    }

    inline const Scalar coeff(int row, int) const
    {
      return m_matrix.coeff(row+rowOffset(), row+colOffset());
    }

    inline Scalar& coeffRef(int index)
    {
      return m_matrix.const_cast_derived().coeffRef(index+rowOffset(), index+colOffset());
    }

    inline const Scalar coeff(int index) const
    {
      return m_matrix.coeff(index+rowOffset(), index+colOffset());
    }

  protected:
    const typename MatrixType::Nested m_matrix;
    const ei_int_if_dynamic<Index> m_index;
};

/** \returns an expression of the main diagonal of the matrix \c *this
  *
  * \c *this is not required to be square.
  *
  * Example: \include MatrixBase_diagonal.cpp
  * Output: \verbinclude MatrixBase_diagonal.out
  *
  * \sa class Diagonal */
template<typename Derived>
inline Diagonal<Derived, 0>
MatrixBase<Derived>::diagonal()
{
  return Diagonal<Derived, 0>(derived());
}

/** This is the const version of diagonal(). */
template<typename Derived>
inline const Diagonal<Derived, 0>
MatrixBase<Derived>::diagonal() const
{
  return Diagonal<Derived, 0>(derived());
}

/** \returns an expression of the \a Index-th sub or super diagonal of the matrix \c *this
  *
  * \c *this is not required to be square.
  *
  * The template parameter \a Index represent a super diagonal if \a Index > 0
  * and a sub diagonal otherwise. \a Index == 0 is equivalent to the main diagonal.
  *
  * Example: \include MatrixBase_diagonal_int.cpp
  * Output: \verbinclude MatrixBase_diagonal_int.out
  *
  * \sa MatrixBase::diagonal(), class Diagonal */
template<typename Derived>
inline Diagonal<Derived, Dynamic>
MatrixBase<Derived>::diagonal(int index)
{
  return Diagonal<Derived, Dynamic>(derived(), index);
}

/** This is the const version of diagonal(int). */
template<typename Derived>
inline const Diagonal<Derived, Dynamic>
MatrixBase<Derived>::diagonal(int index) const
{
  return Diagonal<Derived, Dynamic>(derived(), index);
}

/** \returns an expression of the \a Index-th sub or super diagonal of the matrix \c *this
  *
  * \c *this is not required to be square.
  *
  * The template parameter \a Index represent a super diagonal if \a Index > 0
  * and a sub diagonal otherwise. \a Index == 0 is equivalent to the main diagonal.
  *
  * Example: \include MatrixBase_diagonal_template_int.cpp
  * Output: \verbinclude MatrixBase_diagonal_template_int.out
  *
  * \sa MatrixBase::diagonal(), class Diagonal */
template<typename Derived>
template<int Index>
inline Diagonal<Derived,Index>
MatrixBase<Derived>::diagonal()
{
  return Diagonal<Derived,Index>(derived());
}

/** This is the const version of diagonal<int>(). */
template<typename Derived>
template<int Index>
inline const Diagonal<Derived,Index>
MatrixBase<Derived>::diagonal() const
{
  return Diagonal<Derived,Index>(derived());
}

#endif // EIGEN_DIAGONAL_H
