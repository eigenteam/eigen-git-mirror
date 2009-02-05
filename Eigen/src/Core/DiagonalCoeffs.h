// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
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

#ifndef EIGEN_DIAGONALCOEFFS_H
#define EIGEN_DIAGONALCOEFFS_H

/** \class DiagonalCoeffs
  *
  * \brief Expression of a sub/main/super diagonal of a matrix
  *
  * \param MatrixType the type of the object in which we are taking s sub/main/super diagonal
  * \param DiagId the index of the sub/super diagonal. The default is 0 and it means the main diagonal.
  *
  * The matrix is not required to be square.
  *
  * This class represents an expression of the main diagonal, or any sub/super diagonal
  * of a square matrix. It is the return type of MatrixBase::diagonal() and most of the
  * time this is the only way it is used.
  *
  * \sa MatrixBase::diagonal()
  */
template<typename MatrixType, int DiagId>
struct ei_traits<DiagonalCoeffs<MatrixType,DiagId> >
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename ei_nested<MatrixType>::type MatrixTypeNested;
  typedef typename ei_unref<MatrixTypeNested>::type _MatrixTypeNested;
  enum {
    AbsDiagId = DiagId<0 ? -DiagId : DiagId,
    RowsAtCompileTime = int(MatrixType::SizeAtCompileTime) == Dynamic ? Dynamic
                      : EIGEN_ENUM_MIN(MatrixType::RowsAtCompileTime,
                                       MatrixType::ColsAtCompileTime) - AbsDiagId,
    ColsAtCompileTime = 1,
    MaxRowsAtCompileTime = int(MatrixType::MaxSizeAtCompileTime) == Dynamic ? Dynamic
                            : EIGEN_ENUM_MIN(MatrixType::MaxRowsAtCompileTime,
                                             MatrixType::MaxColsAtCompileTime),
    MaxColsAtCompileTime = 1,
    Flags = (unsigned int)_MatrixTypeNested::Flags & (HereditaryBits | LinearAccessBit),
    CoeffReadCost = _MatrixTypeNested::CoeffReadCost
  };
};

template<typename MatrixType, int DiagId> class DiagonalCoeffs
   : public MatrixBase<DiagonalCoeffs<MatrixType, DiagId> >
{
    enum {
      AbsDiagId = DiagId<0 ? -DiagId : DiagId,
      OffsetRow = DiagId<0 ? -DiagId : 0,
      OffsetCol = DiagId<0 ? 0 : DiagId
    };
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(DiagonalCoeffs)

    inline DiagonalCoeffs(const MatrixType& matrix) : m_matrix(matrix) {}

    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(DiagonalCoeffs)

    inline int rows() const { return std::min(m_matrix.rows(), m_matrix.cols()) - AbsDiagId; }
    inline int cols() const { return 1; }

    inline Scalar& coeffRef(int row, int)
    {
      return m_matrix.const_cast_derived().coeffRef(row+OffsetRow, row+OffsetCol);
    }

    inline const Scalar coeff(int row, int) const
    {
      return m_matrix.coeff(row+OffsetRow, row+OffsetCol);
    }

    inline Scalar& coeffRef(int index)
    {
      return m_matrix.const_cast_derived().coeffRef(index+OffsetRow, index+OffsetCol);
    }

    inline const Scalar coeff(int index) const
    {
      return m_matrix.coeff(index+OffsetRow, index+OffsetCol);
    }

  protected:

    const typename MatrixType::Nested m_matrix;
};

/** \returns an expression of the main diagonal of the matrix \c *this
  *
  * \c *this is not required to be square.
  *
  * Example: \include MatrixBase_diagonal.cpp
  * Output: \verbinclude MatrixBase_diagonal.out
  *
  * \sa class DiagonalCoeffs */
template<typename Derived>
inline DiagonalCoeffs<Derived>
MatrixBase<Derived>::diagonal()
{
  return DiagonalCoeffs<Derived>(derived());
}

/** This is the const version of diagonal(). */
template<typename Derived>
inline const DiagonalCoeffs<Derived>
MatrixBase<Derived>::diagonal() const
{
  return DiagonalCoeffs<Derived>(derived());
}

/** \returns an expression of the \a Id-th sub or super diagonal of the matrix \c *this
  *
  * \c *this is not required to be square.
  *
  * The template parameter \a Id represent a super diagonal if \a Id > 0
  * and a sub diagonal otherwise. \a Id == 0 is equivalent to the main diagonal.
  *
  * \sa MatrixBase::diagonal(), class DiagonalCoeffs */
template<typename Derived>
template<int Id>
inline DiagonalCoeffs<Derived,Id>
MatrixBase<Derived>::diagonal()
{
  return DiagonalCoeffs<Derived,Id>(derived());
}

/** This is the const version of diagonal<int>(). */
template<typename Derived>
template<int Id>
inline const DiagonalCoeffs<Derived,Id>
MatrixBase<Derived>::diagonal() const
{
  return DiagonalCoeffs<Derived,Id>(derived());
}

#endif // EIGEN_DIAGONALCOEFFS_H
