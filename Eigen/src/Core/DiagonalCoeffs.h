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

#ifndef EIGEN_DIAGONALCOEFFS_H
#define EIGEN_DIAGONALCOEFFS_H

/** \class DiagonalCoeffs
  *
  * \brief Expression of the main diagonal of a square matrix
  *
  * \param MatrixType the type of the object in which we are taking the main diagonal
  *
  * This class represents an expression of the main diagonal of a square matrix.
  * It is the return type of MatrixBase::diagonal() and most of the time this is
  * the only way it is used.
  *
  * \sa MatrixBase::diagonal()
  */
template<typename MatrixType> class DiagonalCoeffs
  : public MatrixBase<typename MatrixType::Scalar, DiagonalCoeffs<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Ref MatRef;
    friend class MatrixBase<Scalar, DiagonalCoeffs<MatrixType> >;
    
    DiagonalCoeffs(const MatRef& matrix) : m_matrix(matrix) {}
    
    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(DiagonalCoeffs)
    
  private:
    enum {
      RowsAtCompileTime = MatrixType::Traits::RowsAtCompileTime,
      ColsAtCompileTime = 1
    };

    const DiagonalCoeffs& _ref() const { return *this; }
    int _rows() const { return std::min(m_matrix.rows(), m_matrix.cols()); }
    int _cols() const { return 1; }
    
    Scalar& _coeffRef(int row, int)
    {
      return m_matrix.coeffRef(row, row);
    }
    
    Scalar _coeff(int row, int) const
    {
      return m_matrix.coeff(row, row);
    }
    
  protected:
    MatRef m_matrix;
};

/** \returns an expression of the main diagonal of *this, which must be a square matrix.
  *
  * Example: \include MatrixBase_diagonal.cpp
  * Output: \verbinclude MatrixBase_diagonal.out
  *
  * \sa class DiagonalCoeffs */
template<typename Scalar, typename Derived>
DiagonalCoeffs<Derived>
MatrixBase<Scalar, Derived>::diagonal()
{
  return DiagonalCoeffs<Derived>(ref());
}

/** This is the const version of diagonal(). */
template<typename Scalar, typename Derived>
const DiagonalCoeffs<Derived>
MatrixBase<Scalar, Derived>::diagonal() const
{
  return DiagonalCoeffs<Derived>(ref());
}

#endif // EIGEN_DIAGONALCOEFFS_H
