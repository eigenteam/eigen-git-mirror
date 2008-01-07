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

#ifndef EIGEN_TRANSPOSE_H
#define EIGEN_TRANSPOSE_H

/** \class Transpose
  *
  * \brief Expression of the transpose of a matrix
  *
  * \param MatrixType the type of the object of which we are taking the transpose
  *
  * This class represents an expression of the transpose of a matrix.
  * It is the return type of MatrixBase::transpose() and MatrixBase::adjoint()
  * and most of the time this is the only way it is used.
  *
  * \sa MatrixBase::transpose(), MatrixBase::adjoint()
  */
template<typename MatrixType> class Transpose
  : public MatrixBase<typename MatrixType::Scalar, Transpose<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Ref MatRef;
    friend class MatrixBase<Scalar, Transpose<MatrixType> >;
    
    Transpose(const MatRef& matrix) : m_matrix(matrix) {}
    
    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Transpose)
    
  private:
    static const int RowsAtCompileTime = MatrixType::Traits::ColsAtCompileTime,
                     ColsAtCompileTime = MatrixType::Traits::RowsAtCompileTime;

    const Transpose& _ref() const { return *this; }
    int _rows() const { return m_matrix.cols(); }
    int _cols() const { return m_matrix.rows(); }
    
    Scalar& _coeffRef(int row, int col)
    {
      return m_matrix.coeffRef(col, row);
    }
    
    Scalar _coeff(int row, int col) const
    {
      return m_matrix.coeff(col, row);
    }
    
  protected:
    MatRef m_matrix;
};

/** \returns an expression of the transpose of *this.
  *
  * Example: \include MatrixBase_transpose.cpp
  * Output: \verbinclude MatrixBase_transpose.out
  *
  * \sa adjoint(), class DiagonalCoeffs */
template<typename Scalar, typename Derived>
Transpose<Derived>
MatrixBase<Scalar, Derived>::transpose()
{
  return Transpose<Derived>(ref());
}

/** This is the const version of transpose(). \sa adjoint() */
template<typename Scalar, typename Derived>
const Transpose<Derived>
MatrixBase<Scalar, Derived>::transpose() const
{
  return Transpose<Derived>(ref());
}

#endif // EIGEN_TRANSPOSE_H
