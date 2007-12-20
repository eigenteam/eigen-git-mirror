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

#ifndef EIGEN_TRANSPOSE_H
#define EIGEN_TRANSPOSE_H

template<typename MatrixType> class Transpose
  : public MatrixBase<typename MatrixType::Scalar, Transpose<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Ref MatRef;
    friend class MatrixBase<Scalar, Transpose<MatrixType> >;
    
    Transpose(const MatRef& matrix) : m_matrix(matrix) {}
    
    Transpose(const Transpose& other)
      : m_matrix(other.m_matrix) {}
    
    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Transpose)
    
  private:
    static const int _RowsAtCompileTime = MatrixType::ColsAtCompileTime,
                     _ColsAtCompileTime = MatrixType::RowsAtCompileTime;

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

template<typename Scalar, typename Derived>
Transpose<Derived>
MatrixBase<Scalar, Derived>::transpose() const
{
  return Transpose<Derived>(static_cast<Derived*>(const_cast<MatrixBase*>(this))->ref());
}

#endif // EIGEN_TRANSPOSE_H
