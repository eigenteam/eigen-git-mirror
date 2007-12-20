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

#ifndef EIGEN_CAST_H
#define EIGEN_CAST_H

template<typename NewScalar, typename MatrixType> class Cast : NoOperatorEquals,
  public MatrixBase<NewScalar, Cast<NewScalar, MatrixType> >
{
  public:
    typedef NewScalar Scalar;
    typedef typename MatrixType::Ref MatRef;
    friend class MatrixBase<Scalar, Cast<Scalar, MatrixType> >;
    
    Cast(const MatRef& matrix) : m_matrix(matrix) {}
    
    Cast(const Cast& other)
      : m_matrix(other.m_matrix) {}
    
  private:
    static const int _RowsAtCompileTime = MatrixType::RowsAtCompileTime,
                     _ColsAtCompileTime = MatrixType::ColsAtCompileTime;
    const Cast& _ref() const { return *this; }
    int _rows() const { return m_matrix.rows(); }
    int _cols() const { return m_matrix.cols(); }
    
    Scalar _coeff(int row, int col) const
    {
      return static_cast<Scalar>(m_matrix.coeff(row, col));
    }
    
  protected:
    MatRef m_matrix;
};

/** \returns an expression of *this with the \a Scalar type casted to
  * \a NewScalar. */
template<typename Scalar, typename Derived>
template<typename NewScalar>
const Cast<NewScalar, Derived>
MatrixBase<Scalar, Derived>::cast() const
{
  return Cast<NewScalar, Derived>(static_cast<const Derived*>(this)->ref());
}

#endif // EIGEN_CAST_H
