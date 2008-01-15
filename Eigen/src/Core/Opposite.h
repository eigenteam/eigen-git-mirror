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

#ifndef EIGEN_OPPOSITE_H
#define EIGEN_OPPOSITE_H

/** \class Opposite
  *
  * \brief Expression of the opposite of a matrix or vector
  *
  * \param MatrixType the type of which we are taking the opposite
  *
  * This class represents an expression of the opposite of a matrix or vector.
  * It is the return type of the unary operator- for matrices or vectors, and most
  * of the time this is the only way it is used.
  *
  * \sa class Difference
  */
template<typename MatrixType> class Opposite : NoOperatorEquals,
  public MatrixBase<typename MatrixType::Scalar, Opposite<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Ref MatRef;
    friend class MatrixBase<Scalar, Opposite>;
    typedef MatrixBase<Scalar, Opposite> Base;

    Opposite(const MatRef& matrix) : m_matrix(matrix) {}
    
  private:
    enum {
      RowsAtCompileTime = MatrixType::Traits::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::Traits::ColsAtCompileTime,
      MaxRowsAtCompileTime = MatrixType::Traits::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::Traits::MaxColsAtCompileTime
    };

    const Opposite& _ref() const { return *this; }
    int _rows() const { return m_matrix.rows(); }
    int _cols() const { return m_matrix.cols(); }
    
    Scalar _coeff(int row, int col) const
    {
      return -(m_matrix.coeff(row, col));
    }
    
  protected:
    const MatRef m_matrix;
};

/** \returns an expression of the opposite of \c *this
  */
template<typename Scalar, typename Derived>
const Opposite<Derived>
MatrixBase<Scalar, Derived>::operator-() const
{
  return Opposite<Derived>(ref());
}

#endif // EIGEN_OPPOSITE_H
