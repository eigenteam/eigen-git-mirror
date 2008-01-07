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

#ifndef EIGEN_CONJUGATE_H
#define EIGEN_CONJUGATE_H

/** \class Conjugate
  *
  * \brief Expression of the complex conjugate of a matrix
  *
  * \param MatrixType the type of the object of which we are taking the complex conjugate
  *
  * This class represents an expression of the complex conjugate of a matrix.
  * It is the return type of MatrixBase::conjugate() and is also used by
  * MatrixBase::adjoint() and most of the time these are the only ways it is used.
  *
  * \sa MatrixBase::conjugate(), MatrixBase::adjoint()
  */
template<typename MatrixType> class Conjugate : NoOperatorEquals,
  public MatrixBase<typename MatrixType::Scalar, Conjugate<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Ref MatRef;
    friend class MatrixBase<Scalar, Conjugate<MatrixType> >;
    
    Conjugate(const MatRef& matrix) : m_matrix(matrix) {}
    
  private:
    static const int RowsAtCompileTime = MatrixType::Traits::RowsAtCompileTime,
                     ColsAtCompileTime = MatrixType::Traits::ColsAtCompileTime;

    const Conjugate& _ref() const { return *this; }
    int _rows() const { return m_matrix.rows(); }
    int _cols() const { return m_matrix.cols(); }
    
    Scalar _coeff(int row, int col) const
    {
      return conj(m_matrix.coeff(row, col));
    }
    
  protected:
    MatRef m_matrix;
};

/** \returns an expression of the complex conjugate of *this.
  *
  * \sa adjoint(), class Conjugate */
template<typename Scalar, typename Derived>
const Conjugate<Derived>
MatrixBase<Scalar, Derived>::conjugate() const
{
  return Conjugate<Derived>(ref());
}

/** \returns an expression of the adjoint (i.e. conjugate transpose) of *this.
  *
  * Example: \include MatrixBase_adjoint.cpp
  * Output: \verbinclude MatrixBase_adjoint.out
  *
  * \sa transpose(), conjugate(), class Transpose, class Conjugate */
template<typename Scalar, typename Derived>
const Transpose<Conjugate<Derived> >
MatrixBase<Scalar, Derived>::adjoint() const
{
  return conjugate().transpose();
}

#endif // EIGEN_CONJUGATE_H
