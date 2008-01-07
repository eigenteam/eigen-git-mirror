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

#ifndef EIGEN_IDENTITY_H
#define EIGEN_IDENTITY_H

/** \class Identity
  *
  * \brief Expression of the identity matrix of some size.
  *
  * \sa MatrixBase::identity(int)
  */
template<typename MatrixType> class Identity : NoOperatorEquals,
  public MatrixBase<typename MatrixType::Scalar, Identity<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    friend class MatrixBase<Scalar, Identity<MatrixType> >;
    
    Identity(int rows) : m_rows(rows)
    {
      assert(rows > 0 && RowsAtCompileTime == ColsAtCompileTime);
    }
    
  private:
    static const int RowsAtCompileTime = MatrixType::Traits::RowsAtCompileTime,
                     ColsAtCompileTime = MatrixType::Traits::ColsAtCompileTime;
    
    const Identity& _ref() const { return *this; }
    int _rows() const { return m_rows; }
    int _cols() const { return m_rows; }
    
    Scalar _coeff(int row, int col) const
    {
      return row == col ? static_cast<Scalar>(1) : static_cast<Scalar>(0);
    }
    
  protected:
    int m_rows;
};

/** \returns an expression of the identity matrix of given type and size.
  *
  * \param rows The number of rows of the identity matrix to return. If *this has
  *             fixed size, that size is used as the default argument for \a rows
  *             and is then the only allowed value. If *this has dynamic size,
  *             you can use any positive value for \a rows.
  *
  * \note An identity matrix is a square matrix, so it is required that the type of *this
  *       allows being a square matrix.
  *
  * Example: \include MatrixBase_identity_int.cpp
  * Output: \verbinclude MatrixBase_identity_int.out
  *
  * \sa class Identity, isIdentity()
  */
template<typename Scalar, typename Derived>
const Identity<Derived> MatrixBase<Scalar, Derived>::identity(int rows)
{
  return Identity<Derived>(rows);
}

/** \returns true if *this is approximately equal to the identity matrix,
  *          within the precision given by \a prec.
  *
  * Example: \include MatrixBase_isIdentity.cpp
  * Output: \verbinclude MatrixBase_isIdentity.out
  *
  * \sa class Identity, identity(int)
  */
template<typename Scalar, typename Derived>
bool MatrixBase<Scalar, Derived>::isIdentity
(typename NumTraits<Scalar>::Real prec) const
{
  if(cols() != rows()) return false;
  for(int j = 0; j < cols(); j++)
  {
    if(!Eigen::isApprox(coeff(j, j), static_cast<Scalar>(1), prec))
      return false;
    for(int i = 0; i < j; i++)
      if(!Eigen::isMuchSmallerThan(coeff(i, j), static_cast<Scalar>(1), prec))
        return false;
  }
  return true;
}


#endif // EIGEN_IDENTITY_H
