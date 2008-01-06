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

#ifndef EIGEN_IDENTITY_H
#define EIGEN_IDENTITY_H

template<typename MatrixType> class Identity : NoOperatorEquals,
  public MatrixBase<typename MatrixType::Scalar, Identity<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    friend class MatrixBase<Scalar, Identity<MatrixType> >;
    
    Identity(int rows) : m_rows(rows)
    {
      assert(rows > 0 && _RowsAtCompileTime == _ColsAtCompileTime);
    }
    
    static const TraversalOrder Order = Indifferent;
    static const int RowsAtCompileTime = MatrixType::RowsAtCompileTime,
                     ColsAtCompileTime = MatrixType::ColsAtCompileTime;
    
  private:
    static const TraversalOrder _Order = Indifferent;
    static const int _RowsAtCompileTime = MatrixType::RowsAtCompileTime,
                     _ColsAtCompileTime = MatrixType::ColsAtCompileTime;

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

template<typename Scalar, typename Derived>
const Identity<Derived> MatrixBase<Scalar, Derived>::identity(int rows)
{
  return Identity<Derived>(rows);
}

template<typename Scalar, typename Derived>
bool MatrixBase<Scalar, Derived>::isIdentity
(const typename NumTraits<Scalar>::Real& prec = precision<Scalar>()) const
{
  for(int j = 0; j < col(); j++)
  {
    if(!isApprox(coeff(j, j), static_cast<Scalar>(1)))
      return false;
    for(int i = 0; i < j; i++)
      if(!isMuchSmallerThan(coeff(i, j), static_cast<Scalar>(1)))
        return false;
  }
  return true;
}


#endif // EIGEN_IDENTITY_H
