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

#ifndef EIGEN_ONES_H
#define EIGEN_ONES_H

template<typename MatrixType> class Ones : NoOperatorEquals,
  public MatrixBase<typename MatrixType::Scalar, Ones<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    friend class MatrixBase<Scalar, Ones<MatrixType> >;
  
  private:
    static const int _RowsAtCompileTime = MatrixType::RowsAtCompileTime,
                     _ColsAtCompileTime = MatrixType::ColsAtCompileTime;

    const Ones& _ref() const { return *this; }
    int _rows() const { return m_rows; }
    int _cols() const { return m_cols; }
    
    Scalar _coeff(int, int) const
    {
      return static_cast<Scalar>(1);
    }
  
  public:
    Ones(int rows, int cols) : m_rows(rows), m_cols(cols)
    {
      assert(rows > 0
          && (_RowsAtCompileTime == Dynamic || _RowsAtCompileTime == rows)
          && cols > 0
          && (_ColsAtCompileTime == Dynamic || _ColsAtCompileTime == cols));
    }
    
  protected:
    int m_rows, m_cols;
};

template<typename Scalar, typename Derived>
const Ones<Derived> MatrixBase<Scalar, Derived>::ones(int rows, int cols)
{
  return Ones<Derived>(rows, cols);
}

template<typename Scalar, typename Derived>
const Ones<Derived> MatrixBase<Scalar, Derived>::ones(int size)
{
  assert(IsVectorAtCompileTime);
  if(RowsAtCompileTime == 1) return Ones<Derived>(1, size);
  else return Ones<Derived>(size, 1);
}

template<typename Scalar, typename Derived>
const Ones<Derived> MatrixBase<Scalar, Derived>::ones()
{
  return Ones<Derived>(RowsAtCompileTime, ColsAtCompileTime);
}

#endif // EIGEN_ONES_H
