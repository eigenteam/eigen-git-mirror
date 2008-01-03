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

#ifndef EIGEN_ZERO_H
#define EIGEN_ZERO_H

/** \class Zero
  *
  * \brief Expression of a zero matrix or vector.
  *
  * \sa MatrixBase::zero(), MatrixBase::zero(int), MatrixBase::zero(int,int)
  */
template<typename MatrixType> class Zero : NoOperatorEquals,
  public MatrixBase<typename MatrixType::Scalar, Zero<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    friend class MatrixBase<Scalar, Zero<MatrixType> >;
  
  private:
    static const int _RowsAtCompileTime = MatrixType::RowsAtCompileTime,
                     _ColsAtCompileTime = MatrixType::ColsAtCompileTime;

    const Zero& _ref() const { return *this; }
    int _rows() const { return m_rows; }
    int _cols() const { return m_cols; }
    
    Scalar _coeff(int, int) const
    {
      return static_cast<Scalar>(0);
    }
    
  public:
    Zero(int rows, int cols) : m_rows(rows), m_cols(cols)
    {
      assert(rows > 0
          && (_RowsAtCompileTime == Dynamic || _RowsAtCompileTime == rows)
          && cols > 0
          && (_ColsAtCompileTime == Dynamic || _ColsAtCompileTime == cols));
    }
    
  protected:
    int m_rows, m_cols;
};

/** \returns an expression of a zero matrix.
  *
  * The parameters \a rows and \a cols are the number of rows and of columns of
  * the returned matrix. Must be compatible with this MatrixBase type.
  *
  * This variant is meant to be used for dynamic-size matrix types. For fixed-size types,
  * it is redundant to pass \a rows and \a cols as arguments, so zero() should be used
  * instead.
  *
  * Example: \include MatrixBase_zero_int_int.cpp
  * Output: \verbinclude MatrixBase_zero_int_int.out
  *
  * \sa zero(), zero(int)
  */
template<typename Scalar, typename Derived>
const Zero<Derived> MatrixBase<Scalar, Derived>::zero(int rows, int cols)
{
  return Zero<Derived>(rows, cols);
}

/** \returns an expression of a zero vector.
  *
  * The parameter \a size is the size of the returned vector.
  * Must be compatible with this MatrixBase type.
  *
  * \only_for_vectors
  *
  * This variant is meant to be used for dynamic-size vector types. For fixed-size types,
  * it is redundant to pass \a size as argument, so zero() should be used
  * instead.
  *
  * Example: \include MatrixBase_zero_int.cpp
  * Output: \verbinclude MatrixBase_zero_int.out
  *
  * \sa zero(), zero(int,int)
  */
template<typename Scalar, typename Derived>
const Zero<Derived> MatrixBase<Scalar, Derived>::zero(int size)
{
  assert(IsVectorAtCompileTime);
  if(RowsAtCompileTime == 1) return Zero<Derived>(1, size);
  else return Zero<Derived>(size, 1);
}

/** \returns an expression of a fixed-size zero matrix or vector.
  *
  * This variant is only for fixed-size MatrixBase types. For dynamic-size types, you
  * need to use the variants taking size arguments.
  *
  * Example: \include MatrixBase_zero.cpp
  * Output: \verbinclude MatrixBase_zero.out
  *
  * \sa zero(int), zero(int,int)
  */
template<typename Scalar, typename Derived>
const Zero<Derived> MatrixBase<Scalar, Derived>::zero()
{
  return Zero<Derived>(RowsAtCompileTime, ColsAtCompileTime);
}

#endif // EIGEN_ZERO_H
