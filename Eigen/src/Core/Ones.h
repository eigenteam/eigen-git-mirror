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

/** \class Ones
  *
  * \brief Expression of a matrix where all coefficients equal one.
  *
  * \sa MatrixBase::ones(), MatrixBase::ones(int), MatrixBase::ones(int,int)
  */
template<typename MatrixType> class Ones : NoOperatorEquals,
  public MatrixBase<typename MatrixType::Scalar, Ones<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    friend class MatrixBase<Scalar, Ones<MatrixType> >;
  
    static const int RowsAtCompileTime = MatrixType::RowsAtCompileTime,
                     ColsAtCompileTime = MatrixType::ColsAtCompileTime;

  private:
  
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
          && (RowsAtCompileTime == Dynamic || RowsAtCompileTime == rows)
          && cols > 0
          && (ColsAtCompileTime == Dynamic || ColsAtCompileTime == cols));
    }
    
  protected:
    int m_rows, m_cols;
};

/** \returns an expression of a matrix where all coefficients equal one.
  *
  * The parameters \a rows and \a cols are the number of rows and of columns of
  * the returned matrix. Must be compatible with this MatrixBase type.
  *
  * This variant is meant to be used for dynamic-size matrix types. For fixed-size types,
  * it is redundant to pass \a rows and \a cols as arguments, so ones() should be used
  * instead.
  *
  * Example: \include MatrixBase_ones_int_int.cpp
  * Output: \verbinclude MatrixBase_ones_int_int.out
  *
  * \sa ones(), ones(int)
  */
template<typename Scalar, typename Derived>
const Ones<Derived> MatrixBase<Scalar, Derived>::ones(int rows, int cols)
{
  return Ones<Derived>(rows, cols);
}

/** \returns an expression of a vector where all coefficients equal one.
  *
  * The parameter \a size is the size of the returned vector.
  * Must be compatible with this MatrixBase type.
  *
  * \only_for_vectors
  *
  * This variant is meant to be used for dynamic-size vector types. For fixed-size types,
  * it is redundant to pass \a size as argument, so ones() should be used
  * instead.
  *
  * Example: \include MatrixBase_ones_int.cpp
  * Output: \verbinclude MatrixBase_ones_int.out
  *
  * \sa ones(), ones(int,int)
  */
template<typename Scalar, typename Derived>
const Ones<Derived> MatrixBase<Scalar, Derived>::ones(int size)
{
  assert(Traits::IsVectorAtCompileTime);
  if(Traits::RowsAtCompileTime == 1) return Ones<Derived>(1, size);
  else return Ones<Derived>(size, 1);
}

/** \returns an expression of a fixed-size matrix or vector where all coefficients equal one.
  *
  * This variant is only for fixed-size MatrixBase types. For dynamic-size types, you
  * need to use the variants taking size arguments.
  *
  * Example: \include MatrixBase_ones.cpp
  * Output: \verbinclude MatrixBase_ones.out
  *
  * \sa ones(int), ones(int,int)
  */
template<typename Scalar, typename Derived>
const Ones<Derived> MatrixBase<Scalar, Derived>::ones()
{
  return Ones<Derived>(Traits::RowsAtCompileTime, Traits::ColsAtCompileTime);
}

template<typename Scalar, typename Derived>
bool MatrixBase<Scalar, Derived>::isOnes
(const typename NumTraits<Scalar>::Real& prec = precision<Scalar>()) const
{
  for(int j = 0; j < col(); j++)
    for(int i = 0; i < row(); i++)
      if(!isApprox(coeff(i, j), static_cast<Scalar>(1)))
        return false;
  return true;
}

#endif // EIGEN_ONES_H
