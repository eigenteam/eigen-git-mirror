// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
//
// Eigen is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either 
// version 3 of the License, or (at your option) any later version.
//
// Alternatively, you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of 
// the License, or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License or the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public 
// License and a copy of the GNU General Public License along with
// Eigen. If not, see <http://www.gnu.org/licenses/>.

#ifndef EIGEN_ZERO_H
#define EIGEN_ZERO_H

/** \class Zero
  *
  * \brief Expression of a zero matrix or vector.
  *
  * \sa MatrixBase::zero(), MatrixBase::zero(int), MatrixBase::zero(int,int),
  *     MatrixBase::setZero(), MatrixBase::isZero()
  */
template<typename MatrixType> class Zero : NoOperatorEquals,
  public MatrixBase<typename MatrixType::Scalar, Zero<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    friend class MatrixBase<Scalar, Zero>;
    typedef MatrixBase<Scalar, Zero> Base;
  
  private:
    enum {
      RowsAtCompileTime = MatrixType::Traits::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::Traits::ColsAtCompileTime,
      MaxRowsAtCompileTime = MatrixType::Traits::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::Traits::MaxColsAtCompileTime
    };

    const Zero& _ref() const { return *this; }
    int _rows() const { return m_rows.value(); }
    int _cols() const { return m_cols.value(); }
    
    Scalar _coeff(int, int) const
    {
      return static_cast<Scalar>(0);
    }
    
  public:
    Zero(int rows, int cols) : m_rows(rows), m_cols(cols)
    {
      assert(rows > 0
          && (RowsAtCompileTime == Dynamic || RowsAtCompileTime == rows)
          && cols > 0
          && (ColsAtCompileTime == Dynamic || ColsAtCompileTime == cols));
    }
    
  protected:
    const IntAtRunTimeIfDynamic<RowsAtCompileTime> m_rows;
    const IntAtRunTimeIfDynamic<ColsAtCompileTime> m_cols;
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
  assert(Traits::IsVectorAtCompileTime);
  if(Traits::RowsAtCompileTime == 1) return Zero<Derived>(1, size);
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
  return Zero<Derived>(Traits::RowsAtCompileTime, Traits::ColsAtCompileTime);
}

/** \returns true if *this is approximately equal to the zero matrix,
  *          within the precision given by \a prec.
  *
  * Example: \include MatrixBase_isZero.cpp
  * Output: \verbinclude MatrixBase_isZero.out
  *
  * \sa class Zero, zero()
  */
template<typename Scalar, typename Derived>
bool MatrixBase<Scalar, Derived>::isZero
(typename NumTraits<Scalar>::Real prec) const
{
  for(int j = 0; j < cols(); j++)
    for(int i = 0; i < rows(); i++)
      if(!ei_isMuchSmallerThan(coeff(i, j), static_cast<Scalar>(1), prec))
        return false;
  return true;
}

/** Sets all coefficients in this expression to zero.
  *
  * Example: \include MatrixBase_setZero.cpp
  * Output: \verbinclude MatrixBase_setZero.out
  *
  * \sa class Zero, zero()
  */
template<typename Scalar, typename Derived>
Derived& MatrixBase<Scalar, Derived>::setZero()
{
  return *this = Zero<Derived>(rows(), cols());
}

#endif // EIGEN_ZERO_H
