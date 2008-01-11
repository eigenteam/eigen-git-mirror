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

#ifndef EIGEN_RANDOM_H
#define EIGEN_RANDOM_H

/** \class Random
  *
  * \brief Expression of a random matrix or vector.
  *
  * \sa MatrixBase::random(), MatrixBase::random(int), MatrixBase::random(int,int)
  */
template<typename MatrixType> class Random : NoOperatorEquals,
  public MatrixBase<typename MatrixType::Scalar, Random<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    friend class MatrixBase<Scalar, Random<MatrixType> >;
  
  private:
    enum {
      RowsAtCompileTime = MatrixType::Traits::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::Traits::ColsAtCompileTime
    };
  
    const Random& _ref() const { return *this; }
    int _rows() const { return m_rows; }
    int _cols() const { return m_cols; }
    
    Scalar _coeff(int, int) const
    {
      return Eigen::random<Scalar>();
    }
  
  public:
    Random(int rows, int cols) : m_rows(rows), m_cols(cols)
    {
      assert(rows > 0
          && (RowsAtCompileTime == Dynamic || RowsAtCompileTime == rows)
          && cols > 0
          && (ColsAtCompileTime == Dynamic || ColsAtCompileTime == cols));
    }
    
  protected:
    const int m_rows, m_cols;
};

/** \returns a random matrix (not an expression, the matrix is immediately evaluated).
  *
  * The parameters \a rows and \a cols are the number of rows and of columns of
  * the returned matrix. Must be compatible with this MatrixBase type.
  *
  * This variant is meant to be used for dynamic-size matrix types. For fixed-size types,
  * it is redundant to pass \a rows and \a cols as arguments, so random() should be used
  * instead.
  *
  * Example: \include MatrixBase_random_int_int.cpp
  * Output: \verbinclude MatrixBase_random_int_int.out
  *
  * \sa random(), random(int)
  */
template<typename Scalar, typename Derived>
const Eval<Random<Derived> > MatrixBase<Scalar, Derived>::random(int rows, int cols)
{
  return Random<Derived>(rows, cols).eval();
}

/** \returns a random vector (not an expression, the vector is immediately evaluated).
  *
  * The parameter \a size is the size of the returned vector.
  * Must be compatible with this MatrixBase type.
  *
  * \only_for_vectors
  *
  * This variant is meant to be used for dynamic-size vector types. For fixed-size types,
  * it is redundant to pass \a size as argument, so random() should be used
  * instead.
  *
  * Example: \include MatrixBase_random_int.cpp
  * Output: \verbinclude MatrixBase_random_int.out
  *
  * \sa random(), random(int,int)
  */
template<typename Scalar, typename Derived>
const Eval<Random<Derived> > MatrixBase<Scalar, Derived>::random(int size)
{
  assert(Traits::IsVectorAtCompileTime);
  if(Traits::RowsAtCompileTime == 1) return Random<Derived>(1, size).eval();
  else return Random<Derived>(size, 1).eval();
}

/** \returns a fixed-size random matrix or vector
  * (not an expression, the matrix is immediately evaluated).
  *
  * This variant is only for fixed-size MatrixBase types. For dynamic-size types, you
  * need to use the variants taking size arguments.
  *
  * Example: \include MatrixBase_random.cpp
  * Output: \verbinclude MatrixBase_random.out
  *
  * \sa random(int), random(int,int)
  */
template<typename Scalar, typename Derived>
const Eval<Random<Derived> > MatrixBase<Scalar, Derived>::random()
{
  return Random<Derived>(Traits::RowsAtCompileTime, Traits::ColsAtCompileTime).eval();
}

/** Sets all coefficients in this expression to random values.
  *
  * Example: \include MatrixBase_setRandom.cpp
  * Output: \verbinclude MatrixBase_setRandom.out
  *
  * \sa class Random, random()
  */
template<typename Scalar, typename Derived>
Derived& MatrixBase<Scalar, Derived>::setRandom()
{
  return *this = Random<Derived>(rows(), cols());
}

#endif // EIGEN_RANDOM_H
