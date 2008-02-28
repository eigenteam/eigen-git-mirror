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

#ifndef EIGEN_RANDOM_H
#define EIGEN_RANDOM_H

/** \class Random
  *
  * \brief Expression of a random matrix or vector.
  *
  * \sa MatrixBase::random(), MatrixBase::random(int), MatrixBase::random(int,int),
  *     MatrixBase::setRandom()
  */
template<typename MatrixType> class Random : NoOperatorEquals,
  public MatrixBase<typename MatrixType::Scalar, Random<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    friend class MatrixBase<Scalar, Random>;
    typedef MatrixBase<Scalar, Random> Base;
  
  private:
    enum {
      RowsAtCompileTime = MatrixType::Traits::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::Traits::ColsAtCompileTime,
      MaxRowsAtCompileTime = MatrixType::Traits::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::Traits::MaxColsAtCompileTime
    };
  
    const Random& _ref() const { return *this; }
    int _rows() const { return m_rows.value(); }
    int _cols() const { return m_cols.value(); }
    
    Scalar _coeff(int, int) const
    {
      return ei_random<Scalar>();
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
    const IntAtRunTimeIfDynamic<RowsAtCompileTime> m_rows;
    const IntAtRunTimeIfDynamic<ColsAtCompileTime> m_cols;
};

/** \returns a random matrix (not an expression, the matrix is immediately evaluated).
  *
  * The parameters \a rows and \a cols are the number of rows and of columns of
  * the returned matrix. Must be compatible with this MatrixBase type.
  *
  * This variant is meant to be used for dynamic-size matrix types. For fixed-size types,
  * it is redundant to pass \a rows and \a cols as arguments, so ei_random() should be used
  * instead.
  *
  * Example: \include MatrixBase_random_int_int.cpp
  * Output: \verbinclude MatrixBase_random_int_int.out
  *
  * \sa ei_random(), ei_random(int)
  */
template<typename Scalar, typename Derived>
const Eval<Random<Derived> >
MatrixBase<Scalar, Derived>::random(int rows, int cols)
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
  * it is redundant to pass \a size as argument, so ei_random() should be used
  * instead.
  *
  * Example: \include MatrixBase_random_int.cpp
  * Output: \verbinclude MatrixBase_random_int.out
  *
  * \sa ei_random(), ei_random(int,int)
  */
template<typename Scalar, typename Derived>
const Eval<Random<Derived> >
MatrixBase<Scalar, Derived>::random(int size)
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
  * \sa ei_random(int), ei_random(int,int)
  */
template<typename Scalar, typename Derived>
const Eval<Random<Derived> >
MatrixBase<Scalar, Derived>::random()
{
  return Random<Derived>(Traits::RowsAtCompileTime, Traits::ColsAtCompileTime).eval();
}

/** Sets all coefficients in this expression to random values.
  *
  * Example: \include MatrixBase_setRandom.cpp
  * Output: \verbinclude MatrixBase_setRandom.out
  *
  * \sa class Random, ei_random()
  */
template<typename Scalar, typename Derived>
Derived& MatrixBase<Scalar, Derived>::setRandom()
{
  return *this = Random<Derived>(rows(), cols());
}

#endif // EIGEN_RANDOM_H
