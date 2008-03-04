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

#ifndef EIGEN_ONES_H
#define EIGEN_ONES_H

/** \class Ones
  *
  * \brief Expression of a matrix where all coefficients equal one.
  *
  * \sa MatrixBase::ones(), MatrixBase::ones(int), MatrixBase::ones(int,int),
  *     MatrixBase::setOnes(), MatrixBase::isOnes()
  */
template<typename MatrixType> class Ones : NoOperatorEquals,
  public MatrixBase<typename MatrixType::Scalar, Ones<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    friend class MatrixBase<Scalar, Ones>;
    friend class MatrixBase<Scalar, Ones>::Traits;
    typedef MatrixBase<Scalar, Ones> Base;

  private:
    enum {
      RowsAtCompileTime = MatrixType::Traits::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::Traits::ColsAtCompileTime,
      MaxRowsAtCompileTime = MatrixType::Traits::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::Traits::MaxColsAtCompileTime
    };

    const Ones& _asArg() const { return *this; }
    int _rows() const { return m_rows.value(); }
    int _cols() const { return m_cols.value(); }

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
    const IntAtRunTimeIfDynamic<RowsAtCompileTime> m_rows;
    const IntAtRunTimeIfDynamic<ColsAtCompileTime> m_cols;
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
  * \sa ones(), ones(int), isOnes(), class Ones
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
  * \sa ones(), ones(int,int), isOnes(), class Ones
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
  * \sa ones(int), ones(int,int), isOnes(), class Ones
  */
template<typename Scalar, typename Derived>
const Ones<Derived> MatrixBase<Scalar, Derived>::ones()
{
  return Ones<Derived>(Traits::RowsAtCompileTime, Traits::ColsAtCompileTime);
}

/** \returns true if *this is approximately equal to the matrix where all coefficients
  *          are equal to 1, within the precision given by \a prec.
  *
  * Example: \include MatrixBase_isOnes.cpp
  * Output: \verbinclude MatrixBase_isOnes.out
  *
  * \sa class Ones, ones()
  */
template<typename Scalar, typename Derived>
bool MatrixBase<Scalar, Derived>::isOnes
(typename NumTraits<Scalar>::Real prec) const
{
  for(int j = 0; j < cols(); j++)
    for(int i = 0; i < rows(); i++)
      if(!ei_isApprox(coeff(i, j), static_cast<Scalar>(1), prec))
        return false;
  return true;
}

/** Sets all coefficients in this expression to one.
  *
  * Example: \include MatrixBase_setOnes.cpp
  * Output: \verbinclude MatrixBase_setOnes.out
  *
  * \sa class Ones, ones()
  */
template<typename Scalar, typename Derived>
Derived& MatrixBase<Scalar, Derived>::setOnes()
{
  return *this = Ones<Derived>(rows(), cols());
}

#endif // EIGEN_ONES_H
