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

#ifndef EIGEN_IDENTITY_H
#define EIGEN_IDENTITY_H

/** \class Identity
  *
  * \brief Expression of the identity matrix of some size.
  *
  * \sa MatrixBase::identity(), MatrixBase::identity(int,int), MatrixBase::setIdentity()
  */
template<typename MatrixType>
struct Scalar<Identity<MatrixType> >
{ typedef typename Scalar<MatrixType>::Type Type; };

template<typename MatrixType> class Identity : NoOperatorEquals,
  public MatrixBase<Identity<MatrixType> >
{
  public:
    typedef typename Scalar<MatrixType>::Type Scalar;
    friend class MatrixBase<Identity>;
    friend class MatrixBase<Identity>::Traits;
    typedef MatrixBase<Identity> Base;

    Identity(int rows, int cols) : m_rows(rows), m_cols(cols)
    {
      assert(rows > 0
          && (RowsAtCompileTime == Dynamic || RowsAtCompileTime == rows)
          && cols > 0
          && (ColsAtCompileTime == Dynamic || ColsAtCompileTime == cols));
    }

  private:
    enum {
      RowsAtCompileTime = MatrixType::Traits::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::Traits::ColsAtCompileTime,
      MaxRowsAtCompileTime = MatrixType::Traits::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::Traits::MaxColsAtCompileTime
    };

    const Identity& _asArg() const { return *this; }
    int _rows() const { return m_rows.value(); }
    int _cols() const { return m_cols.value(); }

    Scalar _coeff(int row, int col) const
    {
      return row == col ? static_cast<Scalar>(1) : static_cast<Scalar>(0);
    }

  protected:
    const IntAtRunTimeIfDynamic<RowsAtCompileTime> m_rows;
    const IntAtRunTimeIfDynamic<ColsAtCompileTime> m_cols;
};

/** \returns an expression of the identity matrix (not necessarily square).
  *
  * The parameters \a rows and \a cols are the number of rows and of columns of
  * the returned matrix. Must be compatible with this MatrixBase type.
  *
  * This variant is meant to be used for dynamic-size matrix types. For fixed-size types,
  * it is redundant to pass \a rows and \a cols as arguments, so identity() should be used
  * instead.
  *
  * Example: \include MatrixBase_identity_int_int.cpp
  * Output: \verbinclude MatrixBase_identity_int_int.out
  *
  * \sa identity(), setIdentity(), isIdentity()
  */
template<typename Derived>
const Identity<Derived> MatrixBase<Derived>::identity(int rows, int cols)
{
  return Identity<Derived>(rows, cols);
}

/** \returns an expression of the identity matrix (not necessarily square).
  *
  * This variant is only for fixed-size MatrixBase types. For dynamic-size types, you
  * need to use the variant taking size arguments.
  *
  * Example: \include MatrixBase_identity.cpp
  * Output: \verbinclude MatrixBase_identity.out
  *
  * \sa identity(int,int), setIdentity(), isIdentity()
  */
template<typename Derived>
const Identity<Derived> MatrixBase<Derived>::identity()
{
  return Identity<Derived>(Traits::RowsAtCompileTime, Traits::ColsAtCompileTime);
}

/** \returns true if *this is approximately equal to the identity matrix
  *          (not necessarily square),
  *          within the precision given by \a prec.
  *
  * Example: \include MatrixBase_isIdentity.cpp
  * Output: \verbinclude MatrixBase_isIdentity.out
  *
  * \sa class Identity, identity(), identity(int,int), setIdentity()
  */
template<typename Derived>
bool MatrixBase<Derived>::isIdentity
(typename NumTraits<Scalar>::Real prec) const
{
  for(int j = 0; j < cols(); j++)
  {
    for(int i = 0; i < rows(); i++)
    {
      if(i == j)
      {
        if(!ei_isApprox(coeff(i, j), static_cast<Scalar>(1), prec))
          return false;
      }
      else
      {
        if(!ei_isMuchSmallerThan(coeff(i, j), static_cast<RealScalar>(1), prec))
          return false;
      }
    }
  }
  return true;
}

/** Writes the identity expression (not necessarily square) into *this.
  *
  * Example: \include MatrixBase_setIdentity.cpp
  * Output: \verbinclude MatrixBase_setIdentity.out
  *
  * \sa class Identity, identity(), identity(int,int), isIdentity()
  */
template<typename Derived>
Derived& MatrixBase<Derived>::setIdentity()
{
  return *this = Identity<Derived>(rows(), cols());
}


#endif // EIGEN_IDENTITY_H
