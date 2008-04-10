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

#ifndef EIGEN_FUZZY_H
#define EIGEN_FUZZY_H

/** \returns \c true if \c *this is approximately equal to \a other, within the precision
  * determined by \a prec.
  *
  * \note The fuzzy compares are done multiplicatively. Two vectors \f$ v \f$ and \f$ w \f$
  * are considered to be approximately equal within precision \f$ p \f$ if
  * \f[ \Vert v - w \Vert \leqslant p\,\min(\Vert v\Vert, \Vert w\Vert). \f]
  * For matrices, the comparison is done on all columns.
  *
  * \note Because of the multiplicativeness of this comparison, one can't use this function
  * to check whether \c *this is approximately equal to the zero matrix or vector.
  * Indeed, \c isApprox(zero) returns false unless \c *this itself is exactly the zero matrix
  * or vector. If you want to test whether \c *this is zero, use ei_isMuchSmallerThan(const
  * RealScalar&, RealScalar) instead.
  *
  * \sa ei_isMuchSmallerThan(const RealScalar&, RealScalar) const
  */
template<typename Derived>
template<typename OtherDerived>
bool MatrixBase<Derived>::isApprox(
  const MatrixBase<OtherDerived>& other,
  typename NumTraits<Scalar>::Real prec
) const
{
  ei_assert(rows() == other.rows() && cols() == other.cols());
  if(IsVectorAtCompileTime)
  {
    return((*this - other).norm2() <= std::min(norm2(), other.norm2()) * prec * prec);
  }
  else
  {
    typename Derived::Nested nested(derived());
    typename OtherDerived::Nested otherNested(other.derived());
    for(int i = 0; i < cols(); i++)
      if((nested.col(i) - otherNested.col(i)).norm2()
         > std::min(nested.col(i).norm2(), otherNested.col(i).norm2()) * prec * prec)
        return false;
    return true;
  }
}

/** \returns \c true if the norm of \c *this is much smaller than \a other,
  * within the precision determined by \a prec.
  *
  * \note The fuzzy compares are done multiplicatively. A vector \f$ v \f$ is
  * considered to be much smaller than \f$ x \f$ within precision \f$ p \f$ if
  * \f[ \Vert v \Vert \leqslant p\,\vert x\vert. \f]
  * For matrices, the comparison is done on all columns.
  *
  * \sa isApprox(), isMuchSmallerThan(const MatrixBase<OtherDerived>&, RealScalar) const
  */
template<typename Derived>
bool MatrixBase<Derived>::isMuchSmallerThan(
  const typename NumTraits<Scalar>::Real& other,
  typename NumTraits<Scalar>::Real prec
) const
{
  if(IsVectorAtCompileTime)
  {
    return(norm2() <= ei_abs2(other * prec));
  }
  else
  {
    typename Derived::Nested nested(*this);
    for(int i = 0; i < cols(); i++)
      if(nested.col(i).norm2() > ei_abs2(other * prec))
        return false;
    return true;
  }
}

/** \returns \c true if the norm of \c *this is much smaller than the norm of \a other,
  * within the precision determined by \a prec.
  *
  * \note The fuzzy compares are done multiplicatively. A vector \f$ v \f$ is
  * considered to be much smaller than a vector \f$ w \f$ within precision \f$ p \f$ if
  * \f[ \Vert v \Vert \leqslant p\,\Vert w\Vert. \f]
  * For matrices, the comparison is done on all columns.
  *
  * \sa isApprox(), isMuchSmallerThan(const RealScalar&, RealScalar) const
  */
template<typename Derived>
template<typename OtherDerived>
bool MatrixBase<Derived>::isMuchSmallerThan(
  const MatrixBase<OtherDerived>& other,
  typename NumTraits<Scalar>::Real prec
) const
{
  ei_assert(rows() == other.rows() && cols() == other.cols());
  if(IsVectorAtCompileTime)
  {
    return(norm2() <= other.norm2() * prec * prec);
  }
  else
  {
    typename Derived::Nested nested(*this);
    typename OtherDerived::Nested otherNested(other);
    for(int i = 0; i < cols(); i++)
      if(nested.col(i).norm2() > otherNested.col(i).norm2() * prec * prec)
        return false;
    return true;
  }
}

#endif // EIGEN_FUZZY_H
