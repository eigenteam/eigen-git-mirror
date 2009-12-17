// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2009 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_ANYMATRIXBASE_H
#define EIGEN_ANYMATRIXBASE_H


/** Common base class for all classes T such that MatrixBase has an operator=(T) and a constructor MatrixBase(T).
  *
  * In other words, an AnyMatrixBase object is an object that can be copied into a MatrixBase.
  *
  * Besides MatrixBase-derived classes, this also includes special matrix classes such as diagonal matrices, etc.
  *
  * Notice that this class is trivial, it is only used to disambiguate overloaded functions.
  */
template<typename Derived> struct AnyMatrixBase
{
//   typedef typename ei_plain_matrix_type<Derived>::type PlainMatrixType;

  /** \returns a reference to the derived object */
  Derived& derived() { return *static_cast<Derived*>(this); }
  /** \returns a const reference to the derived object */
  const Derived& derived() const { return *static_cast<const Derived*>(this); }

  inline Derived& const_cast_derived() const
  { return *static_cast<Derived*>(const_cast<AnyMatrixBase*>(this)); }

  /** \returns the number of rows. \sa cols(), RowsAtCompileTime */
  inline int rows() const { return derived().rows(); }
  /** \returns the number of columns. \sa rows(), ColsAtCompileTime*/
  inline int cols() const { return derived().cols(); }

  /** \internal Don't use it, but do the equivalent: \code dst = *this; \endcode */
  template<typename Dest> inline void evalTo(Dest& dst) const
  { derived().evalTo(dst); }

  /** \internal Don't use it, but do the equivalent: \code dst += *this; \endcode */
  template<typename Dest> inline void addToDense(Dest& dst) const
  {
    // This is the default implementation,
    // derived class can reimplement it in a more optimized way.
    typename Dest::PlainMatrixType res(rows(),cols());
    evalTo(res);
    dst += res;
  }

  /** \internal Don't use it, but do the equivalent: \code dst -= *this; \endcode */
  template<typename Dest> inline void subToDense(Dest& dst) const
  {
    // This is the default implementation,
    // derived class can reimplement it in a more optimized way.
    typename Dest::PlainMatrixType res(rows(),cols());
    evalTo(res);
    dst -= res;
  }

  /** \internal Don't use it, but do the equivalent: \code dst.applyOnTheRight(*this); \endcode */
  template<typename Dest> inline void applyThisOnTheRight(Dest& dst) const
  {
    // This is the default implementation,
    // derived class can reimplement it in a more optimized way.
    dst = dst * this->derived();
  }

  /** \internal Don't use it, but do the equivalent: \code dst.applyOnTheLeft(*this); \endcode */
  template<typename Dest> inline void applyThisOnTheLeft(Dest& dst) const
  {
    // This is the default implementation,
    // derived class can reimplement it in a more optimized way.
    dst = this->derived() * dst;
  }

};

/***************************************************************************
* Implementation of matrix base methods
***************************************************************************/

/** Copies the generic expression \a other into *this. \returns a reference to *this.
  * The expression must provide a (templated) evalTo(Derived& dst) const function
  * which does the actual job. In practice, this allows any user to write its own
  * special matrix without having to modify MatrixBase */
template<typename Derived>
template<typename OtherDerived>
Derived& DenseBase<Derived>::operator=(const AnyMatrixBase<OtherDerived> &other)
{
  other.derived().evalTo(derived());
  return derived();
}

template<typename Derived>
template<typename OtherDerived>
Derived& DenseBase<Derived>::operator+=(const AnyMatrixBase<OtherDerived> &other)
{
  other.derived().addToDense(derived());
  return derived();
}

template<typename Derived>
template<typename OtherDerived>
Derived& DenseBase<Derived>::operator-=(const AnyMatrixBase<OtherDerived> &other)
{
  other.derived().subToDense(derived());
  return derived();
}

/** replaces \c *this by \c *this * \a other.
  *
  * \returns a reference to \c *this
  */
template<typename Derived>
template<typename OtherDerived>
inline Derived&
MatrixBase<Derived>::operator*=(const AnyMatrixBase<OtherDerived> &other)
{
  other.derived().applyThisOnTheRight(derived());
  return derived();
}

/** replaces \c *this by \c *this * \a other. It is equivalent to MatrixBase::operator*=() */
template<typename Derived>
template<typename OtherDerived>
inline void MatrixBase<Derived>::applyOnTheRight(const AnyMatrixBase<OtherDerived> &other)
{
  other.derived().applyThisOnTheRight(derived());
}

/** replaces \c *this by \c *this * \a other. */
template<typename Derived>
template<typename OtherDerived>
inline void MatrixBase<Derived>::applyOnTheLeft(const AnyMatrixBase<OtherDerived> &other)
{
  other.derived().applyThisOnTheLeft(derived());
}

#endif // EIGEN_ANYMATRIXBASE_H
