// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
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

#ifndef EIGEN_RETURNBYVALUE_H
#define EIGEN_RETURNBYVALUE_H

/** \class ReturnByValue
  *
  */
template<typename Derived>
struct ei_traits<ReturnByValue<Derived> >
  : public ei_traits<typename ei_traits<Derived>::ReturnMatrixType>
{
  enum {
    // FIXME had to remove the DirectAccessBit for usage like
    //   matrix.inverse().block(...)
    // because the Block ctor with direct access
    // wants to call coeffRef() to get an address, and that fails (infinite recursion) as ReturnByValue
    // doesnt implement coeffRef(). The better fix is probably rather to make Block work directly
    // on the nested type, right?
    Flags = (ei_traits<typename ei_traits<Derived>::ReturnMatrixType>::Flags
             | EvalBeforeNestingBit) & ~DirectAccessBit
  };
};

/* The ReturnByValue object doesn't even have a coeff() method.
 * So the only way that nesting it in an expression can work, is by evaluating it into a plain matrix.
 * So ei_nested always gives the plain return matrix type.
 */
template<typename Derived,int n,typename PlainMatrixType>
struct ei_nested<ReturnByValue<Derived>, n, PlainMatrixType>
{
  typedef typename ei_traits<Derived>::ReturnMatrixType type;
};

template<typename Derived>
  class ReturnByValue : public MatrixBase<ReturnByValue<Derived> >
{
  public:
    EIGEN_GENERIC_PUBLIC_INTERFACE(ReturnByValue)
    typedef typename ei_traits<Derived>::ReturnMatrixType ReturnMatrixType;
    template<typename Dest>
    inline void evalTo(Dest& dst) const
    { static_cast<const Derived* const>(this)->evalTo(dst); }
    inline int rows() const { return static_cast<const Derived* const>(this)->rows(); }
    inline int cols() const { return static_cast<const Derived* const>(this)->cols(); }
};

template<typename Derived>
template<typename OtherDerived>
Derived& MatrixBase<Derived>::operator=(const ReturnByValue<OtherDerived>& other)
{
  other.evalTo(derived());
  return derived();
}

#endif // EIGEN_RETURNBYVALUE_H
