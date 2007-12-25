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

#ifndef EIGEN_DOT_H
#define EIGEN_DOT_H

template<int Index, int Size, typename Derived1, typename Derived2>
struct DotUnroller
{
  static void run(const Derived1 &v1, const Derived2& v2, typename Derived1::Scalar &dot)
  {
    DotUnroller<Index-1, Size, Derived1, Derived2>::run(v1, v2, dot);
    dot += v1.coeff(Index) * conj(v2.coeff(Index));
  }
};

template<int Size, typename Derived1, typename Derived2>
struct DotUnroller<0, Size, Derived1, Derived2>
{
  static void run(const Derived1 &v1, const Derived2& v2, typename Derived1::Scalar &dot)
  {
    dot = v1.coeff(0) * conj(v2.coeff(0));
  }
};

template<int Index, typename Derived1, typename Derived2>
struct DotUnroller<Index, Dynamic, Derived1, Derived2>
{
  static void run(const Derived1&, const Derived2&, typename Derived1::Scalar&) {}
};

// prevent buggy user code from causing an infinite recursion
template<int Index, typename Derived1, typename Derived2>
struct DotUnroller<Index, 0, Derived1, Derived2>
{
  static void run(const Derived1&, const Derived2&, typename Derived1::Scalar&) {}
};

template<typename Scalar, typename Derived>
template<typename OtherDerived>
Scalar MatrixBase<Scalar, Derived>::dot(const OtherDerived& other) const
{
  assert(IsVectorAtCompileTime && OtherDerived::IsVectorAtCompileTime && size() == other.size());
  Scalar res;
  if(EIGEN_UNROLLED_LOOPS && SizeAtCompileTime != Dynamic && SizeAtCompileTime <= 16)
    DotUnroller<SizeAtCompileTime-1, SizeAtCompileTime, Derived, OtherDerived>
      ::run(*static_cast<const Derived*>(this), other, res);
  else
  {
    res = (*this).coeff(0) * conj(other.coeff(0));
    for(int i = 1; i < size(); i++)
      res += (*this).coeff(i)* conj(other.coeff(i));
  }
  return res;
}

template<typename Scalar, typename Derived>
typename NumTraits<Scalar>::Real MatrixBase<Scalar, Derived>::norm2() const
{
  return real(dot(*this));
}

template<typename Scalar, typename Derived>
typename NumTraits<Scalar>::Real MatrixBase<Scalar, Derived>::norm() const
{
  return sqrt(norm2());
}

template<typename Scalar, typename Derived>
ScalarMultiple<Derived> MatrixBase<Scalar, Derived>::normalized() const
{
  return (*this) / norm();
}

#endif // EIGEN_DOT_H
