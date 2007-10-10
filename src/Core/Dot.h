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

#ifndef EI_DOT_H
#define EI_DOT_H

template<int Index, int Size, typename Derived1, typename Derived2>
struct EiDotUnroller
{
  static void run(const Derived1 &v1, const Derived2& v2, typename Derived1::Scalar &dot)
  {
    EiDotUnroller<Index-1, Size, Derived1, Derived2>::run(v1, v2, dot);
    dot += v1[Index] * EiConj(v2[Index]);
  }
};

template<int Size, typename Derived1, typename Derived2>
struct EiDotUnroller<0, Size, Derived1, Derived2>
{
  static void run(const Derived1 &v1, const Derived2& v2, typename Derived1::Scalar &dot)
  {
    dot = v1[0] * EiConj(v2[0]);
  }
};

template<int Index, typename Derived1, typename Derived2>
struct EiDotUnroller<Index, EiDynamic, Derived1, Derived2>
{
  static void run(const Derived1 &v1, const Derived2& v2, typename Derived1::Scalar &dot)
  {
    EI_UNUSED(v1);
    EI_UNUSED(v2);
    EI_UNUSED(dot);
  }
};

template<typename Scalar, typename Derived>
template<typename OtherDerived>
Scalar EiObject<Scalar, Derived>::dot(const OtherDerived& other) const
{
  assert(IsVector && OtherDerived::IsVector && size() == other.size());
  Scalar res;
  if(SizeAtCompileTime != EiDynamic && SizeAtCompileTime <= 16)
    EiDotUnroller<SizeAtCompileTime-1, SizeAtCompileTime, Derived, OtherDerived>
      ::run(*static_cast<const Derived*>(this), other, res);
  else
  {
    res = (*this)[0] * EiConj(other[0]);
    for(int i = 1; i < size(); i++)
      res += (*this)[i]* EiConj(other[i]);
  }
  return res;
}

template<typename Scalar, typename Derived>
typename EiNumTraits<Scalar>::Real EiObject<Scalar, Derived>::norm2() const
{
  assert(IsVector);
  return EiReal(dot(*this));
}

template<typename Scalar, typename Derived>
typename EiNumTraits<Scalar>::Real EiObject<Scalar, Derived>::norm() const
{
  return EiSqrt(norm2());
}

template<typename Scalar, typename Derived>
EiScalarProduct<Derived> EiObject<Scalar, Derived>::normalized() const
{
  return (*this) / norm();
}

#endif // EI_DOT_H
