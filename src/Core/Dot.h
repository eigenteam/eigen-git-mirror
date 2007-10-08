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
    const int i = Index - 1;
    if(i == Size - 1)
      dot = v1[i] * EiConj(v2[i]);
    else
      dot += v1[i] * EiConj(v2[i]);
    EiDotUnroller<Index-1, Size, Derived1, Derived2>::run(v1, v2, dot);
  }
};

template<int Size, typename Derived1, typename Derived2>
struct EiDotUnroller<0, Size, Derived1, Derived2>
{
  static void run(const Derived1 &v1, const Derived2& v2, typename Derived1::Scalar &dot)
  {
    EI_UNUSED(v1);
    EI_UNUSED(v2);
    EI_UNUSED(dot);
  }
};

template<int Size, typename Derived1, typename Derived2>
struct EiDotUnroller<EiDynamic, Size, Derived1, Derived2>
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
    EiDotUnroller<SizeAtCompileTime, SizeAtCompileTime, Derived, OtherDerived>
      ::run(*static_cast<const Derived*>(this), other, res);
  else
  {
    res = (*this)[0] * EiConj(other[0]);
    for(int i = 1; i < size(); i++)
      res += (*this)[i]* EiConj(other[i]);
  }
  return res;
}

#endif // EI_DOT_H
