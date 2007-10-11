// This file is part of gen, a lightweight C++ template library
// for linear algebra. gen itself is part of the KDE project.
//
// Copyright (C) 2006-2007 Benoit Jacob <jacob@math.jussieu.fr>
//
// gen is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation; either version 2 or (at your option) any later version.
//
// gen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along
// with gen; if not, write to the Free Software Foundation, Inc., 51
// Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
//
// As a special exception, if other files instantiate templates or use macros
// or functions from this file, or you compile this file and link it
// with other works to produce a work based on this file, this file does not
// by itself cause the resulting work to be covered by the GNU General Public
// License. This exception does not invalidate any other reasons why a work
// based on this file might be covered by the GNU General Public License.

#ifndef EI_FUZZY_H
#define EI_FUZZY_H

template<typename Scalar, typename Derived>
template<typename OtherDerived>
bool Object<Scalar, Derived>::isApprox(const OtherDerived& other) const
{
  if(IsVector)
  {
    return((*this - other).norm2()
           <= std::min(norm2(), other.norm2())
            * Abs2(NumTraits<Scalar>::epsilon()));
  }
  else
  {
    for(int i = 0; i < cols(); i++)
      if(!col(i).isApprox(other.col(i)))
        return false;
    return true;
  }
}

template<typename Scalar, typename Derived>
bool Object<Scalar, Derived>::isNegligble(const Scalar& other) const
{
  if(IsVector)
  {
    return(norm2() <= Abs2(other) * Abs2(NumTraits<Scalar>::epsilon()));
  }
  else
  {
    for(int i = 0; i < cols(); i++)
      if(!col(i).isNegligible(other))
        return false;
    return true;
  }
}

template<typename Scalar, typename Derived>
template<typename OtherDerived>
bool Object<Scalar, Derived>::isNegligble(const Object<Scalar, OtherDerived>& other) const
{
  if(IsVector)
  {
    return(norm2() <= other.norm2() * Abs2(NumTraits<Scalar>::epsilon()));
  }
  else
  {
    for(int i = 0; i < cols(); i++)
      if(!col(i).isNegligible(other.col(i)))
        return false;
    return true;
  }
}

#endif // EI_FUZZY_H