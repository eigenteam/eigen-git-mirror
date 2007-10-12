// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2007 Michael Olbrich <michael.olbrich@gmx.net>
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

#ifndef EI_COPYHELPER_H
#define EI_COPYHELPER_H

template<int UnrollCount, int Rows> struct CopyHelperUnroller
{
  static const int col = (UnrollCount-1) / Rows;
  static const int row = (UnrollCount-1) % Rows;

  template <typename Derived1, typename Derived2>
  static void run(Derived1 &dst, const Derived2 &src)
  {
    CopyHelperUnroller<UnrollCount-1, Rows>::run(dst, src);
    dst.write(row, col) = src.read(row, col);
  }
};

template<int Rows> struct CopyHelperUnroller<0, Rows>
{
  template <typename Derived1, typename Derived2>
  static void run(Derived1 &dst, const Derived2 &src)
  {
    dst.write(0, 0) = src.read(0, 0);
  }
};

template<int Rows> struct CopyHelperUnroller<Dynamic, Rows>
{
  template <typename Derived1, typename Derived2>
  static void run(Derived1 &dst, const Derived2 &src)
  {
    EI_UNUSED(dst);
    EI_UNUSED(src);
  }
};

template<typename Scalar, typename Derived>
template<typename OtherDerived>
void Object<Scalar, Derived>::_copy_helper(const Object<Scalar, OtherDerived>& other)
{
  if(SizeAtCompileTime != Dynamic && SizeAtCompileTime <= EI_LOOP_UNROLLING_LIMIT)
    CopyHelperUnroller<SizeAtCompileTime, RowsAtCompileTime>::run(*this, other);
  else
    for(int i = 0; i < rows(); i++)
      for(int j = 0; j < cols(); j++)
        write(i, j) = other.read(i, j);
}

#endif // EI_COPYHELPER_H
