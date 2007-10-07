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

#ifndef EI_TRACE_H
#define EI_TRACE_H

template<int CurrentRow, int Rows, typename Derived> struct EiTraceUnroller
{
  typedef typename Derived::Scalar Scalar;

  static void run(const Derived &mat, Scalar &trace)
  {
    if(CurrentRow == Rows - 1)
      trace = mat(CurrentRow, CurrentRow);
    else
      trace += mat(CurrentRow, CurrentRow);
    EiTraceUnroller<CurrentRow-1, Rows, Derived>::run(mat, trace);
  }
};

template<int Rows, typename Derived> struct EiTraceUnroller<-1, Rows, Derived>
{
  typedef typename Derived::Scalar Scalar;

  static void run(const Derived &mat, Scalar &trace)
  {
    EI_UNUSED(mat);
    EI_UNUSED(trace);
  }
};

template<typename Scalar, typename Derived>
Scalar EiObject<Scalar, Derived>::trace() const
{
  assert(rows() == cols());
  Scalar res;
  if(RowsAtCompileTime != EiDynamic && RowsAtCompileTime <= 16)
    EiTraceUnroller<RowsAtCompileTime - 1, RowsAtCompileTime, Derived>
      ::run(*static_cast<const Derived*>(this), res);
  else
  {
    res = read(0, 0);
    for(int i = 1; i < rows(); i++)
      res += read(i, i);
  }
  return res;
}

#endif // EI_TRACE_H
