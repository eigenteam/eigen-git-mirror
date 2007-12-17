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

#ifndef EIGEN_TRACE_H
#define EIGEN_TRACE_H

template<int Index, int Rows, typename Derived> struct TraceUnroller
{
  static void run(const Derived &mat, typename Derived::Scalar &trace)
  {
    TraceUnroller<Index-1, Rows, Derived>::run(mat, trace);
    trace += mat.coeff(Index, Index);
  }
};

template<int Rows, typename Derived> struct TraceUnroller<0, Rows, Derived>
{
  static void run(const Derived &mat, typename Derived::Scalar &trace)
  {
    trace = mat.coeff(0, 0);
  }
};

template<int Index, typename Derived> struct TraceUnroller<Index, Dynamic, Derived>
{
  static void run(const Derived&, typename Derived::Scalar&) {}
};

// prevent buggy user code from causing an infinite recursion
template<int Index, typename Derived> struct TraceUnroller<Index, 0, Derived>
{
  static void run(const Derived&, typename Derived::Scalar&) {}
};

template<typename Scalar, typename Derived>
Scalar MatrixBase<Scalar, Derived>::trace() const
{
  assert(rows() == cols());
  Scalar res;
  if(EIGEN_UNROLLED_LOOPS && RowsAtCompileTime != Dynamic && RowsAtCompileTime <= 16)
    TraceUnroller<RowsAtCompileTime-1, RowsAtCompileTime, Derived>
      ::run(*static_cast<const Derived*>(this), res);
  else
  {
    res = coeff(0, 0);
    for(int i = 1; i < rows(); i++)
      res += coeff(i, i);
  }
  return res;
}

#endif // EIGEN_TRACE_H
