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

/** \returns the trace of *this, which must be a square matrix.
  *
  * \sa diagonal() */
template<typename Scalar, typename Derived>
Scalar MatrixBase<Scalar, Derived>::trace() const
{
  assert(rows() == cols());
  Scalar res;
  if(EIGEN_UNROLLED_LOOPS
  && Traits::RowsAtCompileTime != Dynamic
  && Traits::RowsAtCompileTime <= EIGEN_UNROLLING_LIMIT_PRODUCT)
    TraceUnroller<Traits::RowsAtCompileTime-1,
      Traits::RowsAtCompileTime <= EIGEN_UNROLLING_LIMIT_PRODUCT ? Traits::RowsAtCompileTime : Dynamic, Derived>
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
