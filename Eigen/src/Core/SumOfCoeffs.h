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

#ifndef EIGEN_SUMOFCOEFFS_H
#define EIGEN_SUMOFCOEFFS_H

template<int Index, int Size, typename Derived>
struct ei_sumofcoeffs_unroller
{
  static void run(const Derived &v1, typename Derived::Scalar &dot)
  {
    ei_sumofcoeffs_unroller<Index-1, Size, Derived>::run(v1, dot);
    dot += v1.coeff(Index);
  }
};

template<int Size, typename Derived>
struct ei_sumofcoeffs_unroller<0, Size, Derived>
{
  static void run(const Derived &v1, typename Derived::Scalar &dot)
  {
    dot = v1.coeff(0);
  }
};

template<int Index, typename Derived>
struct ei_sumofcoeffs_unroller<Index, Dynamic, Derived>
{
  static void run(const Derived&, typename Derived::Scalar&) {}
};

// prevent buggy user code from causing an infinite recursion
template<int Index, typename Derived>
struct ei_sumofcoeffs_unroller<Index, 0, Derived>
{
  static void run(const Derived&, typename Derived::Scalar&)
{}
};

/** \returns the sum of all coefficients of *this
  *
  * \only_for_vectors
  *
  * \sa trace()
  */
template<typename Derived>
typename ei_traits<Derived>::Scalar
MatrixBase<Derived>::sum() const
{
  assert(IsVectorAtCompileTime);
  Scalar res;
  if(EIGEN_UNROLLED_LOOPS
  && SizeAtCompileTime != Dynamic
  && SizeAtCompileTime <= EIGEN_UNROLLING_LIMIT)
    ei_sumofcoeffs_unroller<SizeAtCompileTime-1,
                SizeAtCompileTime <= EIGEN_UNROLLING_LIMIT ? SizeAtCompileTime : Dynamic,
                Derived>
      ::run(derived(),res);
  else
  {
    res = coeff(0);
    for(int i = 1; i < size(); i++)
      res += coeff(i);
  }
  return res;
}

/** \returns the trace of \c *this, i.e. the sum of the coefficients on the main diagonal.
  *
  * \c *this can be any matrix, not necessarily square.
  *
  * \sa diagonal(), sum()
  */
template<typename Derived>
typename ei_traits<Derived>::Scalar
MatrixBase<Derived>::trace() const
{
  return diagonal().sum();
}

#endif // EIGEN_SUMOFCOEFFS_H
