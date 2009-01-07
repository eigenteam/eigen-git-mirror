// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_SPARSEREDUX_H
#define EIGEN_SPARSEREDUX_H


template<typename Derived, int Vectorization, int Unrolling>
struct ei_sum_impl<Derived, Vectorization, Unrolling, IsSparse>
{
  typedef typename Derived::Scalar Scalar;
  static Scalar run(const Derived& mat)
  {
    ei_assert(mat.rows()>0 && mat.cols()>0 && "you are using a non initialized matrix");
    Scalar res = 0;
    for (int j=0; j<mat.outerSize(); ++j)
      for (typename Derived::InnerIterator iter(mat,j); iter; ++iter)
        res += iter.value();
    return res;
  }
};

template<typename Derived1, typename Derived2, int Vectorization, int Unrolling>
struct ei_dot_impl<Derived1, Derived2, Vectorization, Unrolling, IsSparse>
{
  typedef typename Derived1::Scalar Scalar;
  static Scalar run(const Derived1& v1, const Derived2& v2)
  {
    ei_assert(v1.size()>0 && "you are using a non initialized vector");
    typename Derived1::InnerIterator i(v1,0);
    typename Derived2::InnerIterator j(v2,0);
    Scalar res = 0;
    while (i && j)
    {
      if (i.index()==j.index())
      {
        res += i.value() * ei_conj(j.value());
        ++i; ++j;
      }
      else if (i.index()<j.index())
        ++i;
      else
        ++j;
    }
    return res;
  }
};

#endif // EIGEN_SPARSEREDUX_H
