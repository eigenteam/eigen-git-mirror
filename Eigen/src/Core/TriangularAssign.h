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

#ifndef EIGEN_TRIANGULAR_ASSIGN_H
#define EIGEN_TRIANGULAR_ASSIGN_H

template<typename Derived1, typename Derived2, int UnrollCount, int Mode>
struct ei_triangular_assign_unroller
{
  enum {
    col = (UnrollCount-1) / Derived1::RowsAtCompileTime,
    row = (UnrollCount-1) % Derived1::RowsAtCompileTime
  };

  static void run(Derived1 &dst, const Derived2 &src)
  {
    ei_triangular_assign_unroller<Derived1, Derived2,
      (Mode & Lower) ?
        ((row==col) ? UnrollCount-1-row : UnrollCount-1)
      : ((row==0)   ? UnrollCount-1-Derived1::ColsAtCompileTime+col : UnrollCount-1),
      Mode>::run(dst, src);
    dst.coeffRef(row, col) = src.coeff(row, col);
  }
};

template<typename Derived1, typename Derived2, int Mode>
struct ei_triangular_assign_unroller<Derived1, Derived2, 1, Mode>
{
  static void run(Derived1 &dst, const Derived2 &src)
  {
    dst.coeffRef(0, 0) = src.coeff(0, 0);
  }
};

// prevent buggy user code from causing an infinite recursion
template<typename Derived1, typename Derived2, int Mode>
struct ei_triangular_assign_unroller<Derived1, Derived2, 0, Mode>
{
  static void run(Derived1 &, const Derived2 &) {}
};

template<typename Derived1, typename Derived2, int Mode>
struct ei_triangular_assign_unroller<Derived1, Derived2, Dynamic, Mode>
{
  static void run(Derived1 &, const Derived2 &) {}
};


template <typename Derived, typename OtherDerived, bool DummyVectorize>
struct ei_assignment_impl<Derived, OtherDerived, DummyVectorize, true>
{
  static void execute(Derived & dst, const OtherDerived & src)
  {
    assert(src.rows()==src.cols());
    assert(dst.rows() == src.rows() && dst.cols() == src.cols());

    const bool unroll = Derived::SizeAtCompileTime * OtherDerived::CoeffReadCost <= EIGEN_UNROLLING_LIMIT;

    if(unroll)
    {
      ei_triangular_assign_unroller
        <Derived, OtherDerived, unroll ? Derived::SizeAtCompileTime : Dynamic, Derived::Flags>::run
          (dst.derived(), src.derived());
    }
    else
    {
      if (Derived::Flags & Lower)
      {
        for(int j = 0; j < dst.cols(); j++)
          for(int i = j; i < dst.rows(); i++)
            dst.coeffRef(i, j) = src.coeff(i, j);
      }
      else
      {
        for(int j = 0; j < dst.cols(); j++)
          for(int i = 0; i <= j; i++)
            dst.coeffRef(i, j) = src.coeff(i, j);
      }
    }
  }
};

#endif // EIGEN_TRIANGULAR_ASSIGN_H
