// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2007 Michael Olbrich <michael.olbrich@gmx.net>
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

#ifndef EIGEN_OPERATOREQUALS_H
#define EIGEN_OPERATOREQUALS_H

template<typename Derived1, typename Derived2, int UnrollCount>
struct ei_matrix_operator_equals_unroller
{
  enum {
    col = (UnrollCount-1) / Derived1::RowsAtCompileTime,
    row = (UnrollCount-1) % Derived1::RowsAtCompileTime
  };

  static void run(Derived1 &dst, const Derived2 &src)
  {
    ei_matrix_operator_equals_unroller<Derived1, Derived2, UnrollCount-1>::run(dst, src);
    dst.coeffRef(row, col) = src.coeff(row, col);
  }
};

template<typename Derived1, typename Derived2>
struct ei_matrix_operator_equals_unroller<Derived1, Derived2, 1>
{
  static void run(Derived1 &dst, const Derived2 &src)
  {
    dst.coeffRef(0, 0) = src.coeff(0, 0);
  }
};

// prevent buggy user code from causing an infinite recursion
template<typename Derived1, typename Derived2>
struct ei_matrix_operator_equals_unroller<Derived1, Derived2, 0>
{
  static void run(Derived1 &, const Derived2 &) {}
};

template<typename Derived1, typename Derived2>
struct ei_matrix_operator_equals_unroller<Derived1, Derived2, Dynamic>
{
  static void run(Derived1 &, const Derived2 &) {}
};

template<typename Derived1, typename Derived2, int UnrollCount>
struct ei_vector_operator_equals_unroller
{
  enum { index = UnrollCount - 1 };

  static void run(Derived1 &dst, const Derived2 &src)
  {
    ei_vector_operator_equals_unroller<Derived1, Derived2, UnrollCount-1>::run(dst, src);
    dst.coeffRef(index) = src.coeff(index);
  }
};

// prevent buggy user code from causing an infinite recursion
template<typename Derived1, typename Derived2>
struct ei_vector_operator_equals_unroller<Derived1, Derived2, 0>
{
  static void run(Derived1 &, const Derived2 &) {}
};

template<typename Derived1, typename Derived2>
struct ei_vector_operator_equals_unroller<Derived1, Derived2, 1>
{
  static void run(Derived1 &dst, const Derived2 &src)
  {
    dst.coeffRef(0) = src.coeff(0);
  }
};

template<typename Derived1, typename Derived2>
struct ei_vector_operator_equals_unroller<Derived1, Derived2, Dynamic>
{
  static void run(Derived1 &, const Derived2 &) {}
};

template<typename Derived>
template<typename OtherDerived>
Derived& MatrixBase<Derived>
  ::operator=(const MatrixBase<OtherDerived>& other)
{
  if(IsVectorAtCompileTime && OtherDerived::IsVectorAtCompileTime)
    // copying a vector expression into a vector
  {
    ei_assert(size() == other.size());
    if(EIGEN_UNROLLED_LOOPS
    && SizeAtCompileTime != Dynamic
    && SizeAtCompileTime <= EIGEN_UNROLLING_LIMIT)
      ei_vector_operator_equals_unroller
        <Derived, OtherDerived,
          SizeAtCompileTime <= EIGEN_UNROLLING_LIMIT ? SizeAtCompileTime : Dynamic>::run
          (*static_cast<Derived*>(this), *static_cast<const OtherDerived*>(&other));
    else
      for(int i = 0; i < size(); i++)
        coeffRef(i) = other.coeff(i);
    return *static_cast<Derived*>(this);
  }
  else // copying a matrix expression into a matrix
  {
    ei_assert(rows() == other.rows() && cols() == other.cols());
    if(EIGEN_UNROLLED_LOOPS
    && SizeAtCompileTime != Dynamic
    && SizeAtCompileTime <= EIGEN_UNROLLING_LIMIT)
    {
      ei_matrix_operator_equals_unroller
        <Derived, OtherDerived,
          SizeAtCompileTime <= EIGEN_UNROLLING_LIMIT ? SizeAtCompileTime : Dynamic>::run
          (*static_cast<Derived*>(this), *static_cast<const OtherDerived*>(&other));
    }
    else
    {
      if(ColsAtCompileTime == Dynamic || RowsAtCompileTime != Dynamic)
      {
        // traverse in column-major order
        for(int j = 0; j < cols(); j++)
          for(int i = 0; i < rows(); i++)
            coeffRef(i, j) = other.coeff(i, j);
      }
      else
      {
        // traverse in row-major order
        // in order to allow the compiler to unroll the inner loop
        for(int i = 0; i < rows(); i++)
          for(int j = 0; j < cols(); j++)
            coeffRef(i, j) = other.coeff(i, j);
      }
    }
    return derived();
  }
}

#endif // EIGEN_OPERATOREQUALS_H
