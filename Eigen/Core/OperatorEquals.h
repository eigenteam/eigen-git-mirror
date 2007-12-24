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

#ifndef EIGEN_OPERATOREQUALS_H
#define EIGEN_OPERATOREQUALS_H

template<typename Derived1, typename Derived2, int UnrollCount, int Rows>
struct MatrixOperatorEqualsUnroller
{
  static const int col = (UnrollCount-1) / Rows;
  static const int row = (UnrollCount-1) % Rows;

  static void run(Derived1 &dst, const Derived2 &src)
  {
    MatrixOperatorEqualsUnroller<Derived1, Derived2, UnrollCount-1, Rows>::run(dst, src);
    dst.coeffRef(row, col) = src.coeff(row, col);
  }
};

// prevent buggy user code from causing an infinite recursion
template<typename Derived1, typename Derived2, int UnrollCount>
struct MatrixOperatorEqualsUnroller<Derived1, Derived2, UnrollCount, 0>
{
  static void run(Derived1 &, const Derived2 &) {}
};

template<typename Derived1, typename Derived2, int Rows>
struct MatrixOperatorEqualsUnroller<Derived1, Derived2, 1, Rows>
{
  static void run(Derived1 &dst, const Derived2 &src)
  {
    dst.coeffRef(0, 0) = src.coeff(0, 0);
  }
};

template<typename Derived1, typename Derived2, int Rows>
struct MatrixOperatorEqualsUnroller<Derived1, Derived2, Dynamic, Rows>
{
  static void run(Derived1 &, const Derived2 &) {}
};

template<typename Derived1, typename Derived2, int UnrollCount>
struct VectorOperatorEqualsUnroller
{
  static const int index = UnrollCount - 1;

  static void run(Derived1 &dst, const Derived2 &src)
  {
    VectorOperatorEqualsUnroller<Derived1, Derived2, UnrollCount-1>::run(dst, src);
    dst.coeffRef(index) = src.coeff(index);
  }
};

// prevent buggy user code from causing an infinite recursion
template<typename Derived1, typename Derived2>
struct VectorOperatorEqualsUnroller<Derived1, Derived2, 0>
{
  static void run(Derived1 &, const Derived2 &) {}
};

template<typename Derived1, typename Derived2>
struct VectorOperatorEqualsUnroller<Derived1, Derived2, 1>
{
  static void run(Derived1 &dst, const Derived2 &src)
  {
    dst.coeffRef(0) = src.coeff(0);
  }
};

template<typename Derived1, typename Derived2>
struct VectorOperatorEqualsUnroller<Derived1, Derived2, Dynamic>
{
  static void run(Derived1 &, const Derived2 &) {}
};

template<typename Scalar, typename Derived>
template<typename OtherDerived>
Derived& MatrixBase<Scalar, Derived>
  ::operator=(const MatrixBase<Scalar, OtherDerived>& other)
{
  if(IsVector && OtherDerived::IsVector) // copying a vector expression into a vector
  {
    assert(size() == other.size());
    if(EIGEN_UNROLLED_LOOPS && SizeAtCompileTime != Dynamic && SizeAtCompileTime <= 25)
      VectorOperatorEqualsUnroller
        <Derived, OtherDerived, SizeAtCompileTime>::run
          (*static_cast<Derived*>(this), *static_cast<const OtherDerived*>(&other));
    else
      for(int i = 0; i < size(); i++)
        coeffRef(i) = other.coeff(i);
    return *static_cast<Derived*>(this);
  }
  else // all other cases (typically, but not necessarily, copying a matrix)
  {
    assert(rows() == other.rows() && cols() == other.cols());
    if(EIGEN_UNROLLED_LOOPS && SizeAtCompileTime != Dynamic && SizeAtCompileTime <= 25)
      MatrixOperatorEqualsUnroller
        <Derived, OtherDerived, SizeAtCompileTime, RowsAtCompileTime>::run
          (*static_cast<Derived*>(this), *static_cast<const OtherDerived*>(&other));
    else
      for(int j = 0; j < cols(); j++) //traverse in column-dominant order
        for(int i = 0; i < rows(); i++)
          coeffRef(i, j) = other.coeff(i, j);
    return *static_cast<Derived*>(this);
  }
}

#endif // EIGEN_OPERATOREQUALS_H
