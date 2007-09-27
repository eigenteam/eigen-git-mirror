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

#ifndef EIGEN_UTIL_H
#define EIGEN_UTIL_H

#include <iostream>
#include <complex>
#include <cassert>

#undef minor

#define EIGEN_UNUSED(x) (void)x
#define EIGEN_CHECK_RANGES(matrix, row, col) \
  assert(row >= 0 && row < (matrix).rows() && col >= 0 && col < (matrix).cols())
#define EIGEN_CHECK_ROW_RANGE(matrix, row) \
  assert(row >= 0 && row < (matrix).rows())
#define EIGEN_CHECK_COL_RANGE(matrix, col) \
  assert(col >= 0 && col < (matrix).cols())

namespace Eigen
{

//forward declarations
template<typename _Scalar, int _Rows, int _Cols> class Matrix;
template<typename MatrixType> class MatrixAlias;
template<typename MatrixType> class MatrixRef;
template<typename MatrixType> class MatrixRow;
template<typename MatrixType> class MatrixCol;
template<typename MatrixType> class MatrixMinor;
template<typename MatrixType> class MatrixBlock;
template<typename Lhs, typename Rhs> class MatrixSum;
template<typename Lhs, typename Rhs> class MatrixDifference;
template<typename Lhs, typename Rhs> class MatrixProduct;
template<typename MatrixType> class ScalarProduct;

template<typename T> struct ForwardDecl
{
  typedef T Ref;
};

template<typename _Scalar, int _Rows, int _Cols> struct ForwardDecl<Matrix<_Scalar, _Rows, _Cols> >
{
  typedef MatrixRef<Matrix<_Scalar, _Rows, _Cols> > Ref;
};

template<typename MatrixType> struct ForwardDecl<MatrixAlias<MatrixType> >
{
  typedef MatrixRef<MatrixAlias<MatrixType> > Ref;
};

const int DynamicSize = -1;

#define EIGEN_UNUSED(x) (void)x

#define EIGEN_INHERIT_ASSIGNMENT_OPERATOR(Derived, Op) \
template<typename OtherScalar, typename OtherDerived> \
Derived& operator Op(const EigenBase<OtherScalar, OtherDerived>& other) \
{ \
  return EigenBase<OtherScalar, Derived>::operator Op(other); \
} \
Derived& operator Op(const Derived& other) \
{ \
  return EigenBase<Scalar, Derived>::operator Op(other); \
}

#define EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Derived) \
EIGEN_INHERIT_ASSIGNMENT_OPERATOR(Derived, =) \
EIGEN_INHERIT_ASSIGNMENT_OPERATOR(Derived, +=) \
EIGEN_INHERIT_ASSIGNMENT_OPERATOR(Derived, -=)

} // namespace Eigen

#endif // EIGEN_UTIL_H
