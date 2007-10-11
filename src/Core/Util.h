// This file is part of gen, a lightweight C++ template library
// for linear algebra. gen itself is part of the KDE project.
//
// Copyright (C) 2006-2007 Benoit Jacob <jacob@math.jussieu.fr>
//
// gen is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation; either version 2 or (at your option) any later version.
//
// gen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along
// with gen; if not, write to the Free Software Foundation, Inc., 51
// Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
//
// As a special exception, if other files instantiate templates or use macros
// or functions from this file, or you compile this file and link it
// with other works to produce a work based on this file, this file does not
// by itself cause the resulting work to be covered by the GNU General Public
// License. This exception does not invalidate any other reasons why a work
// based on this file might be covered by the GNU General Public License.

#ifndef EI_UTIL_H
#define EI_UTIL_H

#undef minor

#define USING_EIGEN_DATA_TYPES \
EI_USING_MATRIX_TYPEDEFS \
using Eigen::Matrix;

#define EI_UNUSED(x) (void)x
#define EI_CHECK_RANGES(matrix, row, col) \
  assert(row >= 0 && row < (matrix).rows() && col >= 0 && col < (matrix).cols())
#define EI_CHECK_ROW_RANGE(matrix, row) \
  assert(row >= 0 && row < (matrix).rows())
#define EI_CHECK_COL_RANGE(matrix, col) \
  assert(col >= 0 && col < (matrix).cols())

//forward declarations
template<typename _Scalar, int _Rows, int _Cols> class Matrix;
template<typename MatrixType> class MatrixRef;
template<typename MatrixType> class MatrixConstRef;
template<typename MatrixType> class Row;
template<typename MatrixType> class Column;
template<typename MatrixType> class Minor;
template<typename MatrixType> class Block;
template<typename MatrixType> class Transpose;
template<typename MatrixType> class Conjugate;
template<typename Lhs, typename Rhs> class Sum;
template<typename Lhs, typename Rhs> class Difference;
template<typename Lhs, typename Rhs> class MatrixProduct;
template<typename MatrixType> class ScalarProduct;
template<typename MatrixType> class Random;
template<typename ExpressionType> class Eval;

template<typename T> struct ForwardDecl
{
  typedef T Ref;
  typedef T ConstRef;
};

template<typename _Scalar, int _Rows, int _Cols>
struct ForwardDecl<Matrix<_Scalar, _Rows, _Cols> >
{
  typedef MatrixRef<Matrix<_Scalar, _Rows, _Cols> > Ref;
  typedef MatrixConstRef<Matrix<_Scalar, _Rows, _Cols> > ConstRef;
};

const int Dynamic = -1;

#define EI_LOOP_UNROLLING_LIMIT 25

#define EI_UNUSED(x) (void)x

#ifdef __GNUC__
# define EI_ALWAYS_INLINE __attribute__((always_inline))
# define EI_RESTRICT      /*__restrict__*/
#else
# define EI_ALWAYS_INLINE
# define EI_RESTRICT
#endif

#define EI_INHERIT_ASSIGNMENT_OPERATOR(Derived, Op) \
template<typename OtherScalar, typename OtherDerived> \
Derived& operator Op(const Object<OtherScalar, OtherDerived>& other) \
{ \
  return Object<Scalar, Derived>::operator Op(other); \
} \
Derived& operator Op(const Derived& other) \
{ \
  return Object<Scalar, Derived>::operator Op(other); \
}

#define EI_INHERIT_SCALAR_ASSIGNMENT_OPERATOR(Derived, Op) \
template<typename Other> \
Derived& operator Op(const Other& scalar) \
{ \
  return Object<Scalar, Derived>::operator Op(scalar); \
}

#define EI_INHERIT_ASSIGNMENT_OPERATORS(Derived) \
EI_INHERIT_ASSIGNMENT_OPERATOR(Derived, =) \
EI_INHERIT_ASSIGNMENT_OPERATOR(Derived, +=) \
EI_INHERIT_ASSIGNMENT_OPERATOR(Derived, -=) \
EI_INHERIT_SCALAR_ASSIGNMENT_OPERATOR(Derived, *=) \
EI_INHERIT_SCALAR_ASSIGNMENT_OPERATOR(Derived, /=)

#endif // EI_UTIL_H
