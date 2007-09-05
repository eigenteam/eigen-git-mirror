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

#include<iostream>
#include<cassert>

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
template<typename T, int Rows, int Cols> class Matrix;
template<typename T> class MatrixX;
template<typename T, int Size> class Vector;
template<typename T> class VectorX;
template<typename Derived> class MatrixBase;

template<typename T> struct ForwardDecl;
template<typename T, int Rows, int Cols> struct ForwardDecl< Matrix<T, Rows, Cols> >
{ typedef T Scalar; };
template<typename T> struct ForwardDecl< MatrixX<T> >
{ typedef T Scalar; };
template<typename T, int Size> struct ForwardDecl< Vector<T, Size> >
{ typedef T Scalar; };
template<typename T> struct ForwardDecl< VectorX<T> >
{ typedef T Scalar; };
template<typename T, int Rows, int Cols> struct ForwardDecl< MatrixBase<Matrix<T, Rows, Cols> > >
{ typedef T Scalar; };
template<typename T> struct ForwardDecl< MatrixBase<MatrixX<T> > >
{ typedef T Scalar; };
template<typename T, int Size> struct ForwardDecl< MatrixBase<Vector<T, Size> > >
{ typedef T Scalar; };
template<typename T> struct ForwardDecl< MatrixBase<VectorX<T> > >
{ typedef T Scalar; };

template<typename MatrixType> class MatrixRef;

} // namespace Eigen

#endif // EIGEN_UTIL_H
