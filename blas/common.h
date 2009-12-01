// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_BLAS_COMMON_H
#define EIGEN_BLAS_COMMON_H

#ifndef SCALAR
#error the token SCALAR must be defined to compile this file
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#include <blas.h>

#ifdef __cplusplus
}
#endif


#define NOTR    0
#define TR      1
#define ADJ     2

#define LEFT    0
#define RIGHT   1

#define UP      0
#define LO      1

#define NUNIT   0
#define UNIT    1

#define OP(X)   (   ((X)=='N' || (X)=='n') ? NOTR   \
                  : ((X)=='T' || (X)=='t') ? TR     \
                  : ((X)=='C' || (X)=='c') ? ADJ    \
                  : 0xff)

#define SIDE(X) (   ((X)=='L' || (X)=='l') ? LEFT   \
                  : ((X)=='R' || (X)=='r') ? RIGHT  \
                  : 0xff)

#define UPLO(X) (   ((X)=='U' || (X)=='u') ? UP     \
                  : ((X)=='L' || (X)=='l') ? LO     \
                  : 0xff)

#define DIAG(X) (   ((X)=='N' || (X)=='N') ? NUNIT  \
                  : ((X)=='U' || (X)=='u') ? UNIT   \
                  : 0xff)

#include <Eigen/Core>
#include <Eigen/Jacobi>
using namespace Eigen;

template<typename T>
Block<Map<Matrix<T,Dynamic,Dynamic> >, Dynamic, Dynamic>
matrix(T* data, int rows, int cols, int stride)
{
  return Map<Matrix<T,Dynamic,Dynamic> >(data, stride, cols).block(0,0,rows,cols);
}

template<typename T>
Block<Map<Matrix<T,Dynamic,Dynamic,RowMajor> >, Dynamic, 1>
vector(T* data, int size, int incr)
{
  return Map<Matrix<T,Dynamic,Dynamic,RowMajor> >(data, size, incr).col(0);
}

template<typename T>
Map<Matrix<T,Dynamic,1> >
vector(T* data, int size)
{
  return Map<Matrix<T,Dynamic,1> >(data, size);
}

typedef SCALAR Scalar;
typedef NumTraits<Scalar>::Real RealScalar;
typedef std::complex<RealScalar> Complex;

enum
{
  IsComplex = Eigen::NumTraits<SCALAR>::IsComplex,
  Conj = IsComplex
};

typedef Block<Map<Matrix<Scalar,Dynamic,Dynamic> >, Dynamic, Dynamic> MatrixType;
typedef Block<Map<Matrix<Scalar,Dynamic,Dynamic, RowMajor> >, Dynamic, 1> StridedVectorType;
typedef Map<Matrix<Scalar,Dynamic,1> > CompactVectorType;

#define EIGEN_BLAS_FUNC(X) EIGEN_CAT(SCALAR_SUFFIX,X##_)

#endif // EIGEN_BLAS_COMMON_H
