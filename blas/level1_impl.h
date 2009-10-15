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

#include "common.h"

int EIGEN_BLAS_FUNC(axpy)(int *n, RealScalar *palpha, RealScalar *px, int *incx, RealScalar *py, int *incy)
{
  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar* y = reinterpret_cast<Scalar*>(py);
  Scalar alpha  = *reinterpret_cast<Scalar*>(palpha);

  if(*incx==1 && *incy==1)
    vector(y,*n) += alpha * vector(x,*n);
  else
    vector(y,*n,*incy) += alpha * vector(x,*n,*incx);

  return 1;
}

// computes the sum of magnitudes of all vector elements or, for a complex vector x, the sum
// res = |Rex1| + |Imx1| + |Rex2| + |Imx2| + ... + |Rexn| + |Imxn|, where x is a vector of order n
RealScalar EIGEN_BLAS_FUNC(asum)(int *n, RealScalar *px, int *incx)
{
  int size = IsComplex ? 2* *n : *n;

  if(*incx==1)
    return vector(px,size).cwise().abs().sum();
  else
    return vector(px,size,*incx).cwise().abs().sum();

  return 1;
}

int EIGEN_BLAS_FUNC(copy)(int *n, RealScalar *px, int *incx, RealScalar *py, int *incy)
{
  int size = IsComplex ? 2* *n : *n;

  if(*incx==1 && *incy==1)
    vector(py,size) = vector(px,size);
  else
    vector(py,size,*incy) = vector(px,size,*incx);

  return 1;
}

// computes a vector-vector dot product.
Scalar EIGEN_BLAS_FUNC(dot)(int *n, RealScalar *px, int *incx, RealScalar *py, int *incy)
{
  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar* y = reinterpret_cast<Scalar*>(py);

  if(*incx==1 && *incy==1)
    return (vector(x,*n).cwise()*vector(y,*n)).sum();

  return (vector(x,*n,*incx).cwise()*vector(y,*n,*incy)).sum();
}

/*

// computes a vector-vector dot product with extended precision.
Scalar EIGEN_BLAS_FUNC(sdot)(int *n, RealScalar *px, int *incx, RealScalar *py, int *incy)
{
  // TODO
  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar* y = reinterpret_cast<Scalar*>(py);

  if(*incx==1 && *incy==1)
    return vector(x,*n).dot(vector(y,*n));

  return vector(x,*n,*incx).dot(vector(y,*n,*incy));
}

*/

#if ISCOMPLEX

// computes a dot product of a conjugated vector with another vector.
Scalar EIGEN_BLAS_FUNC(dotc)(int *n, RealScalar *px, int *incx, RealScalar *py, int *incy)
{
  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar* y = reinterpret_cast<Scalar*>(py);

  if(*incx==1 && *incy==1)
    return vector(x,*n).dot(vector(y,*n));

  return vector(x,*n,*incx).dot(vector(y,*n,*incy));
}

// computes a vector-vector dot product without complex conjugation.
Scalar EIGEN_BLAS_FUNC(dotu)(int *n, RealScalar *px, int *incx, RealScalar *py, int *incy)
{
  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar* y = reinterpret_cast<Scalar*>(py);

  if(*incx==1 && *incy==1)
    return (vector(x,*n).cwise()*vector(y,*n)).sum();

  return (vector(x,*n,*incx).cwise()*vector(y,*n,*incy)).sum();
}

#endif // ISCOMPLEX

// computes the Euclidean norm of a vector.
Scalar EIGEN_BLAS_FUNC(nrm2)(int *n, RealScalar *px, int *incx)
{
  Scalar* x = reinterpret_cast<Scalar*>(px);

  if(*incx==1)
    return vector(x,*n).norm();

  return vector(x,*n,*incx).norm();
}

int EIGEN_BLAS_FUNC(rot)(int *n, RealScalar *px, int *incx, RealScalar *py, int *incy, RealScalar *pc, RealScalar *ps)
{
  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar* y = reinterpret_cast<Scalar*>(py);
  Scalar c = *reinterpret_cast<Scalar*>(pc);
  Scalar s = *reinterpret_cast<Scalar*>(ps);

  StridedVectorType vx(vector(x,*n,*incx));
  StridedVectorType vy(vector(y,*n,*incy));
  ei_apply_rotation_in_the_plane(vx, vy, PlanarRotation<Scalar>(c,s));
  return 1;
}

int EIGEN_BLAS_FUNC(rotg)(RealScalar *pa, RealScalar *pb, RealScalar *pc, RealScalar *ps)
{
  Scalar a = *reinterpret_cast<Scalar*>(pa);
  Scalar b = *reinterpret_cast<Scalar*>(pb);
  Scalar* c = reinterpret_cast<Scalar*>(pc);
  Scalar* s = reinterpret_cast<Scalar*>(ps);

  PlanarRotation<Scalar> r;
  r.makeGivens(a,b);
  *c = r.c();
  *s = r.s();

  return 1;
}

#if !ISCOMPLEX
/*
// performs rotation of points in the modified plane.
int EIGEN_BLAS_FUNC(rotm)(int *n, RealScalar *px, int *incx, RealScalar *py, int *incy, RealScalar *param)
{
  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar* y = reinterpret_cast<Scalar*>(py);

  // TODO

  return 0;
}

// computes the modified parameters for a Givens rotation.
int EIGEN_BLAS_FUNC(rotmg)(RealScalar *d1, RealScalar *d2, RealScalar *x1, RealScalar *x2, RealScalar *param)
{
  // TODO

  return 0;
}
*/
#endif // !ISCOMPLEX

int EIGEN_BLAS_FUNC(scal)(int *n, RealScalar *px, int *incx, RealScalar *palpha)
{
  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar alpha = *reinterpret_cast<Scalar*>(palpha);

  if(*incx==1)
    vector(x,*n) *= alpha;

  vector(x,*n,*incx) *= alpha;

  return 1;
}

int EIGEN_BLAS_FUNC(swap)(int *n, RealScalar *px, int *incx, RealScalar *py, int *incy)
{
  int size = IsComplex ? 2* *n : *n;

  if(*incx==1 && *incy==1)
    vector(py,size).swap(vector(px,size));
  else
    vector(py,size,*incy).swap(vector(px,size,*incx));

  return 1;
}

#if !ISCOMPLEX

RealScalar EIGEN_BLAS_FUNC(casum)(int *n, RealScalar *px, int *incx)
{
  Complex* x = reinterpret_cast<Complex*>(px);

  if(*incx==1)
    return vector(x,*n).cwise().abs().sum();
  else
    return vector(x,*n,*incx).cwise().abs().sum();

  return 1;
}

#endif // ISCOMPLEX
