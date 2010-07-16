// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
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

  if(*incx==1 && *incy==1)    vector(y,*n) += alpha * vector(x,*n);
  else if(*incx>0 && *incy>0) vector(y,*n,*incy) += alpha * vector(x,*n,*incx);
  else if(*incx>0 && *incy<0) vector(y,*n,-*incy).reverse() += alpha * vector(x,*n,*incx);
  else if(*incx<0 && *incy>0) vector(y,*n,*incy) += alpha * vector(x,*n,-*incx).reverse();
  else if(*incx<0 && *incy<0) vector(y,*n,-*incy).reverse() += alpha * vector(x,*n,-*incx).reverse();

  return 0;
}

#if !ISCOMPLEX
// computes the sum of magnitudes of all vector elements or, for a complex vector x, the sum
// res = |Rex1| + |Imx1| + |Rex2| + |Imx2| + ... + |Rexn| + |Imxn|, where x is a vector of order n
RealScalar EIGEN_BLAS_FUNC(asum)(int *n, RealScalar *px, int *incx)
{
//   std::cerr << "_asum " << *n << " " << *incx << "\n";

  Scalar* x = reinterpret_cast<Scalar*>(px);

  if(*n<=0) return 0;

  if(*incx==1)  return vector(x,*n).cwiseAbs().sum();
  else          return vector(x,*n,std::abs(*incx)).cwiseAbs().sum();
}
#else

struct ei_scalar_norm1_op {
  typedef RealScalar result_type;
  EIGEN_EMPTY_STRUCT_CTOR(ei_scalar_norm1_op)
  inline RealScalar operator() (const Scalar& a) const { return ei_norm1(a); }
};
namespace Eigen {
template<> struct ei_functor_traits<ei_scalar_norm1_op >
{
  enum { Cost = 3 * NumTraits<Scalar>::AddCost, PacketAccess = 0 };
};
}

RealScalar EIGEN_CAT(EIGEN_CAT(REAL_SCALAR_SUFFIX,SCALAR_SUFFIX),asum_)(int *n, RealScalar *px, int *incx)
{
//   std::cerr << "__asum " << *n << " " << *incx << "\n";

  Complex* x = reinterpret_cast<Complex*>(px);

  if(*n<=0) return 0;

  if(*incx==1)  return vector(x,*n).unaryExpr<ei_scalar_norm1_op>().sum();
  else          return vector(x,*n,std::abs(*incx)).unaryExpr<ei_scalar_norm1_op>().sum();
}
#endif

int EIGEN_BLAS_FUNC(copy)(int *n, RealScalar *px, int *incx, RealScalar *py, int *incy)
{
//   std::cerr << "_copy " << *n << " " << *incx << " " << *incy << "\n";

  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar* y = reinterpret_cast<Scalar*>(py);

  if(*incx==1 && *incy==1)    vector(y,*n) = vector(x,*n);
  else if(*incx>0 && *incy>0) vector(y,*n,*incy) = vector(x,*n,*incx);
  else if(*incx>0 && *incy<0) vector(y,*n,-*incy).reverse() = vector(x,*n,*incx);
  else if(*incx<0 && *incy>0) vector(y,*n,*incy) = vector(x,*n,-*incx).reverse();
  else if(*incx<0 && *incy<0) vector(y,*n,-*incy).reverse() = vector(x,*n,-*incx).reverse();

  return 0;
}

// computes a vector-vector dot product.
Scalar EIGEN_BLAS_FUNC(dot)(int *n, RealScalar *px, int *incx, RealScalar *py, int *incy)
{
//   std::cerr << "_dot " << *n << " " << *incx << " " << *incy << "\n";

  if(*n<=0)
    return 0;

  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar* y = reinterpret_cast<Scalar*>(py);

  if(*incx==1 && *incy==1)    return (vector(x,*n).cwiseProduct(vector(y,*n))).sum();
  else if(*incx>0 && *incy>0) return (vector(x,*n,*incx).cwiseProduct(vector(y,*n,*incy))).sum();
  else if(*incx<0 && *incy>0) return (vector(x,*n,-*incx).reverse().cwiseProduct(vector(y,*n,*incy))).sum();
  else if(*incx>0 && *incy<0) return (vector(x,*n,*incx).cwiseProduct(vector(y,*n,-*incy).reverse())).sum();
  else if(*incx<0 && *incy<0) return (vector(x,*n,-*incx).reverse().cwiseProduct(vector(y,*n,-*incy).reverse())).sum();
  else return 0;
}

int EIGEN_CAT(EIGEN_CAT(i,SCALAR_SUFFIX),amax_)(int *n, RealScalar *px, int *incx)
{
//   std::cerr << "i_amax " << *n << " " << *incx << "\n";

  Scalar* x = reinterpret_cast<Scalar*>(px);

  if(*n<=0)
    return 0;

  DenseIndex ret;

  if(*incx==1)  vector(x,*n).cwiseAbs().maxCoeff(&ret);
  else          vector(x,*n,std::abs(*incx)).cwiseAbs().maxCoeff(&ret);

  return ret+1;
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
void EIGEN_BLAS_FUNC(dotc)(RealScalar* dot, int *n, RealScalar *px, int *incx, RealScalar *py, int *incy)
{

  std::cerr << "Eigen BLAS: _dotc is not implemented yet\n";

  return;

  // TODO: find how to return a complex to fortran

//   std::cerr << "_dotc " << *n << " " << *incx << " " << *incy << "\n";

  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar* y = reinterpret_cast<Scalar*>(py);

  if(*incx==1 && *incy==1)
    *reinterpret_cast<Scalar*>(dot) = vector(x,*n).dot(vector(y,*n));
  else
    *reinterpret_cast<Scalar*>(dot) = vector(x,*n,*incx).dot(vector(y,*n,*incy));
}

// computes a vector-vector dot product without complex conjugation.
void EIGEN_BLAS_FUNC(dotu)(RealScalar* dot, int *n, RealScalar *px, int *incx, RealScalar *py, int *incy)
{
  std::cerr << "Eigen BLAS: _dotu is not implemented yet\n";

  return;

  // TODO: find how to return a complex to fortran

//   std::cerr << "_dotu " << *n << " " << *incx << " " << *incy << "\n";

  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar* y = reinterpret_cast<Scalar*>(py);

  if(*incx==1 && *incy==1)
    *reinterpret_cast<Scalar*>(dot) = (vector(x,*n).cwiseProduct(vector(y,*n))).sum();
  else
    *reinterpret_cast<Scalar*>(dot) = (vector(x,*n,*incx).cwiseProduct(vector(y,*n,*incy))).sum();
}

#endif // ISCOMPLEX

#if !ISCOMPLEX
// computes the Euclidean norm of a vector.
Scalar EIGEN_BLAS_FUNC(nrm2)(int *n, RealScalar *px, int *incx)
{
//   std::cerr << "_nrm2 " << *n << " " << *incx << "\n";
  Scalar* x = reinterpret_cast<Scalar*>(px);

  if(*n<=0)
    return 0;

  if(*incx==1)  return vector(x,*n).norm();
  else          return vector(x,*n,std::abs(*incx)).norm();
}
#else
RealScalar EIGEN_CAT(EIGEN_CAT(REAL_SCALAR_SUFFIX,SCALAR_SUFFIX),nrm2_)(int *n, RealScalar *px, int *incx)
{
//   std::cerr << "__nrm2 " << *n << " " << *incx << "\n";
  Scalar* x = reinterpret_cast<Scalar*>(px);

  if(*n<=0)
    return 0;

  if(*incx==1)
    return vector(x,*n).norm();

  return vector(x,*n,*incx).norm();
}
#endif

int EIGEN_BLAS_FUNC(rot)(int *n, RealScalar *px, int *incx, RealScalar *py, int *incy, RealScalar *pc, RealScalar *ps)
{
//   std::cerr << "_rot " << *n << " " << *incx << " " << *incy << "\n";
  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar* y = reinterpret_cast<Scalar*>(py);
  Scalar c = *reinterpret_cast<Scalar*>(pc);
  Scalar s = *reinterpret_cast<Scalar*>(ps);

  if(*n<=0)
    return 0;

  StridedVectorType vx(vector(x,*n,std::abs(*incx)));
  StridedVectorType vy(vector(y,*n,std::abs(*incy)));

  Reverse<StridedVectorType> rvx(vx);
  Reverse<StridedVectorType> rvy(vy);

       if(*incx<0 && *incy>0) ei_apply_rotation_in_the_plane(rvx, vy, PlanarRotation<Scalar>(c,s));
  else if(*incx>0 && *incy<0) ei_apply_rotation_in_the_plane(vx, rvy, PlanarRotation<Scalar>(c,s));
  else                        ei_apply_rotation_in_the_plane(vx, vy,  PlanarRotation<Scalar>(c,s));


  return 0;
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

  return 0;
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

int EIGEN_BLAS_FUNC(scal)(int *n, RealScalar *palpha, RealScalar *px, int *incx)
{
  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar alpha = *reinterpret_cast<Scalar*>(palpha);

//   std::cerr << "_scal " << *n << " " << alpha << " " << *incx << "\n";

  if(*n<=0)
    return 0;

  if(*incx==1)  vector(x,*n) *= alpha;
  else          vector(x,*n,std::abs(*incx)) *= alpha;

  return 0;
}

#if ISCOMPLEX
int EIGEN_CAT(EIGEN_CAT(SCALAR_SUFFIX,REAL_SCALAR_SUFFIX),scal_)(int *n, RealScalar *palpha, RealScalar *px, int *incx)
{
  Scalar* x = reinterpret_cast<Scalar*>(px);
  RealScalar alpha = *palpha;

//   std::cerr << "__scal " << *n << " " << alpha << " " << *incx << "\n";

  if(*n<=0)
    return 0;

  if(*incx==1)  vector(x,*n) *= alpha;
  else          vector(x,*n,std::abs(*incx)) *= alpha;

  return 0;
}
#endif // ISCOMPLEX

int EIGEN_BLAS_FUNC(swap)(int *n, RealScalar *px, int *incx, RealScalar *py, int *incy)
{
//   std::cerr << "_swap " << *n << " " << *incx << " " << *incy << "\n";

  Scalar* x = reinterpret_cast<Scalar*>(px);
  Scalar* y = reinterpret_cast<Scalar*>(py);

  if(*n<=0)
    return 0;

  if(*incx==1 && *incy==1)    vector(y,*n).swap(vector(x,*n));
  else if(*incx>0 && *incy>0) vector(y,*n,*incy).swap(vector(x,*n,*incx));
  else if(*incx>0 && *incy<0) vector(y,*n,-*incy).reverse().swap(vector(x,*n,*incx));
  else if(*incx<0 && *incy>0) vector(y,*n,*incy).swap(vector(x,*n,-*incx).reverse());
  else if(*incx<0 && *incy<0) vector(y,*n,-*incy).reverse().swap(vector(x,*n,-*incx).reverse());

  return 1;
}

