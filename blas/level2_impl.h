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

int EIGEN_BLAS_FUNC(gemv)(char *opa, int *m, int *n, RealScalar *palpha, RealScalar *pa, int *lda, RealScalar *pb, int *incb, RealScalar *pbeta, RealScalar *pc, int *incc)
{
  Scalar* a = reinterpret_cast<Scalar*>(pa);
  Scalar* b = reinterpret_cast<Scalar*>(pb);
  Scalar* c = reinterpret_cast<Scalar*>(pc);
  Scalar alpha  = *reinterpret_cast<Scalar*>(palpha);
  Scalar beta   = *reinterpret_cast<Scalar*>(pbeta);

  if(beta!=Scalar(1))
    vector(c, *m, *incc) *= beta;

  if(OP(*opa)==NOTR)
    if(*incc==1)
      vector(c,*m)        += alpha * matrix(a,*m,*n,*lda) * vector(b,*n,*incb);
    else
      vector(c,*m,*incc)  += alpha * matrix(a,*m,*n,*lda) * vector(b,*n,*incb);
  else if(OP(*opa)==TR)
    if(*incb==1)
      vector(c,*m,*incc)  += alpha * matrix(a,*n,*m,*lda).transpose() * vector(b,*n);
    else
      vector(c,*m,*incc)  += alpha * matrix(a,*n,*m,*lda).transpose() * vector(b,*n,*incb);
  else if(OP(*opa)==TR)
    if(*incb==1)
      vector(c,*m,*incc)  += alpha * matrix(a,*n,*m,*lda).adjoint() * vector(b,*n);
    else
      vector(c,*m,*incc)  += alpha * matrix(a,*n,*m,*lda).adjoint() * vector(b,*n,*incb);
  else
    return 0;

  return 1;
}


int EIGEN_BLAS_FUNC(trsv)(char *uplo, char *opa, char *diag, int *n, RealScalar *pa, int *lda, RealScalar *pb, int *incb)
{
  return 0;

  typedef void (*functype)(int, const Scalar *, int, Scalar *, int);
  functype func[16];

  static bool init = false;
  if(!init)
  {
    for(int k=0; k<16; ++k)
      func[k] = 0;

//     func[NOTR  | (UP << 2) | (NUNIT << 3)] = (ei_triangular_solve_vector<Scalar, UpperTriangular|0,          false,ColMajor,ColMajor>::run);
//     func[TR    | (UP << 2) | (NUNIT << 3)] = (ei_triangular_solve_vector<Scalar, UpperTriangular|0,          false,RowMajor,ColMajor>::run);
//     func[ADJ   | (UP << 2) | (NUNIT << 3)] = (ei_triangular_solve_vector<Scalar, UpperTriangular|0,          Conj, RowMajor,ColMajor>::run);
//
//     func[NOTR  | (LO << 2) | (NUNIT << 3)] = (ei_triangular_solve_vector<Scalar, LowerTriangular|0,          false,ColMajor,ColMajor>::run);
//     func[TR    | (LO << 2) | (NUNIT << 3)] = (ei_triangular_solve_vector<Scalar, LowerTriangular|0,          false,RowMajor,ColMajor>::run);
//     func[ADJ   | (LO << 2) | (NUNIT << 3)] = (ei_triangular_solve_vector<Scalar, LowerTriangular|0,          Conj, RowMajor,ColMajor>::run);
//
//     func[NOTR  | (UP << 3) | (UNIT  << 3)] = (ei_triangular_solve_vector<Scalar, UpperTriangular|UnitDiagBit,false,ColMajor,ColMajor>::run);
//     func[TR    | (UP << 2) | (UNIT  << 3)] = (ei_triangular_solve_vector<Scalar, UpperTriangular|UnitDiagBit,false,RowMajor,ColMajor>::run);
//     func[ADJ   | (UP << 2) | (UNIT  << 3)] = (ei_triangular_solve_vector<Scalar, UpperTriangular|UnitDiagBit,Conj, RowMajor,ColMajor>::run);
//
//     func[NOTR  | (LO << 2) | (UNIT  << 3)] = (ei_triangular_solve_vector<Scalar, LowerTriangular|UnitDiagBit,false,ColMajor,ColMajor>::run);
//     func[TR    | (LO << 2) | (UNIT  << 3)] = (ei_triangular_solve_vector<Scalar, LowerTriangular|UnitDiagBit,false,RowMajor,ColMajor>::run);
//     func[ADJ   | (LO << 2) | (UNIT  << 3)] = (ei_triangular_solve_vector<Scalar, LowerTriangular|UnitDiagBit,Conj, RowMajor,ColMajor>::run);

    init = true;
  }

  Scalar* a = reinterpret_cast<Scalar*>(pa);
  Scalar* b = reinterpret_cast<Scalar*>(pb);

  int code = OP(*opa) | (UPLO(*uplo) << 2) | (DIAG(*diag) << 3);
  if(code>=16 || func[code]==0)
    return 0;

  func[code](*n, a, *lda, b, *incb);
  return 0;
}



int EIGEN_BLAS_FUNC(trmv)(char *uplo, char *opa, char *diag, int *n, RealScalar *pa, int *lda, RealScalar *pb, int *incb)
{
  return 0;
  // TODO

  typedef void (*functype)(int, const Scalar *, int, const Scalar *, int, Scalar *, int);
  functype func[16];

  static bool init = false;
  if(!init)
  {
    for(int k=0; k<16; ++k)
      func[k] = 0;

//     func[NOTR  | (UP << 2) | (NUNIT << 3)] = (ei_product_triangular_matrix_vector<Scalar,UpperTriangular|0,          true, ColMajor,false,ColMajor,false,ColMajor>::run);
//     func[TR    | (UP << 2) | (NUNIT << 3)] = (ei_product_triangular_matrix_vector<Scalar,UpperTriangular|0,          true, RowMajor,false,ColMajor,false,ColMajor>::run);
//     func[ADJ   | (UP << 2) | (NUNIT << 3)] = (ei_product_triangular_matrix_vector<Scalar,UpperTriangular|0,          true, RowMajor,Conj, ColMajor,false,ColMajor>::run);
//
//     func[NOTR  | (LO << 2) | (NUNIT << 3)] = (ei_product_triangular_matrix_vector<Scalar,LowerTriangular|0,          true, ColMajor,false,ColMajor,false,ColMajor>::run);
//     func[TR    | (LO << 2) | (NUNIT << 3)] = (ei_product_triangular_matrix_vector<Scalar,LowerTriangular|0,          true, RowMajor,false,ColMajor,false,ColMajor>::run);
//     func[ADJ   | (LO << 2) | (NUNIT << 3)] = (ei_product_triangular_matrix_vector<Scalar,LowerTriangular|0,          true, RowMajor,Conj, ColMajor,false,ColMajor>::run);
//
//     func[NOTR  | (UP << 2) | (UNIT  << 3)] = (ei_product_triangular_matrix_vector<Scalar,UpperTriangular|UnitDiagBit,true, ColMajor,false,ColMajor,false,ColMajor>::run);
//     func[TR    | (UP << 2) | (UNIT  << 3)] = (ei_product_triangular_matrix_vector<Scalar,UpperTriangular|UnitDiagBit,true, RowMajor,false,ColMajor,false,ColMajor>::run);
//     func[ADJ   | (UP << 2) | (UNIT  << 3)] = (ei_product_triangular_matrix_vector<Scalar,UpperTriangular|UnitDiagBit,true, RowMajor,Conj, ColMajor,false,ColMajor>::run);
//
//     func[NOTR  | (LO << 2) | (UNIT  << 3)] = (ei_product_triangular_matrix_vector<Scalar,LowerTriangular|UnitDiagBit,true, ColMajor,false,ColMajor,false,ColMajor>::run);
//     func[TR    | (LO << 2) | (UNIT  << 3)] = (ei_product_triangular_matrix_vector<Scalar,LowerTriangular|UnitDiagBit,true, RowMajor,false,ColMajor,false,ColMajor>::run);
//     func[ADJ   | (LO << 2) | (UNIT  << 3)] = (ei_product_triangular_matrix_vector<Scalar,LowerTriangular|UnitDiagBit,true, RowMajor,Conj, ColMajor,false,ColMajor>::run);

    init = true;
  }

  Scalar* a = reinterpret_cast<Scalar*>(pa);
  Scalar* b = reinterpret_cast<Scalar*>(pb);

  int code = OP(*opa) | (UPLO(*uplo) << 2) | (DIAG(*diag) << 3);
  if(code>=16 || func[code]==0)
    return 0;

  func[code](*n, a, *lda, b, *incb, b, *incb);
  return 0;
}

// y = alpha*A*x + beta*y
int EIGEN_BLAS_FUNC(ssymv) (char *uplo, int *n, RealScalar *palpha, RealScalar *pa, int *lda, RealScalar *px, int *incx, RealScalar *pbeta, RealScalar *py, int *incy)
{
  return 0;

  // TODO
}

int EIGEN_BLAS_FUNC(syr)(char *uplo, int *n, RealScalar *palpha, RealScalar *pa, int *inca, RealScalar *pc, int *ldc)
{
  return 0;

  // TODO
  typedef void (*functype)(int, const Scalar *, int, Scalar *, int, Scalar);
  functype func[2];

  static bool init = false;
  if(!init)
  {
    for(int k=0; k<2; ++k)
      func[k] = 0;

//     func[UP] = (ei_selfadjoint_product<Scalar,ColMajor,ColMajor,false,UpperTriangular>::run);
//     func[LO] = (ei_selfadjoint_product<Scalar,ColMajor,ColMajor,false,LowerTriangular>::run);

    init = true;
  }

  Scalar* a = reinterpret_cast<Scalar*>(pa);
  Scalar* c = reinterpret_cast<Scalar*>(pc);
  Scalar alpha = *reinterpret_cast<Scalar*>(palpha);

  int code = UPLO(*uplo);
  if(code>=2 || func[code]==0)
    return 0;

  func[code](*n, a, *inca, c, *ldc, alpha);
  return 1;
}



int EIGEN_BLAS_FUNC(syr2)(char *uplo, int *n, RealScalar *palpha, RealScalar *pa, int *inca, RealScalar *pb, int *incb, RealScalar *pc, int *ldc)
{
  return 0;

  // TODO
  typedef void (*functype)(int, const Scalar *, int, const Scalar *, int, Scalar *, int, Scalar);
  functype func[2];

  static bool init = false;
  if(!init)
  {
    for(int k=0; k<2; ++k)
      func[k] = 0;

//     func[UP] = (ei_selfadjoint_product<Scalar,ColMajor,ColMajor,false,UpperTriangular>::run);
//     func[LO] = (ei_selfadjoint_product<Scalar,ColMajor,ColMajor,false,LowerTriangular>::run);

    init = true;
  }

  Scalar* a = reinterpret_cast<Scalar*>(pa);
  Scalar* b = reinterpret_cast<Scalar*>(pb);
  Scalar* c = reinterpret_cast<Scalar*>(pc);
  Scalar alpha = *reinterpret_cast<Scalar*>(palpha);

  int code = UPLO(*uplo);
  if(code>=2 || func[code]==0)
    return 0;

  func[code](*n, a, *inca, b, *incb, c, *ldc, alpha);
  return 1;
}


#if ISCOMPLEX

#endif // ISCOMPLEX
