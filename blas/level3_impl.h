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

int EIGEN_BLAS_FUNC(gemm)(char *opa, char *opb, int *m, int *n, int *k, RealScalar *palpha, RealScalar *pa, int *lda, RealScalar *pb, int *ldb, RealScalar *pbeta, RealScalar *pc, int *ldc)
{
  typedef void (*functype)(int, int, int, const Scalar *, int, const Scalar *, int, Scalar *, int, Scalar);
  functype func[12];

  static bool init = false;
  if(!init)
  {
    for(int k=0; k<12; ++k)
      func[k] = 0;
    func[NOTR  | (NOTR << 2)] = (ei_general_matrix_matrix_product<Scalar,ColMajor,false,ColMajor,false,ColMajor>::run);
    func[TR    | (NOTR << 2)] = (ei_general_matrix_matrix_product<Scalar,RowMajor,false,ColMajor,false,ColMajor>::run);
    func[ADJ   | (NOTR << 2)] = (ei_general_matrix_matrix_product<Scalar,RowMajor,Conj, ColMajor,false,ColMajor>::run);
    func[NOTR  | (TR   << 2)] = (ei_general_matrix_matrix_product<Scalar,ColMajor,false,RowMajor,false,ColMajor>::run);
    func[TR    | (TR   << 2)] = (ei_general_matrix_matrix_product<Scalar,RowMajor,false,RowMajor,false,ColMajor>::run);
    func[ADJ   | (TR   << 2)] = (ei_general_matrix_matrix_product<Scalar,RowMajor,Conj, RowMajor,false,ColMajor>::run);
    func[NOTR  | (ADJ  << 2)] = (ei_general_matrix_matrix_product<Scalar,ColMajor,false,RowMajor,Conj, ColMajor>::run);
    func[TR    | (ADJ  << 2)] = (ei_general_matrix_matrix_product<Scalar,RowMajor,false,RowMajor,Conj, ColMajor>::run);
    func[ADJ   | (ADJ  << 2)] = (ei_general_matrix_matrix_product<Scalar,RowMajor,Conj, RowMajor,Conj, ColMajor>::run);
    init = true;
  }

  Scalar* a = reinterpret_cast<Scalar*>(pa);
  Scalar* b = reinterpret_cast<Scalar*>(pb);
  Scalar* c = reinterpret_cast<Scalar*>(pc);
  Scalar alpha  = *reinterpret_cast<Scalar*>(palpha);
  Scalar beta   = *reinterpret_cast<Scalar*>(pbeta);

  if(beta!=Scalar(1))
    matrix(c, *m, *n, *ldc) *= beta;

  int code = OP(*opa) | (OP(*opb) << 2);
  if(code>=12 || func[code]==0)
    return 0;

  func[code](*m, *n, *k, a, *lda, b, *ldb, c, *ldc, alpha);
  return 1;
}

int EIGEN_BLAS_FUNC(trsm)(char *side, char *uplo, char *opa, char *diag, int *m, int *n, RealScalar *palpha,  RealScalar *pa, int *lda, RealScalar *pb, int *ldb)
{
  typedef void (*functype)(int, int, const Scalar *, int, Scalar *, int);
  functype func[32];

  static bool init = false;
  if(!init)
  {
    for(int k=0; k<32; ++k)
      func[k] = 0;

    func[NOTR  | (LEFT  << 2) | (UP << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheLeft, UpperTriangular|0,          false,ColMajor,ColMajor>::run);
    func[TR    | (LEFT  << 2) | (UP << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheLeft, UpperTriangular|0,          false,RowMajor,ColMajor>::run);
    func[ADJ   | (LEFT  << 2) | (UP << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheLeft, UpperTriangular|0,          Conj, RowMajor,ColMajor>::run);

    func[NOTR  | (RIGHT << 2) | (UP << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheRight,UpperTriangular|0,          false,ColMajor,ColMajor>::run);
    func[TR    | (RIGHT << 2) | (UP << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheRight,UpperTriangular|0,          false,RowMajor,ColMajor>::run);
    func[ADJ   | (RIGHT << 2) | (UP << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheRight,UpperTriangular|0,          Conj, RowMajor,ColMajor>::run);

    func[NOTR  | (LEFT  << 2) | (LO << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheLeft, LowerTriangular|0,          false,ColMajor,ColMajor>::run);
    func[TR    | (LEFT  << 2) | (LO << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheLeft, LowerTriangular|0,          false,RowMajor,ColMajor>::run);
    func[ADJ   | (LEFT  << 2) | (LO << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheLeft, LowerTriangular|0,          Conj, RowMajor,ColMajor>::run);

    func[NOTR  | (RIGHT << 2) | (LO << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheRight,LowerTriangular|0,          false,ColMajor,ColMajor>::run);
    func[TR    | (RIGHT << 2) | (LO << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheRight,LowerTriangular|0,          false,RowMajor,ColMajor>::run);
    func[ADJ   | (RIGHT << 2) | (LO << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheRight,LowerTriangular|0,          Conj, RowMajor,ColMajor>::run);


    func[NOTR  | (LEFT  << 2) | (UP << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheLeft, UpperTriangular|UnitDiagBit,false,ColMajor,ColMajor>::run);
    func[TR    | (LEFT  << 2) | (UP << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheLeft, UpperTriangular|UnitDiagBit,false,RowMajor,ColMajor>::run);
    func[ADJ   | (LEFT  << 2) | (UP << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheLeft, UpperTriangular|UnitDiagBit,Conj, RowMajor,ColMajor>::run);

    func[NOTR  | (RIGHT << 2) | (UP << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheRight,UpperTriangular|UnitDiagBit,false,ColMajor,ColMajor>::run);
    func[TR    | (RIGHT << 2) | (UP << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheRight,UpperTriangular|UnitDiagBit,false,RowMajor,ColMajor>::run);
    func[ADJ   | (RIGHT << 2) | (UP << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheRight,UpperTriangular|UnitDiagBit,Conj, RowMajor,ColMajor>::run);

    func[NOTR  | (LEFT  << 2) | (LO << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheLeft, LowerTriangular|UnitDiagBit,false,ColMajor,ColMajor>::run);
    func[TR    | (LEFT  << 2) | (LO << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheLeft, LowerTriangular|UnitDiagBit,false,RowMajor,ColMajor>::run);
    func[ADJ   | (LEFT  << 2) | (LO << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheLeft, LowerTriangular|UnitDiagBit,Conj, RowMajor,ColMajor>::run);

    func[NOTR  | (RIGHT << 2) | (LO << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheRight,LowerTriangular|UnitDiagBit,false,ColMajor,ColMajor>::run);
    func[TR    | (RIGHT << 2) | (LO << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheRight,LowerTriangular|UnitDiagBit,false,RowMajor,ColMajor>::run);
    func[ADJ   | (RIGHT << 2) | (LO << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheRight,LowerTriangular|UnitDiagBit,Conj, RowMajor,ColMajor>::run);

    init = true;
  }

  Scalar* a = reinterpret_cast<Scalar*>(pa);
  Scalar* b = reinterpret_cast<Scalar*>(pb);
  Scalar  alpha = *reinterpret_cast<Scalar*>(palpha);

  // TODO handle alpha

  int code = OP(*opa) | (SIDE(*side) << 2) | (UPLO(*uplo) << 3) | (DIAG(*diag) << 4);
  if(code>=32 || func[code]==0)
    return 0;

  func[code](*m, *n, a, *lda, b, *ldb);
  return 1;
}


// b = alpha*op(a)*b  for side = 'L'or'l'
// b = alpha*b*op(a)  for side = 'R'or'r'
int EIGEN_BLAS_FUNC(trmm)(char *side, char *uplo, char *opa, char *diag, int *m, int *n, RealScalar *palpha,  RealScalar *pa, int *lda, RealScalar *pb, int *ldb)
{
  typedef void (*functype)(int, int, const Scalar *, int, const Scalar *, int, Scalar *, int, Scalar);
  functype func[32];

  static bool init = false;
  if(!init)
  {
    for(int k=0; k<32; ++k)
      func[k] = 0;

    func[NOTR  | (LEFT  << 2) | (UP << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,UpperTriangular|0,          true, ColMajor,false,ColMajor,false,ColMajor>::run);
    func[TR    | (LEFT  << 2) | (UP << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,UpperTriangular|0,          true, RowMajor,false,ColMajor,false,ColMajor>::run);
    func[ADJ   | (LEFT  << 2) | (UP << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,UpperTriangular|0,          true, RowMajor,Conj, ColMajor,false,ColMajor>::run);

    func[NOTR  | (RIGHT << 2) | (UP << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,UpperTriangular|0,          false,ColMajor,false,ColMajor,false,ColMajor>::run);
    func[TR    | (RIGHT << 2) | (UP << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,UpperTriangular|0,          false,ColMajor,false,RowMajor,false,ColMajor>::run);
    func[ADJ   | (RIGHT << 2) | (UP << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,UpperTriangular|0,          false,ColMajor,false,RowMajor,Conj, ColMajor>::run);

    func[NOTR  | (LEFT  << 2) | (LO << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,LowerTriangular|0,          true, ColMajor,false,ColMajor,false,ColMajor>::run);
    func[TR    | (LEFT  << 2) | (LO << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,LowerTriangular|0,          true, RowMajor,false,ColMajor,false,ColMajor>::run);
    func[ADJ   | (LEFT  << 2) | (LO << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,LowerTriangular|0,          true, RowMajor,Conj, ColMajor,false,ColMajor>::run);

    func[NOTR  | (RIGHT << 2) | (LO << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,LowerTriangular|0,          false,ColMajor,false,ColMajor,false,ColMajor>::run);
    func[TR    | (RIGHT << 2) | (LO << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,LowerTriangular|0,          false,ColMajor,false,RowMajor,false,ColMajor>::run);
    func[ADJ   | (RIGHT << 2) | (LO << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,LowerTriangular|0,          false,ColMajor,false,RowMajor,Conj, ColMajor>::run);

    func[NOTR  | (LEFT  << 2) | (UP << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,UpperTriangular|UnitDiagBit,true, ColMajor,false,ColMajor,false,ColMajor>::run);
    func[TR    | (LEFT  << 2) | (UP << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,UpperTriangular|UnitDiagBit,true, RowMajor,false,ColMajor,false,ColMajor>::run);
    func[ADJ   | (LEFT  << 2) | (UP << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,UpperTriangular|UnitDiagBit,true, RowMajor,Conj, ColMajor,false,ColMajor>::run);

    func[NOTR  | (RIGHT << 2) | (UP << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,UpperTriangular|UnitDiagBit,false,ColMajor,false,ColMajor,false,ColMajor>::run);
    func[TR    | (RIGHT << 2) | (UP << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,UpperTriangular|UnitDiagBit,false,ColMajor,false,RowMajor,false,ColMajor>::run);
    func[ADJ   | (RIGHT << 2) | (UP << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,UpperTriangular|UnitDiagBit,false,ColMajor,false,RowMajor,Conj, ColMajor>::run);

    func[NOTR  | (LEFT  << 2) | (LO << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,LowerTriangular|UnitDiagBit,true, ColMajor,false,ColMajor,false,ColMajor>::run);
    func[TR    | (LEFT  << 2) | (LO << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,LowerTriangular|UnitDiagBit,true, RowMajor,false,ColMajor,false,ColMajor>::run);
    func[ADJ   | (LEFT  << 2) | (LO << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,LowerTriangular|UnitDiagBit,true, RowMajor,Conj, ColMajor,false,ColMajor>::run);

    func[NOTR  | (RIGHT << 2) | (LO << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,LowerTriangular|UnitDiagBit,false,ColMajor,false,ColMajor,false,ColMajor>::run);
    func[TR    | (RIGHT << 2) | (LO << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,LowerTriangular|UnitDiagBit,false,ColMajor,false,RowMajor,false,ColMajor>::run);
    func[ADJ   | (RIGHT << 2) | (LO << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,LowerTriangular|UnitDiagBit,false,ColMajor,false,RowMajor,Conj, ColMajor>::run);

    init = true;
  }

  Scalar* a = reinterpret_cast<Scalar*>(pa);
  Scalar* b = reinterpret_cast<Scalar*>(pb);
  Scalar  alpha = *reinterpret_cast<Scalar*>(palpha);

  int code = OP(*opa) | (SIDE(*side) << 2) | (UPLO(*uplo) << 3) | (DIAG(*diag) << 4);
  if(code>=32 || func[code]==0)
    return 0;

  func[code](*m, *n, a, *lda, b, *ldb, b, *ldb, alpha);
  return 1;
}

// c = alpha*a*b + beta*c  for side = 'L'or'l'
// c = alpha*b*a + beta*c  for side = 'R'or'r
int EIGEN_BLAS_FUNC(symm)(char *side, char *uplo, int *m, int *n, RealScalar *palpha, RealScalar *pa, int *lda, RealScalar *pb, int *ldb, RealScalar *pbeta, RealScalar *pc, int *ldc)
{
  Scalar* a = reinterpret_cast<Scalar*>(pa);
  Scalar* b = reinterpret_cast<Scalar*>(pb);
  Scalar* c = reinterpret_cast<Scalar*>(pc);
  Scalar alpha = *reinterpret_cast<Scalar*>(palpha);
  Scalar beta  = *reinterpret_cast<Scalar*>(pbeta);

  if(beta!=Scalar(1))
    matrix(c, *m, *n, *ldc) *= beta;

  if(SIDE(*side)==LEFT)
    if(UPLO(*uplo)==UP)
      ei_product_selfadjoint_matrix<Scalar, RowMajor,true,false, ColMajor,false,false, ColMajor>::run(*m, *n, a, *lda, b, *ldb, c, *ldc, alpha);
    else if(UPLO(*uplo)==LO)
      ei_product_selfadjoint_matrix<Scalar, ColMajor,true,false, ColMajor,false,false, ColMajor>::run(*m, *n, a, *lda, b, *ldb, c, *ldc, alpha);
    else
      return 0;
  else if(SIDE(*side)==RIGHT)
    if(UPLO(*uplo)==UP)
      ei_product_selfadjoint_matrix<Scalar, ColMajor,false,false, RowMajor,true,false, ColMajor>::run(*m, *n, b, *ldb, a, *lda, c, *ldc, alpha);
    else if(UPLO(*uplo)==LO)
      ei_product_selfadjoint_matrix<Scalar, ColMajor,false,false, ColMajor,true,false, ColMajor>::run(*m, *n, b, *ldb, a, *lda, c, *ldc, alpha);
    else
      return 0;
  else
    return 0;

  return 1;
}

// c = alpha*a*a' + beta*c  for op = 'N'or'n'
// c = alpha*a'*a + beta*c  for op = 'T'or't','C'or'c'
int EIGEN_BLAS_FUNC(syrk)(char *uplo, char *op, int *n, int *k, RealScalar *palpha, RealScalar *pa, int *lda, RealScalar *pbeta, RealScalar *pc, int *ldc)
{
  typedef void (*functype)(int, int, const Scalar *, int, Scalar *, int, Scalar);
  functype func[8];

  static bool init = false;
  if(!init)
  {
    for(int k=0; k<8; ++k)
      func[k] = 0;

    func[NOTR  | (UP << 2)] = (ei_selfadjoint_product<Scalar,ColMajor,ColMajor,true, UpperTriangular>::run);
    func[TR    | (UP << 2)] = (ei_selfadjoint_product<Scalar,RowMajor,ColMajor,false,UpperTriangular>::run);
    func[ADJ   | (UP << 2)] = (ei_selfadjoint_product<Scalar,RowMajor,ColMajor,false,UpperTriangular>::run);

    func[NOTR  | (LO << 2)] = (ei_selfadjoint_product<Scalar,ColMajor,ColMajor,true, LowerTriangular>::run);
    func[TR    | (LO << 2)] = (ei_selfadjoint_product<Scalar,RowMajor,ColMajor,false,LowerTriangular>::run);
    func[ADJ   | (LO << 2)] = (ei_selfadjoint_product<Scalar,RowMajor,ColMajor,false,LowerTriangular>::run);

    init = true;
  }

  Scalar* a = reinterpret_cast<Scalar*>(pa);
  Scalar* c = reinterpret_cast<Scalar*>(pc);
  Scalar alpha = *reinterpret_cast<Scalar*>(palpha);
  Scalar beta  = *reinterpret_cast<Scalar*>(pbeta);

  int code = OP(*op) | (UPLO(*uplo) << 2);
  if(code>=8 || func[code]==0)
    return 0;

  if(beta!=Scalar(1))
    matrix(c, *n, *n, *ldc) *= beta;

  func[code](*n, *k, a, *lda, c, *ldc, alpha);
  return 1;
}

// c = alpha*a*b' + alpha*b*a' + beta*c  for op = 'N'or'n'
// c = alpha*a'*b + alpha*b'*a + beta*c  for op = 'T'or't'
int EIGEN_BLAS_FUNC(syr2k)(char *uplo, char *op, int *n, int *k, RealScalar *palpha, RealScalar *pa, int *lda, RealScalar *pb, int *ldb, RealScalar *pbeta, RealScalar *pc, int *ldc)
{
  Scalar* a = reinterpret_cast<Scalar*>(pa);
  Scalar* b = reinterpret_cast<Scalar*>(pb);
  Scalar* c = reinterpret_cast<Scalar*>(pc);
  Scalar alpha = *reinterpret_cast<Scalar*>(palpha);
  Scalar beta  = *reinterpret_cast<Scalar*>(pbeta);

  // TODO

  return 0;
}


#if ISCOMPLEX

// c = alpha*a*b + beta*c  for side = 'L'or'l'
// c = alpha*b*a + beta*c  for side = 'R'or'r
int EIGEN_BLAS_FUNC(hemm)(char *side, char *uplo, int *m, int *n, RealScalar *palpha, RealScalar *pa, int *lda, RealScalar *pb, int *ldb, RealScalar *pbeta, RealScalar *pc, int *ldc)
{
  Scalar* a = reinterpret_cast<Scalar*>(pa);
  Scalar* b = reinterpret_cast<Scalar*>(pb);
  Scalar* c = reinterpret_cast<Scalar*>(pc);
  Scalar alpha = *reinterpret_cast<Scalar*>(palpha);
  Scalar beta  = *reinterpret_cast<Scalar*>(pbeta);

  if(beta!=Scalar(1))
    matrix(c, *m, *n, *ldc) *= beta;

  if(SIDE(*side)==LEFT)
    if(UPLO(*uplo)==UP)
      ei_product_selfadjoint_matrix<Scalar, RowMajor,true,Conj,  ColMajor,false,false, ColMajor>::run(*m, *n, a, *lda, b, *ldb, c, *ldc, alpha);
    else if(UPLO(*uplo)==LO)
      ei_product_selfadjoint_matrix<Scalar, ColMajor,true,false, ColMajor,false,false, ColMajor>::run(*m, *n, a, *lda, b, *ldb, c, *ldc, alpha);
    else
      return 0;
  else if(SIDE(*side)==RIGHT)
    if(UPLO(*uplo)==UP)
      ei_product_selfadjoint_matrix<Scalar, ColMajor,false,false, RowMajor,true,Conj,  ColMajor>::run(*m, *n, b, *ldb, a, *lda, c, *ldc, alpha);
    else if(UPLO(*uplo)==LO)
      ei_product_selfadjoint_matrix<Scalar, ColMajor,false,false, ColMajor,true,false, ColMajor>::run(*m, *n, b, *ldb, a, *lda, c, *ldc, alpha);
    else
      return 0;
  else
    return 0;

  return 1;
}

// c = alpha*a*conj(a') + beta*c  for op = 'N'or'n'
// c = alpha*conj(a')*a + beta*c  for op  = 'C'or'c'
int EIGEN_BLAS_FUNC(herk)(char *uplo, char *op, int *n, int *k, RealScalar *palpha, RealScalar *pa, int *lda, RealScalar *pbeta, RealScalar *pc, int *ldc)
{
  typedef void (*functype)(int, int, const Scalar *, int, Scalar *, int, Scalar);
  functype func[8];

  static bool init = false;
  if(!init)
  {
    for(int k=0; k<8; ++k)
      func[k] = 0;

    func[NOTR  | (UP << 2)] = (ei_selfadjoint_product<Scalar,ColMajor,ColMajor,true, UpperTriangular>::run);
    func[ADJ   | (UP << 2)] = (ei_selfadjoint_product<Scalar,RowMajor,ColMajor,false,UpperTriangular>::run);

    func[NOTR  | (LO << 2)] = (ei_selfadjoint_product<Scalar,ColMajor,ColMajor,true, LowerTriangular>::run);
    func[ADJ   | (LO << 2)] = (ei_selfadjoint_product<Scalar,RowMajor,ColMajor,false,LowerTriangular>::run);

    init = true;
  }

  Scalar* a = reinterpret_cast<Scalar*>(pa);
  Scalar* c = reinterpret_cast<Scalar*>(pc);
  Scalar alpha = *reinterpret_cast<Scalar*>(palpha);
  Scalar beta  = *reinterpret_cast<Scalar*>(pbeta);

  int code = OP(*op) | (UPLO(*uplo) << 2);
  if(code>=8 || func[code]==0)
    return 0;

  if(beta!=Scalar(1))
    matrix(c, *n, *n, *ldc) *= beta;

  func[code](*n, *k, a, *lda, c, *ldc, alpha);
  return 1;
}

// c = alpha*a*conj(b') + conj(alpha)*b*conj(a') + beta*c,  for op = 'N'or'n'
// c = alpha*conj(b')*a + conj(alpha)*conj(a')*b + beta*c,  for op = 'C'or'c'
int EIGEN_BLAS_FUNC(her2k)(char *uplo, char *op, int *n, int *k, RealScalar *palpha, RealScalar *pa, int *lda, RealScalar *pb, int *ldb, RealScalar *pbeta, RealScalar *pc, int *ldc)
{
  Scalar* a = reinterpret_cast<Scalar*>(pa);
  Scalar* b = reinterpret_cast<Scalar*>(pb);
  Scalar* c = reinterpret_cast<Scalar*>(pc);
  Scalar alpha = *reinterpret_cast<Scalar*>(palpha);
  Scalar beta  = *reinterpret_cast<Scalar*>(pbeta);

  // TODO

  return 0;
}

#endif // ISCOMPLEX
