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
//   std::cerr << "in gemm " << *opa << " " << *opb << " " << *m << " " << *n << " " << *k << " " << *lda << " " << *ldb << " " << *ldc << " " << *palpha << " " << *pbeta << "\n";
  typedef void (*functype)(int, int, int, const Scalar *, int, const Scalar *, int, Scalar *, int, Scalar);
  static functype func[12];

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

  int code = OP(*opa) | (OP(*opb) << 2);
  if(code>=12 || func[code]==0 || (*m<0) || (*n<0) || (*k<0))
  {
    int info = 1;
    xerbla_("GEMM", &info, 4);
    return 0;
  }

  if(beta!=Scalar(1))
    if(beta==Scalar(0))
      matrix(c, *m, *n, *ldc).setZero();
    else
      matrix(c, *m, *n, *ldc) *= beta;

  func[code](*m, *n, *k, a, *lda, b, *ldb, c, *ldc, alpha);
  return 0;
}

int EIGEN_BLAS_FUNC(trsm)(char *side, char *uplo, char *opa, char *diag, int *m, int *n, RealScalar *palpha,  RealScalar *pa, int *lda, RealScalar *pb, int *ldb)
{
//   std::cerr << "in trsm " << *side << " " << *uplo << " " << *opa << " " << *diag << " " << *m << "," << *n << " " << *palpha << " " << *lda << " " << *ldb<< "\n";
  typedef void (*functype)(int, int, const Scalar *, int, Scalar *, int);
  static functype func[32];

  static bool init = false;
  if(!init)
  {
    for(int k=0; k<32; ++k)
      func[k] = 0;

    func[NOTR  | (LEFT  << 2) | (UP << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheLeft, Upper|0,          false,ColMajor,ColMajor>::run);
    func[TR    | (LEFT  << 2) | (UP << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheLeft, Lower|0,          false,RowMajor,ColMajor>::run);
    func[ADJ   | (LEFT  << 2) | (UP << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheLeft, Lower|0,          Conj, RowMajor,ColMajor>::run);

    func[NOTR  | (RIGHT << 2) | (UP << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheRight,Upper|0,          false,ColMajor,ColMajor>::run);
    func[TR    | (RIGHT << 2) | (UP << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheRight,Lower|0,          false,RowMajor,ColMajor>::run);
    func[ADJ   | (RIGHT << 2) | (UP << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheRight,Lower|0,          Conj, RowMajor,ColMajor>::run);

    func[NOTR  | (LEFT  << 2) | (LO << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheLeft, Lower|0,          false,ColMajor,ColMajor>::run);
    func[TR    | (LEFT  << 2) | (LO << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheLeft, Upper|0,          false,RowMajor,ColMajor>::run);
    func[ADJ   | (LEFT  << 2) | (LO << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheLeft, Upper|0,          Conj, RowMajor,ColMajor>::run);

    func[NOTR  | (RIGHT << 2) | (LO << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheRight,Lower|0,          false,ColMajor,ColMajor>::run);
    func[TR    | (RIGHT << 2) | (LO << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheRight,Upper|0,          false,RowMajor,ColMajor>::run);
    func[ADJ   | (RIGHT << 2) | (LO << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheRight,Upper|0,          Conj, RowMajor,ColMajor>::run);


    func[NOTR  | (LEFT  << 2) | (UP << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheLeft, Upper|UnitDiag,false,ColMajor,ColMajor>::run);
    func[TR    | (LEFT  << 2) | (UP << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheLeft, Lower|UnitDiag,false,RowMajor,ColMajor>::run);
    func[ADJ   | (LEFT  << 2) | (UP << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheLeft, Lower|UnitDiag,Conj, RowMajor,ColMajor>::run);

    func[NOTR  | (RIGHT << 2) | (UP << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheRight,Upper|UnitDiag,false,ColMajor,ColMajor>::run);
    func[TR    | (RIGHT << 2) | (UP << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheRight,Lower|UnitDiag,false,RowMajor,ColMajor>::run);
    func[ADJ   | (RIGHT << 2) | (UP << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheRight,Lower|UnitDiag,Conj, RowMajor,ColMajor>::run);

    func[NOTR  | (LEFT  << 2) | (LO << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheLeft, Lower|UnitDiag,false,ColMajor,ColMajor>::run);
    func[TR    | (LEFT  << 2) | (LO << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheLeft, Upper|UnitDiag,false,RowMajor,ColMajor>::run);
    func[ADJ   | (LEFT  << 2) | (LO << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheLeft, Upper|UnitDiag,Conj, RowMajor,ColMajor>::run);

    func[NOTR  | (RIGHT << 2) | (LO << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheRight,Lower|UnitDiag,false,ColMajor,ColMajor>::run);
    func[TR    | (RIGHT << 2) | (LO << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheRight,Upper|UnitDiag,false,RowMajor,ColMajor>::run);
    func[ADJ   | (RIGHT << 2) | (LO << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,OnTheRight,Upper|UnitDiag,Conj, RowMajor,ColMajor>::run);

    init = true;
  }

  Scalar* a = reinterpret_cast<Scalar*>(pa);
  Scalar* b = reinterpret_cast<Scalar*>(pb);
  Scalar  alpha = *reinterpret_cast<Scalar*>(palpha);

  int code = OP(*opa) | (SIDE(*side) << 2) | (UPLO(*uplo) << 3) | (DIAG(*diag) << 4);
  if(code>=32 || func[code]==0 || *m<0 || *n <0)
  {
    int info=1;
    xerbla_("TRSM",&info,4);
    return 0;
  }

  if(SIDE(*side)==LEFT)
    func[code](*m, *n, a, *lda, b, *ldb);
  else
    func[code](*n, *m, a, *lda, b, *ldb);

  if(alpha!=Scalar(1))
    matrix(b,*m,*n,*ldb) *= alpha;

  return 0;
}


// b = alpha*op(a)*b  for side = 'L'or'l'
// b = alpha*b*op(a)  for side = 'R'or'r'
int EIGEN_BLAS_FUNC(trmm)(char *side, char *uplo, char *opa, char *diag, int *m, int *n, RealScalar *palpha,  RealScalar *pa, int *lda, RealScalar *pb, int *ldb)
{
//   std::cerr << "in trmm " << *side << " " << *uplo << " " << *opa << " " << *diag << " " << *m << " " << *n << " " << *lda << " " << *ldb << " " << *palpha << "\n";
  typedef void (*functype)(int, int, const Scalar *, int, const Scalar *, int, Scalar *, int, Scalar);
  static functype func[32];
  static bool init = false;
  if(!init)
  {
    for(int k=0; k<32; ++k)
      func[k] = 0;

    func[NOTR  | (LEFT  << 2) | (UP << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,Upper|0,          true, ColMajor,false,ColMajor,false,ColMajor>::run);
    func[TR    | (LEFT  << 2) | (UP << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,Lower|0,          true, RowMajor,false,ColMajor,false,ColMajor>::run);
    func[ADJ   | (LEFT  << 2) | (UP << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,Lower|0,          true, RowMajor,Conj, ColMajor,false,ColMajor>::run);

    func[NOTR  | (RIGHT << 2) | (UP << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,Upper|0,          false,ColMajor,false,ColMajor,false,ColMajor>::run);
    func[TR    | (RIGHT << 2) | (UP << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,Lower|0,          false,ColMajor,false,RowMajor,false,ColMajor>::run);
    func[ADJ   | (RIGHT << 2) | (UP << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,Lower|0,          false,ColMajor,false,RowMajor,Conj, ColMajor>::run);

    func[NOTR  | (LEFT  << 2) | (LO << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,Lower|0,          true, ColMajor,false,ColMajor,false,ColMajor>::run);
    func[TR    | (LEFT  << 2) | (LO << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,Upper|0,          true, RowMajor,false,ColMajor,false,ColMajor>::run);
    func[ADJ   | (LEFT  << 2) | (LO << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,Upper|0,          true, RowMajor,Conj, ColMajor,false,ColMajor>::run);

    func[NOTR  | (RIGHT << 2) | (LO << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,Lower|0,          false,ColMajor,false,ColMajor,false,ColMajor>::run);
    func[TR    | (RIGHT << 2) | (LO << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,Upper|0,          false,ColMajor,false,RowMajor,false,ColMajor>::run);
    func[ADJ   | (RIGHT << 2) | (LO << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,Upper|0,          false,ColMajor,false,RowMajor,Conj, ColMajor>::run);

    func[NOTR  | (LEFT  << 2) | (UP << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,Upper|UnitDiag,true, ColMajor,false,ColMajor,false,ColMajor>::run);
    func[TR    | (LEFT  << 2) | (UP << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,Lower|UnitDiag,true, RowMajor,false,ColMajor,false,ColMajor>::run);
    func[ADJ   | (LEFT  << 2) | (UP << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,Lower|UnitDiag,true, RowMajor,Conj, ColMajor,false,ColMajor>::run);

    func[NOTR  | (RIGHT << 2) | (UP << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,Upper|UnitDiag,false,ColMajor,false,ColMajor,false,ColMajor>::run);
    func[TR    | (RIGHT << 2) | (UP << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,Lower|UnitDiag,false,ColMajor,false,RowMajor,false,ColMajor>::run);
    func[ADJ   | (RIGHT << 2) | (UP << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,Lower|UnitDiag,false,ColMajor,false,RowMajor,Conj, ColMajor>::run);

    func[NOTR  | (LEFT  << 2) | (LO << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,Lower|UnitDiag,true, ColMajor,false,ColMajor,false,ColMajor>::run);
    func[TR    | (LEFT  << 2) | (LO << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,Upper|UnitDiag,true, RowMajor,false,ColMajor,false,ColMajor>::run);
    func[ADJ   | (LEFT  << 2) | (LO << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,Upper|UnitDiag,true, RowMajor,Conj, ColMajor,false,ColMajor>::run);

    func[NOTR  | (RIGHT << 2) | (LO << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,Lower|UnitDiag,false,ColMajor,false,ColMajor,false,ColMajor>::run);
    func[TR    | (RIGHT << 2) | (LO << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,Upper|UnitDiag,false,ColMajor,false,RowMajor,false,ColMajor>::run);
    func[ADJ   | (RIGHT << 2) | (LO << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,Upper|UnitDiag,false,ColMajor,false,RowMajor,Conj, ColMajor>::run);

    init = true;
  }

  Scalar* a = reinterpret_cast<Scalar*>(pa);
  Scalar* b = reinterpret_cast<Scalar*>(pb);
  Scalar  alpha = *reinterpret_cast<Scalar*>(palpha);

  int code = OP(*opa) | (SIDE(*side) << 2) | (UPLO(*uplo) << 3) | (DIAG(*diag) << 4);
  if(code>=32 || func[code]==0 || *m<0 || *n <0)
  {
    int info=1;
    xerbla_("TRMM",&info,4);
    return 0;
  }

  // FIXME find a way to avoid this copy
  Matrix<Scalar,Dynamic,Dynamic> tmp = matrix(b,*m,*n,*ldb);
  matrix(b,*m,*n,*ldb).setZero();

  if(SIDE(*side)==LEFT)
    func[code](*m, *n, a, *lda, tmp.data(), tmp.outerStride(), b, *ldb, alpha);
  else
    func[code](*n, *m, tmp.data(), tmp.outerStride(), a, *lda, b, *ldb, alpha);
  return 1;
}

// c = alpha*a*b + beta*c  for side = 'L'or'l'
// c = alpha*b*a + beta*c  for side = 'R'or'r
int EIGEN_BLAS_FUNC(symm)(char *side, char *uplo, int *m, int *n, RealScalar *palpha, RealScalar *pa, int *lda, RealScalar *pb, int *ldb, RealScalar *pbeta, RealScalar *pc, int *ldc)
{
//   std::cerr << "in symm " << *side << " " << *uplo << " " << *m << "x" << *n << " lda:" << *lda << " ldb:" << *ldb << " ldc:" << *ldc << " alpha:" << *palpha << " beta:" << *pbeta << " "
//             << pa << " " << pb << " " << pc << "\n";
  Scalar* a = reinterpret_cast<Scalar*>(pa);
  Scalar* b = reinterpret_cast<Scalar*>(pb);
  Scalar* c = reinterpret_cast<Scalar*>(pc);
  Scalar alpha = *reinterpret_cast<Scalar*>(palpha);
  Scalar beta  = *reinterpret_cast<Scalar*>(pbeta);

  if(*m<0 || *n<0)
  {
    int info=1;
    xerbla_("SYMM",&info,4);
    return 0;
  }

  if(beta!=Scalar(1))
    if(beta==Scalar(0))
      matrix(c, *m, *n, *ldc).setZero();
    else
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

  return 0;
}

// c = alpha*a*a' + beta*c  for op = 'N'or'n'
// c = alpha*a'*a + beta*c  for op = 'T'or't','C'or'c'
int EIGEN_BLAS_FUNC(syrk)(char *uplo, char *op, int *n, int *k, RealScalar *palpha, RealScalar *pa, int *lda, RealScalar *pbeta, RealScalar *pc, int *ldc)
{
//   std::cerr << "in syrk " << *uplo << " " << *op << " " << *n << " " << *k << " " << *palpha << " " << *lda << " " << *pbeta << "\n";
  typedef void (*functype)(int, int, const Scalar *, int, Scalar *, int, Scalar);
  static functype func[8];

  static bool init = false;
  if(!init)
  {
    for(int k=0; k<8; ++k)
      func[k] = 0;

    func[NOTR  | (UP << 2)] = (ei_selfadjoint_product<Scalar,ColMajor,ColMajor,true, Upper>::run);
    func[TR    | (UP << 2)] = (ei_selfadjoint_product<Scalar,RowMajor,ColMajor,false,Upper>::run);
    func[ADJ   | (UP << 2)] = (ei_selfadjoint_product<Scalar,RowMajor,ColMajor,false,Upper>::run);

    func[NOTR  | (LO << 2)] = (ei_selfadjoint_product<Scalar,ColMajor,ColMajor,true, Lower>::run);
    func[TR    | (LO << 2)] = (ei_selfadjoint_product<Scalar,RowMajor,ColMajor,false,Lower>::run);
    func[ADJ   | (LO << 2)] = (ei_selfadjoint_product<Scalar,RowMajor,ColMajor,false,Lower>::run);

    init = true;
  }

  Scalar* a = reinterpret_cast<Scalar*>(pa);
  Scalar* c = reinterpret_cast<Scalar*>(pc);
  Scalar alpha = *reinterpret_cast<Scalar*>(palpha);
  Scalar beta  = *reinterpret_cast<Scalar*>(pbeta);

  int code = OP(*op) | (UPLO(*uplo) << 2);
  if(code>=8 || func[code]==0 || *n<0 || *k<0)
  {
    int info=1;
    xerbla_("SYRK",&info,4);
    return 0;
  }

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
  static functype func[8];

  static bool init = false;
  if(!init)
  {
    for(int k=0; k<8; ++k)
      func[k] = 0;

    func[NOTR  | (UP << 2)] = (ei_selfadjoint_product<Scalar,ColMajor,ColMajor,true, Upper>::run);
    func[ADJ   | (UP << 2)] = (ei_selfadjoint_product<Scalar,RowMajor,ColMajor,false,Upper>::run);

    func[NOTR  | (LO << 2)] = (ei_selfadjoint_product<Scalar,ColMajor,ColMajor,true, Lower>::run);
    func[ADJ   | (LO << 2)] = (ei_selfadjoint_product<Scalar,RowMajor,ColMajor,false,Lower>::run);

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
