// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
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
  typedef void (*functype)(DenseIndex, DenseIndex, DenseIndex, const Scalar *, DenseIndex, const Scalar *, DenseIndex, Scalar *, DenseIndex, Scalar, ei_level3_blocking<Scalar,Scalar>&, Eigen::GemmParallelInfo<DenseIndex>*);
  static functype func[12];

  static bool init = false;
  if(!init)
  {
    for(int k=0; k<12; ++k)
      func[k] = 0;
    func[NOTR  | (NOTR << 2)] = (ei_general_matrix_matrix_product<DenseIndex,Scalar,ColMajor,false,Scalar,ColMajor,false,ColMajor>::run);
    func[TR    | (NOTR << 2)] = (ei_general_matrix_matrix_product<DenseIndex,Scalar,RowMajor,false,Scalar,ColMajor,false,ColMajor>::run);
    func[ADJ   | (NOTR << 2)] = (ei_general_matrix_matrix_product<DenseIndex,Scalar,RowMajor,Conj, Scalar,ColMajor,false,ColMajor>::run);
    func[NOTR  | (TR   << 2)] = (ei_general_matrix_matrix_product<DenseIndex,Scalar,ColMajor,false,Scalar,RowMajor,false,ColMajor>::run);
    func[TR    | (TR   << 2)] = (ei_general_matrix_matrix_product<DenseIndex,Scalar,RowMajor,false,Scalar,RowMajor,false,ColMajor>::run);
    func[ADJ   | (TR   << 2)] = (ei_general_matrix_matrix_product<DenseIndex,Scalar,RowMajor,Conj, Scalar,RowMajor,false,ColMajor>::run);
    func[NOTR  | (ADJ  << 2)] = (ei_general_matrix_matrix_product<DenseIndex,Scalar,ColMajor,false,Scalar,RowMajor,Conj, ColMajor>::run);
    func[TR    | (ADJ  << 2)] = (ei_general_matrix_matrix_product<DenseIndex,Scalar,RowMajor,false,Scalar,RowMajor,Conj, ColMajor>::run);
    func[ADJ   | (ADJ  << 2)] = (ei_general_matrix_matrix_product<DenseIndex,Scalar,RowMajor,Conj, Scalar,RowMajor,Conj, ColMajor>::run);
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
  {
    if(beta==Scalar(0)) matrix(c, *m, *n, *ldc).setZero();
    else                matrix(c, *m, *n, *ldc) *= beta;
  }

  ei_gemm_blocking_space<ColMajor,Scalar,Scalar,Dynamic,Dynamic,Dynamic> blocking(*m,*n,*k);

  func[code](*m, *n, *k, a, *lda, b, *ldb, c, *ldc, alpha, blocking, 0);
  return 0;
}

int EIGEN_BLAS_FUNC(trsm)(char *side, char *uplo, char *opa, char *diag, int *m, int *n, RealScalar *palpha,  RealScalar *pa, int *lda, RealScalar *pb, int *ldb)
{
//   std::cerr << "in trsm " << *side << " " << *uplo << " " << *opa << " " << *diag << " " << *m << "," << *n << " " << *palpha << " " << *lda << " " << *ldb<< "\n";
  typedef void (*functype)(DenseIndex, DenseIndex, const Scalar *, DenseIndex, Scalar *, DenseIndex);
  static functype func[32];

  static bool init = false;
  if(!init)
  {
    for(int k=0; k<32; ++k)
      func[k] = 0;

    func[NOTR  | (LEFT  << 2) | (UP << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,DenseIndex,OnTheLeft, Upper|0,          false,ColMajor,ColMajor>::run);
    func[TR    | (LEFT  << 2) | (UP << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,DenseIndex,OnTheLeft, Lower|0,          false,RowMajor,ColMajor>::run);
    func[ADJ   | (LEFT  << 2) | (UP << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,DenseIndex,OnTheLeft, Lower|0,          Conj, RowMajor,ColMajor>::run);

    func[NOTR  | (RIGHT << 2) | (UP << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,DenseIndex,OnTheRight,Upper|0,          false,ColMajor,ColMajor>::run);
    func[TR    | (RIGHT << 2) | (UP << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,DenseIndex,OnTheRight,Lower|0,          false,RowMajor,ColMajor>::run);
    func[ADJ   | (RIGHT << 2) | (UP << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,DenseIndex,OnTheRight,Lower|0,          Conj, RowMajor,ColMajor>::run);

    func[NOTR  | (LEFT  << 2) | (LO << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,DenseIndex,OnTheLeft, Lower|0,          false,ColMajor,ColMajor>::run);
    func[TR    | (LEFT  << 2) | (LO << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,DenseIndex,OnTheLeft, Upper|0,          false,RowMajor,ColMajor>::run);
    func[ADJ   | (LEFT  << 2) | (LO << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,DenseIndex,OnTheLeft, Upper|0,          Conj, RowMajor,ColMajor>::run);

    func[NOTR  | (RIGHT << 2) | (LO << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,DenseIndex,OnTheRight,Lower|0,          false,ColMajor,ColMajor>::run);
    func[TR    | (RIGHT << 2) | (LO << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,DenseIndex,OnTheRight,Upper|0,          false,RowMajor,ColMajor>::run);
    func[ADJ   | (RIGHT << 2) | (LO << 3) | (NUNIT << 4)] = (ei_triangular_solve_matrix<Scalar,DenseIndex,OnTheRight,Upper|0,          Conj, RowMajor,ColMajor>::run);


    func[NOTR  | (LEFT  << 2) | (UP << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,DenseIndex,OnTheLeft, Upper|UnitDiag,false,ColMajor,ColMajor>::run);
    func[TR    | (LEFT  << 2) | (UP << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,DenseIndex,OnTheLeft, Lower|UnitDiag,false,RowMajor,ColMajor>::run);
    func[ADJ   | (LEFT  << 2) | (UP << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,DenseIndex,OnTheLeft, Lower|UnitDiag,Conj, RowMajor,ColMajor>::run);

    func[NOTR  | (RIGHT << 2) | (UP << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,DenseIndex,OnTheRight,Upper|UnitDiag,false,ColMajor,ColMajor>::run);
    func[TR    | (RIGHT << 2) | (UP << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,DenseIndex,OnTheRight,Lower|UnitDiag,false,RowMajor,ColMajor>::run);
    func[ADJ   | (RIGHT << 2) | (UP << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,DenseIndex,OnTheRight,Lower|UnitDiag,Conj, RowMajor,ColMajor>::run);

    func[NOTR  | (LEFT  << 2) | (LO << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,DenseIndex,OnTheLeft, Lower|UnitDiag,false,ColMajor,ColMajor>::run);
    func[TR    | (LEFT  << 2) | (LO << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,DenseIndex,OnTheLeft, Upper|UnitDiag,false,RowMajor,ColMajor>::run);
    func[ADJ   | (LEFT  << 2) | (LO << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,DenseIndex,OnTheLeft, Upper|UnitDiag,Conj, RowMajor,ColMajor>::run);

    func[NOTR  | (RIGHT << 2) | (LO << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,DenseIndex,OnTheRight,Lower|UnitDiag,false,ColMajor,ColMajor>::run);
    func[TR    | (RIGHT << 2) | (LO << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,DenseIndex,OnTheRight,Upper|UnitDiag,false,RowMajor,ColMajor>::run);
    func[ADJ   | (RIGHT << 2) | (LO << 3) | (UNIT  << 4)] = (ei_triangular_solve_matrix<Scalar,DenseIndex,OnTheRight,Upper|UnitDiag,Conj, RowMajor,ColMajor>::run);

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
  typedef void (*functype)(DenseIndex, DenseIndex, DenseIndex, const Scalar *, DenseIndex, const Scalar *, DenseIndex, Scalar *, DenseIndex, Scalar);
  static functype func[32];
  static bool init = false;
  if(!init)
  {
    for(int k=0; k<32; ++k)
      func[k] = 0;

    func[NOTR  | (LEFT  << 2) | (UP << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,DenseIndex,Upper|0,          true, ColMajor,false,ColMajor,false,ColMajor>::run);
    func[TR    | (LEFT  << 2) | (UP << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,DenseIndex,Lower|0,          true, RowMajor,false,ColMajor,false,ColMajor>::run);
    func[ADJ   | (LEFT  << 2) | (UP << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,DenseIndex,Lower|0,          true, RowMajor,Conj, ColMajor,false,ColMajor>::run);

    func[NOTR  | (RIGHT << 2) | (UP << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,DenseIndex,Upper|0,          false,ColMajor,false,ColMajor,false,ColMajor>::run);
    func[TR    | (RIGHT << 2) | (UP << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,DenseIndex,Lower|0,          false,ColMajor,false,RowMajor,false,ColMajor>::run);
    func[ADJ   | (RIGHT << 2) | (UP << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,DenseIndex,Lower|0,          false,ColMajor,false,RowMajor,Conj, ColMajor>::run);

    func[NOTR  | (LEFT  << 2) | (LO << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,DenseIndex,Lower|0,          true, ColMajor,false,ColMajor,false,ColMajor>::run);
    func[TR    | (LEFT  << 2) | (LO << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,DenseIndex,Upper|0,          true, RowMajor,false,ColMajor,false,ColMajor>::run);
    func[ADJ   | (LEFT  << 2) | (LO << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,DenseIndex,Upper|0,          true, RowMajor,Conj, ColMajor,false,ColMajor>::run);

    func[NOTR  | (RIGHT << 2) | (LO << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,DenseIndex,Lower|0,          false,ColMajor,false,ColMajor,false,ColMajor>::run);
    func[TR    | (RIGHT << 2) | (LO << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,DenseIndex,Upper|0,          false,ColMajor,false,RowMajor,false,ColMajor>::run);
    func[ADJ   | (RIGHT << 2) | (LO << 3) | (NUNIT << 4)] = (ei_product_triangular_matrix_matrix<Scalar,DenseIndex,Upper|0,          false,ColMajor,false,RowMajor,Conj, ColMajor>::run);

    func[NOTR  | (LEFT  << 2) | (UP << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,DenseIndex,Upper|UnitDiag,true, ColMajor,false,ColMajor,false,ColMajor>::run);
    func[TR    | (LEFT  << 2) | (UP << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,DenseIndex,Lower|UnitDiag,true, RowMajor,false,ColMajor,false,ColMajor>::run);
    func[ADJ   | (LEFT  << 2) | (UP << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,DenseIndex,Lower|UnitDiag,true, RowMajor,Conj, ColMajor,false,ColMajor>::run);

    func[NOTR  | (RIGHT << 2) | (UP << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,DenseIndex,Upper|UnitDiag,false,ColMajor,false,ColMajor,false,ColMajor>::run);
    func[TR    | (RIGHT << 2) | (UP << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,DenseIndex,Lower|UnitDiag,false,ColMajor,false,RowMajor,false,ColMajor>::run);
    func[ADJ   | (RIGHT << 2) | (UP << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,DenseIndex,Lower|UnitDiag,false,ColMajor,false,RowMajor,Conj, ColMajor>::run);

    func[NOTR  | (LEFT  << 2) | (LO << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,DenseIndex,Lower|UnitDiag,true, ColMajor,false,ColMajor,false,ColMajor>::run);
    func[TR    | (LEFT  << 2) | (LO << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,DenseIndex,Upper|UnitDiag,true, RowMajor,false,ColMajor,false,ColMajor>::run);
    func[ADJ   | (LEFT  << 2) | (LO << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,DenseIndex,Upper|UnitDiag,true, RowMajor,Conj, ColMajor,false,ColMajor>::run);

    func[NOTR  | (RIGHT << 2) | (LO << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,DenseIndex,Lower|UnitDiag,false,ColMajor,false,ColMajor,false,ColMajor>::run);
    func[TR    | (RIGHT << 2) | (LO << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,DenseIndex,Upper|UnitDiag,false,ColMajor,false,RowMajor,false,ColMajor>::run);
    func[ADJ   | (RIGHT << 2) | (LO << 3) | (UNIT  << 4)] = (ei_product_triangular_matrix_matrix<Scalar,DenseIndex,Upper|UnitDiag,false,ColMajor,false,RowMajor,Conj, ColMajor>::run);

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

  if(*m==0 || *n==0)
    return 1;

  // FIXME find a way to avoid this copy
  Matrix<Scalar,Dynamic,Dynamic,ColMajor> tmp = matrix(b,*m,*n,*ldb);
  matrix(b,*m,*n,*ldb).setZero();

  if(SIDE(*side)==LEFT)
    func[code](*m, *n, *m, a, *lda, tmp.data(), tmp.outerStride(), b, *ldb, alpha);
  else
    func[code](*m, *n, *n, tmp.data(), tmp.outerStride(), a, *lda, b, *ldb, alpha);
  return 1;
}

// c = alpha*a*b + beta*c  for side = 'L'or'l'
// c = alpha*b*a + beta*c  for side = 'R'or'r
int EIGEN_BLAS_FUNC(symm)(char *side, char *uplo, int *m, int *n, RealScalar *palpha, RealScalar *pa, int *lda, RealScalar *pb, int *ldb, RealScalar *pbeta, RealScalar *pc, int *ldc)
{
//   std::cerr << "in symm " << *side << " " << *uplo << " " << *m << "x" << *n << " lda:" << *lda << " ldb:" << *ldb << " ldc:" << *ldc << " alpha:" << *palpha << " beta:" << *pbeta << "\n";
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
  {
    if(beta==Scalar(0)) matrix(c, *m, *n, *ldc).setZero();
    else                matrix(c, *m, *n, *ldc) *= beta;
  }

  if(*m==0 || *n==0)
  {
    return 1;
  }

  #if ISCOMPLEX
  // FIXME add support for symmetric complex matrix
  int size = (SIDE(*side)==LEFT) ? (*m) : (*n);
  Matrix<Scalar,Dynamic,Dynamic,ColMajor> matA(size,size);
  if(UPLO(*uplo)==UP)
  {
    matA.triangularView<Upper>() = matrix(a,size,size,*lda);
    matA.triangularView<Lower>() = matrix(a,size,size,*lda).transpose();
  }
  else if(UPLO(*uplo)==LO)
  {
    matA.triangularView<Lower>() = matrix(a,size,size,*lda);
    matA.triangularView<Upper>() = matrix(a,size,size,*lda).transpose();
  }
  if(SIDE(*side)==LEFT)
    matrix(c, *m, *n, *ldc) += alpha * matA * matrix(b, *m, *n, *ldb);
  else if(SIDE(*side)==RIGHT)
    matrix(c, *m, *n, *ldc) += alpha * matrix(b, *m, *n, *ldb) * matA;
  #else
  if(SIDE(*side)==LEFT)
    if(UPLO(*uplo)==UP)       ei_product_selfadjoint_matrix<Scalar, DenseIndex, RowMajor,true,false, ColMajor,false,false, ColMajor>::run(*m, *n, a, *lda, b, *ldb, c, *ldc, alpha);
    else if(UPLO(*uplo)==LO)  ei_product_selfadjoint_matrix<Scalar, DenseIndex, ColMajor,true,false, ColMajor,false,false, ColMajor>::run(*m, *n, a, *lda, b, *ldb, c, *ldc, alpha);
    else                      return 0;
  else if(SIDE(*side)==RIGHT)
    if(UPLO(*uplo)==UP)       ei_product_selfadjoint_matrix<Scalar, DenseIndex, ColMajor,false,false, RowMajor,true,false, ColMajor>::run(*m, *n, b, *ldb, a, *lda, c, *ldc, alpha);
    else if(UPLO(*uplo)==LO)  ei_product_selfadjoint_matrix<Scalar, DenseIndex, ColMajor,false,false, ColMajor,true,false, ColMajor>::run(*m, *n, b, *ldb, a, *lda, c, *ldc, alpha);
    else                      return 0;
  else
    return 0;
  #endif

  return 0;
}

// c = alpha*a*a' + beta*c  for op = 'N'or'n'
// c = alpha*a'*a + beta*c  for op = 'T'or't','C'or'c'
int EIGEN_BLAS_FUNC(syrk)(char *uplo, char *op, int *n, int *k, RealScalar *palpha, RealScalar *pa, int *lda, RealScalar *pbeta, RealScalar *pc, int *ldc)
{
//   std::cerr << "in syrk " << *uplo << " " << *op << " " << *n << " " << *k << " " << *palpha << " " << *lda << " " << *pbeta << " " << *ldc << "\n";
  typedef void (*functype)(DenseIndex, DenseIndex, const Scalar *, DenseIndex, Scalar *, DenseIndex, Scalar);
  static functype func[8];

  static bool init = false;
  if(!init)
  {
    for(int k=0; k<8; ++k)
      func[k] = 0;

    func[NOTR  | (UP << 2)] = (ei_selfadjoint_product<Scalar,DenseIndex,ColMajor,ColMajor,true, Upper>::run);
    func[TR    | (UP << 2)] = (ei_selfadjoint_product<Scalar,DenseIndex,RowMajor,ColMajor,false,Upper>::run);
    func[ADJ   | (UP << 2)] = (ei_selfadjoint_product<Scalar,DenseIndex,RowMajor,ColMajor,false,Upper>::run);

    func[NOTR  | (LO << 2)] = (ei_selfadjoint_product<Scalar,DenseIndex,ColMajor,ColMajor,true, Lower>::run);
    func[TR    | (LO << 2)] = (ei_selfadjoint_product<Scalar,DenseIndex,RowMajor,ColMajor,false,Lower>::run);
    func[ADJ   | (LO << 2)] = (ei_selfadjoint_product<Scalar,DenseIndex,RowMajor,ColMajor,false,Lower>::run);

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
  {
    if(UPLO(*uplo)==UP) matrix(c, *n, *n, *ldc).triangularView<Upper>() *= beta;
    else                matrix(c, *n, *n, *ldc).triangularView<Lower>() *= beta;
  }

  #if ISCOMPLEX
  // FIXME add support for symmetric complex matrix
  if(UPLO(*uplo)==UP)
  {
    if(OP(*op)==NOTR)
      matrix(c, *n, *n, *ldc).triangularView<Upper>() += alpha * matrix(a,*n,*k,*lda) * matrix(a,*n,*k,*lda).transpose();
    else
      matrix(c, *n, *n, *ldc).triangularView<Upper>() += alpha * matrix(a,*k,*n,*lda).transpose() * matrix(a,*k,*n,*lda);
  }
  else
  {
    if(OP(*op)==NOTR)
      matrix(c, *n, *n, *ldc).triangularView<Lower>() += alpha * matrix(a,*n,*k,*lda) * matrix(a,*n,*k,*lda).transpose();
    else
      matrix(c, *n, *n, *ldc).triangularView<Lower>() += alpha * matrix(a,*k,*n,*lda).transpose() * matrix(a,*k,*n,*lda);
  }
  #else
  func[code](*n, *k, a, *lda, c, *ldc, alpha);
  #endif

  return 0;
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

  if(*n<=0 || *k<0)
  {
    return 0;
  }

  if(beta!=Scalar(1))
  {
    if(UPLO(*uplo)==UP) matrix(c, *n, *n, *ldc).triangularView<Upper>() *= beta;
    else                matrix(c, *n, *n, *ldc).triangularView<Lower>() *= beta;
  }

  if(*k==0)
    return 1;

  if(OP(*op)==NOTR)
  {
    if(UPLO(*uplo)==UP)
    {
      matrix(c, *n, *n, *ldc).triangularView<Upper>()
        += alpha *matrix(a, *n, *k, *lda)*matrix(b, *n, *k, *ldb).transpose()
        +  alpha*matrix(b, *n, *k, *ldb)*matrix(a, *n, *k, *lda).transpose();
    }
    else if(UPLO(*uplo)==LO)
      matrix(c, *n, *n, *ldc).triangularView<Lower>()
        += alpha*matrix(a, *n, *k, *lda)*matrix(b, *n, *k, *ldb).transpose()
        +  alpha*matrix(b, *n, *k, *ldb)*matrix(a, *n, *k, *lda).transpose();
  }
  else if(OP(*op)==TR || OP(*op)==ADJ)
  {
    if(UPLO(*uplo)==UP)
      matrix(c, *n, *n, *ldc).triangularView<Upper>()
        += alpha*matrix(a, *k, *n, *lda).transpose()*matrix(b, *k, *n, *ldb)
        +  alpha*matrix(b, *k, *n, *ldb).transpose()*matrix(a, *k, *n, *lda);
    else if(UPLO(*uplo)==LO)
      matrix(c, *n, *n, *ldc).triangularView<Lower>()
        += alpha*matrix(a, *k, *n, *lda).transpose()*matrix(b, *k, *n, *ldb)
        +  alpha*matrix(b, *k, *n, *ldb).transpose()*matrix(a, *k, *n, *lda);
  }

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

//   std::cerr << "in hemm " << *side << " " << *uplo << " " << *m << " " << *n << " " << alpha << " " << *lda << " " << beta << " " << *ldc << "\n";

  if(*m<0 || *n<0)
  {
    return 0;
  }

  if(beta==Scalar(0))
    matrix(c, *m, *n, *ldc).setZero();
  else if(beta!=Scalar(1))
    matrix(c, *m, *n, *ldc) *= beta;
  
  if(*m==0 || *n==0)
  {
    return 1;
  }

  if(SIDE(*side)==LEFT)
  {
    if(UPLO(*uplo)==UP)       ei_product_selfadjoint_matrix<Scalar,DenseIndex,RowMajor,true,Conj,  ColMajor,false,false, ColMajor>
                                ::run(*m, *n, a, *lda, b, *ldb, c, *ldc, alpha);
    else if(UPLO(*uplo)==LO)  ei_product_selfadjoint_matrix<Scalar,DenseIndex,ColMajor,true,false, ColMajor,false,false, ColMajor>
                                ::run(*m, *n, a, *lda, b, *ldb, c, *ldc, alpha);
    else                      return 0;
  }
  else if(SIDE(*side)==RIGHT)
  {
    if(UPLO(*uplo)==UP)       matrix(c,*m,*n,*ldc) += alpha * matrix(b,*m,*n,*ldb) * matrix(a,*n,*n,*lda).selfadjointView<Upper>();/*ei_product_selfadjoint_matrix<Scalar,DenseIndex,ColMajor,false,false, RowMajor,true,Conj,  ColMajor>
                                ::run(*m, *n, b, *ldb, a, *lda, c, *ldc, alpha);*/
    else if(UPLO(*uplo)==LO)  ei_product_selfadjoint_matrix<Scalar,DenseIndex,ColMajor,false,false, ColMajor,true,false, ColMajor>
                                ::run(*m, *n, b, *ldb, a, *lda, c, *ldc, alpha);
    else                      return 0;
  }
  else
  {
    return 0;
  }

  return 0;
}

// c = alpha*a*conj(a') + beta*c  for op = 'N'or'n'
// c = alpha*conj(a')*a + beta*c  for op  = 'C'or'c'
int EIGEN_BLAS_FUNC(herk)(char *uplo, char *op, int *n, int *k, RealScalar *palpha, RealScalar *pa, int *lda, RealScalar *pbeta, RealScalar *pc, int *ldc)
{
  typedef void (*functype)(DenseIndex, DenseIndex, const Scalar *, DenseIndex, Scalar *, DenseIndex, Scalar);
  static functype func[8];

  static bool init = false;
  if(!init)
  {
    for(int k=0; k<8; ++k)
      func[k] = 0;

    func[NOTR  | (UP << 2)] = (ei_selfadjoint_product<Scalar,DenseIndex,ColMajor,ColMajor,true, Upper>::run);
    func[ADJ   | (UP << 2)] = (ei_selfadjoint_product<Scalar,DenseIndex,RowMajor,ColMajor,false,Upper>::run);

    func[NOTR  | (LO << 2)] = (ei_selfadjoint_product<Scalar,DenseIndex,ColMajor,ColMajor,true, Lower>::run);
    func[ADJ   | (LO << 2)] = (ei_selfadjoint_product<Scalar,DenseIndex,RowMajor,ColMajor,false,Lower>::run);

    init = true;
  }

  Scalar* a = reinterpret_cast<Scalar*>(pa);
  Scalar* c = reinterpret_cast<Scalar*>(pc);
  RealScalar alpha = *palpha;
  RealScalar beta  = *pbeta;

//   std::cerr << "in herk " << *uplo << " " << *op << " " << *n << " " << *k << " " << alpha << " " << *lda << " " << beta << " " << *ldc << "\n";

  if(*n<0 || *k<0)
  {
    return 0;
  }

  int code = OP(*op) | (UPLO(*uplo) << 2);
  if(code>=8 || func[code]==0)
    return 0;

  if(beta!=RealScalar(1))
  {
    if(UPLO(*uplo)==UP) matrix(c, *n, *n, *ldc).triangularView<StrictlyUpper>() *= beta;
    else                matrix(c, *n, *n, *ldc).triangularView<StrictlyLower>() *= beta;

    matrix(c, *n, *n, *ldc).diagonal().real() *= beta;
    matrix(c, *n, *n, *ldc).diagonal().imag().setZero();
  }

  if(*k>0 && alpha!=RealScalar(0))
  {
    func[code](*n, *k, a, *lda, c, *ldc, alpha);
    matrix(c, *n, *n, *ldc).diagonal().imag().setZero();
  }
  return 0;
}

// c = alpha*a*conj(b') + conj(alpha)*b*conj(a') + beta*c,  for op = 'N'or'n'
// c = alpha*conj(a')*b + conj(alpha)*conj(b')*a + beta*c,  for op = 'C'or'c'
int EIGEN_BLAS_FUNC(her2k)(char *uplo, char *op, int *n, int *k, RealScalar *palpha, RealScalar *pa, int *lda, RealScalar *pb, int *ldb, RealScalar *pbeta, RealScalar *pc, int *ldc)
{
  Scalar* a = reinterpret_cast<Scalar*>(pa);
  Scalar* b = reinterpret_cast<Scalar*>(pb);
  Scalar* c = reinterpret_cast<Scalar*>(pc);
  Scalar alpha = *reinterpret_cast<Scalar*>(palpha);
  RealScalar beta  = *pbeta;

  if(*n<=0 || *k<0)
  {
    return 0;
  }

  if(beta!=RealScalar(1))
  {
    if(UPLO(*uplo)==UP) matrix(c, *n, *n, *ldc).triangularView<StrictlyUpper>() *= beta;
    else                matrix(c, *n, *n, *ldc).triangularView<StrictlyLower>() *= beta;

    matrix(c, *n, *n, *ldc).diagonal().real() *= beta;
    matrix(c, *n, *n, *ldc).diagonal().imag().setZero();
  }
  else if(*k>0 && alpha!=Scalar(0))
    matrix(c, *n, *n, *ldc).diagonal().imag().setZero();

  if(*k==0)
    return 1;

  if(OP(*op)==NOTR)
  {
    if(UPLO(*uplo)==UP)
    {
      matrix(c, *n, *n, *ldc).triangularView<Upper>()
        +=         alpha *matrix(a, *n, *k, *lda)*matrix(b, *n, *k, *ldb).adjoint()
        +  ei_conj(alpha)*matrix(b, *n, *k, *ldb)*matrix(a, *n, *k, *lda).adjoint();
    }
    else if(UPLO(*uplo)==LO)
      matrix(c, *n, *n, *ldc).triangularView<Lower>()
        += alpha*matrix(a, *n, *k, *lda)*matrix(b, *n, *k, *ldb).adjoint()
        +  ei_conj(alpha)*matrix(b, *n, *k, *ldb)*matrix(a, *n, *k, *lda).adjoint();
  }
  else if(OP(*op)==ADJ)
  {
    if(UPLO(*uplo)==UP)
      matrix(c, *n, *n, *ldc).triangularView<Upper>()
        += alpha*matrix(a, *k, *n, *lda).adjoint()*matrix(b, *k, *n, *ldb)
        +  ei_conj(alpha)*matrix(b, *k, *n, *ldb).adjoint()*matrix(a, *k, *n, *lda);
    else if(UPLO(*uplo)==LO)
      matrix(c, *n, *n, *ldc).triangularView<Lower>()
        += alpha*matrix(a, *k, *n, *lda).adjoint()*matrix(b, *k, *n, *ldb)
        +  ei_conj(alpha)*matrix(b, *k, *n, *ldb).adjoint()*matrix(a, *k, *n, *lda);
  }

  return 1;
}

#endif // ISCOMPLEX
