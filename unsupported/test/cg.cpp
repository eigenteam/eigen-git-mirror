// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <gael.guennebaud@inria.fr>
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

#include "sparse.h"
#include <Eigen/IterativeSolvers>

template<typename Scalar,typename Index> void cg(int size)
{
  double density = (std::max)(8./(size*size), 0.01);
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;
  typedef Matrix<Scalar,Dynamic,1> DenseVector;
  typedef SparseMatrix<Scalar,ColMajor,Index> SparseMatrixType;  

  SparseMatrixType m2(size,size);
  DenseMatrix refMat2(size,size);

  DenseVector b = DenseVector::Random(size);
  DenseVector ref_x(size), x(size);

  initSparse<Scalar>(density, refMat2, m2, ForceNonZeroDiag|MakeLowerTriangular, 0, 0);
//   for(int i=0; i<rows; ++i)
//     m2.coeffRef(i,i) = refMat2(i,i) = internal::abs(internal::real(refMat2(i,i)));
    
  SparseMatrixType m3 = m2 * m2.adjoint(), m3_lo(size,size), m3_up(size,size);
  DenseMatrix refMat3 = refMat2 * refMat2.adjoint();

  m3_lo.template selfadjointView<Lower>().rankUpdate(m2,0);
  m3_up.template selfadjointView<Upper>().rankUpdate(m2,0);

  ref_x = refMat3.template selfadjointView<Lower>().llt().solve(b);

  x = ConjugateGradient<SparseMatrixType, Lower>().compute(m3).solve(b);
  VERIFY(ref_x.isApprox(x,test_precision<Scalar>()) && "ConjugateGradient: solve, full storage, lower");

  x.setRandom();
  x = ConjugateGradient<SparseMatrixType, Lower>().compute(m3).solveWithGuess(b,x);
  VERIFY(ref_x.isApprox(x,test_precision<Scalar>()) && "ConjugateGradient: solveWithGuess, full storage, lower");

  x = ConjugateGradient<SparseMatrixType, Upper>().compute(m3).solve(b);
  VERIFY(ref_x.isApprox(x,test_precision<Scalar>()) && "ConjugateGradient: solve, full storage, upper, single dense rhs");

  x = ConjugateGradient<SparseMatrixType, Lower>(m3_lo).solve(b);
  VERIFY(ref_x.isApprox(x,test_precision<Scalar>()) && "ConjugateGradient: solve, lower only, single dense rhs");

  x = ConjugateGradient<SparseMatrixType, Upper>(m3_up).solve(b);
  VERIFY(ref_x.isApprox(x,test_precision<Scalar>()) && "ConjugateGradient: solve, upper only, single dense rhs");



  x = ConjugateGradient<SparseMatrixType, Lower, IdentityPreconditioner>().compute(m3).solve(b);
  VERIFY(ref_x.isApprox(x,test_precision<Scalar>()) && "ConjugateGradient: solve, full storage, lower");

  x = ConjugateGradient<SparseMatrixType, Upper, IdentityPreconditioner>().compute(m3).solve(b);
  VERIFY(ref_x.isApprox(x,test_precision<Scalar>()) && "ConjugateGradient: solve, full storage, upper, single dense rhs");

  x = ConjugateGradient<SparseMatrixType, Lower, IdentityPreconditioner>(m3_lo).solve(b);
  VERIFY(ref_x.isApprox(x,test_precision<Scalar>()) && "ConjugateGradient: solve, lower only, single dense rhs");

  x = ConjugateGradient<SparseMatrixType, Upper, IdentityPreconditioner>(m3_up).solve(b);
  VERIFY(ref_x.isApprox(x,test_precision<Scalar>()) && "ConjugateGradient: solve, upper only, single dense rhs");
}

void test_cg()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( (cg<double,int>(8)) );
    CALL_SUBTEST_1( (cg<double,long int>(8)) );
    CALL_SUBTEST_2( (cg<std::complex<double>,int>(internal::random<int>(1,300))) );
    CALL_SUBTEST_1( (cg<double,int>(internal::random<int>(1,300))) );
  }
}
