// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <g.gael@free.fr>
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
#include <Eigen/SparseExtra>

#ifdef EIGEN_CHOLMOD_SUPPORT
#include <Eigen/CholmodSupport>
#endif

#ifdef EIGEN_TAUCS_SUPPORT
#include <Eigen/TaucsSupport>
#endif

template<typename Scalar> void sparse_ldlt(int rows, int cols)
{
  double density = std::max(8./(rows*cols), 0.01);
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;
  typedef Matrix<Scalar,Dynamic,1> DenseVector;

  SparseMatrix<Scalar> m2(rows, cols);
  DenseMatrix refMat2(rows, cols);

  DenseVector b = DenseVector::Random(cols);
  DenseVector refX(cols), x(cols);

  initSparse<Scalar>(density, refMat2, m2, ForceNonZeroDiag|MakeUpperTriangular, 0, 0);
  for(int i=0; i<rows; ++i)
    m2.coeffRef(i,i) = refMat2(i,i) = ei_abs(ei_real(refMat2(i,i)));

  refX = refMat2.template selfadjointView<Upper>().ldlt().solve(b);
  typedef SparseMatrix<Scalar,Upper|SelfAdjoint> SparseSelfAdjointMatrix;
  x = b;
  SparseLDLT<SparseSelfAdjointMatrix> ldlt(m2);
  if (ldlt.succeeded())
    ldlt.solveInPlace(x);
  else
    std::cerr << "warning LDLT failed\n";

  VERIFY_IS_APPROX(refMat2.template selfadjointView<Upper>() * x, b);
  VERIFY(refX.isApprox(x,test_precision<Scalar>()) && "LDLT: default");

#ifdef EIGEN_CHOLMOD_SUPPORT
  x = b;
  SparseLDLT<SparseSelfAdjointMatrix, Cholmod> ldlt2(m2);
  if (ldlt2.succeeded())
    ldlt2.solveInPlace(x);
  else
    std::cerr << "warning LDLT failed\n";

  VERIFY_IS_APPROX(refMat2.template selfadjointView<Upper>() * x, b);
  VERIFY(refX.isApprox(x,test_precision<Scalar>()) && "LDLT: cholmod solveInPlace");


  SparseLDLT<SparseSelfAdjointMatrix, Cholmod> ldlt3(m2);
  if (ldlt3.succeeded())
    x = ldlt3.solve(b);
  else
    std::cerr << "warning LDLT failed\n";

  VERIFY_IS_APPROX(refMat2.template selfadjointView<Upper>() * x, b);
  VERIFY(refX.isApprox(x,test_precision<Scalar>()) && "LDLT: cholmod solve");

#endif
}

void test_sparse_ldlt()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(sparse_ldlt<double>(8, 8) );
    int s = ei_random<int>(1,300);
    CALL_SUBTEST_2(sparse_ldlt<std::complex<double> >(s,s) );
    CALL_SUBTEST_1(sparse_ldlt<double>(s,s) );
  }
}
