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

template<typename Scalar> void sparse_llt(int rows, int cols)
{
  double density = std::max(8./(rows*cols), 0.01);
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;
  typedef Matrix<Scalar,Dynamic,1> DenseVector;

    // TODO fix the issue with complex (see SparseLLT::solveInPlace)
    SparseMatrix<Scalar> m2(rows, cols);
    DenseMatrix refMat2(rows, cols);

    DenseVector b = DenseVector::Random(cols);
    DenseVector refX(cols), x(cols);

    initSparse<Scalar>(density, refMat2, m2, ForceNonZeroDiag|MakeLowerTriangular, 0, 0);

    for(int i=0; i<rows; ++i)
      m2.coeffRef(i,i) = refMat2(i,i) = internal::abs(internal::real(refMat2(i,i)));

    refX = refMat2.template selfadjointView<Lower>().llt().solve(b);
    if (!NumTraits<Scalar>::IsComplex)
    {
      x = b;
      SparseLLT<SparseMatrix<Scalar> > (m2).solveInPlace(x);
      VERIFY(refX.isApprox(x,test_precision<Scalar>()) && "LLT: default");
    }
        
#ifdef EIGEN_CHOLMOD_SUPPORT
    // legacy API
    {
      // Cholmod, as configured in CholmodSupport.h, only supports self-adjoint matrices
      SparseMatrix<Scalar> m3 = m2.adjoint()*m2;
      DenseMatrix refMat3 = refMat2.adjoint()*refMat2;
      
      refX = refMat3.template selfadjointView<Lower>().llt().solve(b);
      
      x = b;
      SparseLLT<SparseMatrix<Scalar>, Cholmod>(m3).solveInPlace(x);
      VERIFY((m3*x).isApprox(b,test_precision<Scalar>()) && "LLT legacy: cholmod solveInPlace");
      
      x = SparseLLT<SparseMatrix<Scalar>, Cholmod>(m3).solve(b);
      VERIFY(refX.isApprox(x,test_precision<Scalar>()) && "LLT legacy: cholmod solve");
    }
    
    // new API
    {
      // Cholmod, as configured in CholmodSupport.h, only supports self-adjoint matrices
      SparseMatrix<Scalar> m3 = m2.adjoint()*m2;
      DenseMatrix refMat3 = refMat2.adjoint()*refMat2;
      
      refX = refMat3.template selfadjointView<Lower>().llt().solve(b);

      x = CholmodDecomposition<SparseMatrix<Scalar>, Lower>(m3).solve(b);
      VERIFY(refX.isApprox(x,test_precision<Scalar>()) && "LLT: cholmod solve");
      
      x = CholmodDecomposition<SparseMatrix<Scalar>, Upper>(m3).solve(b);
      VERIFY(refX.isApprox(x,test_precision<Scalar>()) && "LLT: cholmod solve");
    }
#endif

}

void test_sparse_llt()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(sparse_llt<double>(8, 8) );
    int s = internal::random<int>(1,300);
    CALL_SUBTEST_2(sparse_llt<std::complex<double> >(s,s) );
    CALL_SUBTEST_1(sparse_llt<double>(s,s) );
  }
}
