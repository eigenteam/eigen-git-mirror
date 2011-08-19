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

template<typename Scalar,typename Index> void sparse_llt(int rows, int cols)
{
  double density = (std::max)(8./(rows*cols), 0.01);
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;
  typedef Matrix<Scalar,Dynamic,1> DenseVector;
  typedef SparseMatrix<Scalar,ColMajor,Index> SparseMatrixType;

    // TODO fix the issue with complex (see SparseLLT::solveInPlace)
    SparseMatrixType m2(rows, cols);
    DenseMatrix refMat2(rows, cols);

    DenseVector b = DenseVector::Random(cols);
    DenseVector ref_x(cols), x(cols);
    DenseMatrix B = DenseMatrix::Random(rows,cols);
    DenseMatrix ref_X(rows,cols), X(rows,cols);

    initSparse<Scalar>(density, refMat2, m2, ForceNonZeroDiag|MakeLowerTriangular, 0, 0);

    for(int i=0; i<rows; ++i)
      m2.coeffRef(i,i) = refMat2(i,i) = internal::abs(internal::real(refMat2(i,i)));

    ref_x = refMat2.template selfadjointView<Lower>().llt().solve(b);
    if (!NumTraits<Scalar>::IsComplex)
    {
      x = b;
      SparseLLT<SparseMatrixType > (m2).solveInPlace(x);
      VERIFY(ref_x.isApprox(x,test_precision<Scalar>()) && "LLT: default");
    }
        
#ifdef EIGEN_CHOLMOD_SUPPORT
    // legacy API
    {
      // Cholmod, as configured in CholmodSupport.h, only supports self-adjoint matrices
      SparseMatrixType m3 = m2.adjoint()*m2;
      DenseMatrix refMat3 = refMat2.adjoint()*refMat2;
      
      ref_x = refMat3.template selfadjointView<Lower>().llt().solve(b);
      
      x = b;
      SparseLLT<SparseMatrixType, Cholmod>(m3).solveInPlace(x);
      VERIFY((m3*x).isApprox(b,test_precision<Scalar>()) && "LLT legacy: cholmod solveInPlace");
      
      x = SparseLLT<SparseMatrixType, Cholmod>(m3).solve(b);
      VERIFY(ref_x.isApprox(x,test_precision<Scalar>()) && "LLT legacy: cholmod solve");
    }
    
    // new API
    {
      // Cholmod, as configured in CholmodSupport.h, only supports self-adjoint matrices
      SparseMatrixType m3 = m2 * m2.adjoint(), m3_lo(rows,rows), m3_up(rows,rows);
      DenseMatrix refMat3 = refMat2 * refMat2.adjoint();
      
      m3_lo.template selfadjointView<Lower>().rankUpdate(m2,0);
      m3_up.template selfadjointView<Upper>().rankUpdate(m2,0);
      
      // with a single vector as the rhs
      ref_x = refMat3.template selfadjointView<Lower>().llt().solve(b);

      x = CholmodDecomposition<SparseMatrixType, Lower>(m3).solve(b);
      VERIFY(ref_x.isApprox(x,test_precision<Scalar>()) && "LLT: cholmod solve, single dense rhs");
      
      x = CholmodDecomposition<SparseMatrixType, Upper>(m3).solve(b);
      VERIFY(ref_x.isApprox(x,test_precision<Scalar>()) && "LLT: cholmod solve, single dense rhs");
      
      x = CholmodDecomposition<SparseMatrixType, Lower>(m3_lo).solve(b);
      VERIFY(ref_x.isApprox(x,test_precision<Scalar>()) && "LLT: cholmod solve, single dense rhs");
      
      x = CholmodDecomposition<SparseMatrixType, Upper>(m3_up).solve(b);
      VERIFY(ref_x.isApprox(x,test_precision<Scalar>()) && "LLT: cholmod solve, single dense rhs");
      
      
      // with multiple rhs
      ref_X = refMat3.template selfadjointView<Lower>().llt().solve(B);

      #ifndef EIGEN_DEFAULT_TO_ROW_MAJOR
      // TODO make sure the API is properly documented about this fact
      X = CholmodDecomposition<SparseMatrixType, Lower>(m3).solve(B);
      VERIFY(ref_X.isApprox(X,test_precision<Scalar>()) && "LLT: cholmod solve, multiple dense rhs");
      
      X = CholmodDecomposition<SparseMatrixType, Upper>(m3).solve(B);
      VERIFY(ref_X.isApprox(X,test_precision<Scalar>()) && "LLT: cholmod solve, multiple dense rhs");
      #endif
      
      
      // with a sparse rhs
      SparseMatrixType spB(rows,cols), spX(rows,cols);
      B.diagonal().array() += 1;
      spB = B.sparseView(0.5,1);
      
      ref_X = refMat3.template selfadjointView<Lower>().llt().solve(DenseMatrix(spB));

      spX = CholmodDecomposition<SparseMatrixType, Lower>(m3).solve(spB);
      VERIFY(ref_X.isApprox(spX.toDense(),test_precision<Scalar>()) && "LLT: cholmod solve, multiple sparse rhs");
      
      spX = CholmodDecomposition<SparseMatrixType, Upper>(m3).solve(spB);
      VERIFY(ref_X.isApprox(spX.toDense(),test_precision<Scalar>()) && "LLT: cholmod solve, multiple sparse rhs");
    }
#endif

}

void test_sparse_llt()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( (sparse_llt<double,int>(8, 8)) );
    int s = internal::random<int>(1,300);
    CALL_SUBTEST_2( (sparse_llt<std::complex<double>,int>(s,s)) );
    CALL_SUBTEST_1( (sparse_llt<double,int>(s,s)) );
    CALL_SUBTEST_1( (sparse_llt<double,long int>(s,s)) );
  }
}
