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

#ifdef EIGEN_UMFPACK_SUPPORT
#include <Eigen/UmfPackSupport>
#endif

#ifdef EIGEN_SUPERLU_SUPPORT
#include <Eigen/SuperLUSupport>
#endif


template<typename Scalar> void sparse_lu_legacy(int rows, int cols)
{
  double density = (std::max)(8./(rows*cols), 0.01);
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;
  typedef Matrix<Scalar,Dynamic,1> DenseVector;

  DenseVector vec1 = DenseVector::Random(rows);

  std::vector<Vector2i> zeroCoords;
  std::vector<Vector2i> nonzeroCoords;

    SparseMatrix<Scalar> m2(rows, cols);
    DenseMatrix refMat2(rows, cols);

    DenseVector b = DenseVector::Random(cols);
    DenseVector refX(cols), x(cols);

    initSparse<Scalar>(density, refMat2, m2, ForceNonZeroDiag, &zeroCoords, &nonzeroCoords);

    FullPivLU<DenseMatrix> refLu(refMat2);
    refX = refLu.solve(b);
    #if defined(EIGEN_SUPERLU_SUPPORT) || defined(EIGEN_UMFPACK_SUPPORT)
    Scalar refDet = refLu.determinant();
    #endif
    x.setZero();
    // // SparseLU<SparseMatrix<Scalar> > (m2).solve(b,&x);
    // // VERIFY(refX.isApprox(x,test_precision<Scalar>()) && "LU: default");
    
    #ifdef EIGEN_UMFPACK_SUPPORT
    {
      // check solve
      x.setZero();
      SparseLU<SparseMatrix<Scalar>,UmfPack> lu(m2);
      VERIFY(lu.succeeded() && "umfpack LU decomposition failed");
      VERIFY(lu.solve(b,&x) && "umfpack LU solving failed");
      VERIFY(refX.isApprox(x,test_precision<Scalar>()) && "LU: umfpack");
      VERIFY_IS_APPROX(refDet,lu.determinant());
        // TODO check the extracted data
        //std::cerr << slu.matrixL() << "\n";
    }
    #endif
    
    #ifdef EIGEN_SUPERLU_SUPPORT
    // legacy, deprecated API
    {
      x.setZero();
      SparseLU<SparseMatrix<Scalar>,SuperLULegacy> slu(m2);
      if (slu.succeeded())
      {
        DenseVector oldb = b;
        if (slu.solve(b,&x)) {
          VERIFY(refX.isApprox(x,test_precision<Scalar>()) && "LU: SuperLU");
        }
        else
          std::cerr << "super lu solving failed\n";
        VERIFY(oldb.isApprox(b) && "the rhs should not be modified!");
        
        // std::cerr << refDet << " == " << slu.determinant() << "\n";
        if (slu.solve(b, &x, SvTranspose)) {
          VERIFY(b.isApprox(m2.transpose() * x, test_precision<Scalar>()));
        }
        else
          std::cerr << "super lu solving failed\n";

        if (slu.solve(b, &x, SvAdjoint)) {
         VERIFY(b.isApprox(m2.adjoint() * x, test_precision<Scalar>()));
        }
        else
          std::cerr << "super lu solving failed\n";

        if (!NumTraits<Scalar>::IsComplex) {
          VERIFY_IS_APPROX(refDet,slu.determinant()); // FIXME det is not very stable for complex
        }
      }
      else
        std::cerr << "super lu factorize failed\n";
    }
    #endif
    
}

void test_sparse_lu_legacy()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(sparse_lu_legacy<double>(8, 8) );
    int s = internal::random<int>(1,300);
    CALL_SUBTEST_1(sparse_lu_legacy<std::complex<double> >(s,s) );
    CALL_SUBTEST_1(sparse_lu_legacy<double>(s,s) );
  }
}
