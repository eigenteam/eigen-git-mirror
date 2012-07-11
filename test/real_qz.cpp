// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Alexey Korepanov <kaikaikai@yandex.ru>
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

#include "main.h"
#include <limits>
#include <Eigen/Eigenvalues>

template<typename MatrixType> void real_qz(const MatrixType& m)
{
  /* this test covers the following files:
     RealQZ.h
  */
  
  typedef typename MatrixType::Index Index;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;
  typedef Matrix<RealScalar, MatrixType::RowsAtCompileTime, 1> RealVectorType;
  typedef typename std::complex<typename NumTraits<typename MatrixType::Scalar>::Real> Complex;
  
  Index dim = m.cols();
  
  MatrixType A = MatrixType::Random(dim,dim),
             B = MatrixType::Random(dim,dim);

  RealQZ<MatrixType> qz(A,B);
  
  VERIFY_IS_EQUAL(qz.info(), Success);
  // check for zeros
  bool all_zeros = true;
  for (Index i=0; i<A.cols(); i++)
    for (Index j=0; j<i; j++) {
      if (internal::abs(qz.matrixT()(i,j))!=Scalar(0.0))
        all_zeros = false;
      if (j<i-1 && internal::abs(qz.matrixS()(i,j))!=Scalar(0.0))
        all_zeros = false;
      if (j==i-1 && j>0 && internal::abs(qz.matrixS()(i,j))!=Scalar(0.0) && internal::abs(qz.matrixS()(i-1,j-1))!=Scalar(0.0))
        all_zeros = false;
    }
  VERIFY_IS_EQUAL(all_zeros, true);
  VERIFY_IS_APPROX(qz.matrixQ()*qz.matrixS()*qz.matrixZ(), A);
  VERIFY_IS_APPROX(qz.matrixQ()*qz.matrixT()*qz.matrixZ(), B);
  VERIFY_IS_APPROX(qz.matrixQ()*qz.matrixQ().adjoint(), MatrixType::Identity(dim,dim));
  VERIFY_IS_APPROX(qz.matrixZ()*qz.matrixZ().adjoint(), MatrixType::Identity(dim,dim));
}

void test_real_qz()
{
  int s;
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( real_qz(Matrix4f()) );
    s = internal::random<int>(1,EIGEN_TEST_MAX_SIZE/4);
    CALL_SUBTEST_2( real_qz(MatrixXd(s,s)) );

    // some trivial but implementation-wise tricky cases
    CALL_SUBTEST_2( real_qz(MatrixXd(1,1)) );
    CALL_SUBTEST_2( real_qz(MatrixXd(2,2)) );
    CALL_SUBTEST_3( real_qz(Matrix<double,1,1>()) );
    CALL_SUBTEST_4( real_qz(Matrix2d()) );
  }
  
  EIGEN_UNUSED_VARIABLE(s)
}
