// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
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
#include <Eigen/Cholesky>
#include <Eigen/LU>

template<typename MatrixType> void cholesky(const MatrixType& m)
{
  /* this test covers the following files:
     Cholesky.h CholeskyWithoutSquareRoot.h
  */
  int rows = m.rows();
  int cols = m.cols();

  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> SquareMatrixType;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;

  MatrixType a = test_random_matrix<MatrixType>(rows,cols);
  VectorType vecB = test_random_matrix<VectorType>(rows);
  MatrixType matB = test_random_matrix<MatrixType>(rows,cols);
  SquareMatrixType covMat =  a * a.adjoint();

  if (rows>1)
  {
    CholeskyWithoutSquareRoot<SquareMatrixType> cholnosqrt(covMat);
    VERIFY_IS_APPROX(covMat, cholnosqrt.matrixL() * cholnosqrt.vectorD().asDiagonal() * cholnosqrt.matrixL().adjoint());
  //   cout << (covMat * cholnosqrt.solve(vecB)).transpose().format(6) << endl;
  //   cout << vecB.transpose().format(6) << endl << "----------" << endl;
    VERIFY((covMat * cholnosqrt.solve(vecB)).isApprox(vecB, test_precision<RealScalar>()*RealScalar(100))); // FIXME
    VERIFY((covMat * cholnosqrt.solve(matB)).isApprox(matB, test_precision<RealScalar>()*RealScalar(100))); // FIXME
  }

  Cholesky<SquareMatrixType> chol(covMat);
  VERIFY_IS_APPROX(covMat, chol.matrixL() * chol.matrixL().adjoint());
//   cout << (covMat * chol.solve(vecB)).transpose().format(6) << endl;
//   cout << vecB.transpose().format(6) << endl << "----------" << endl;
  VERIFY((covMat * chol.solve(vecB)).isApprox(vecB, test_precision<RealScalar>()*RealScalar(100))); // FIXME
  VERIFY((covMat * chol.solve(matB)).isApprox(matB, test_precision<RealScalar>()*RealScalar(100))); // FIXME
}

void test_cholesky()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST( cholesky(Matrix<float,1,1>()) );
    CALL_SUBTEST( cholesky(Matrix<float,2,2>()) );
//     CALL_SUBTEST( cholesky(Matrix3f()) );
//     CALL_SUBTEST( cholesky(Matrix4d()) );
//     CALL_SUBTEST( cholesky(MatrixXcd(7,7)) );
//     CALL_SUBTEST( cholesky(MatrixXf(19,19)) );
//     CALL_SUBTEST( cholesky(MatrixXd(33,33)) );
  }
}
