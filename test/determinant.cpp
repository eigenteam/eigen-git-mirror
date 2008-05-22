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
#include <Eigen/LU>

template<typename MatrixType> void nullDeterminant(const MatrixType& m)
{
  /* this test covers the following files:
     Determinant.h
  */
  int rows = m.rows();
  int cols = m.cols();

  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, MatrixType::ColsAtCompileTime> SquareMatrixType;
  typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, 1> VectorType;

  MatrixType dinv(rows, cols), dnotinv(rows, cols);

  dinv.col(0).setOnes();
  dinv.block(0,1, rows, cols-2).setRandom();

  dnotinv.col(0).setOnes();
  dnotinv.block(0,1, rows, cols-2).setRandom();
  dnotinv.col(cols-1).setOnes();

  for (int i=0 ; i<rows ; ++i)
  {
    dnotinv.row(i).block(0,1,1,cols-2) = ei_random<Scalar>(99.999999,100.00000001)*dnotinv.row(i).block(0,1,1,cols-2).normalized();
    dnotinv(i,cols-1) = dnotinv.row(i).block(0,1,1,cols-2).norm2();
    dinv(i,cols-1) = dinv.row(i).block(0,1,1,cols-2).norm2();
  }

  SquareMatrixType invertibleCovarianceMatrix = dinv.transpose() * dinv;
  SquareMatrixType notInvertibleCovarianceMatrix = dnotinv.transpose() * dnotinv;

  std::cout << notInvertibleCovarianceMatrix << "\n" << notInvertibleCovarianceMatrix.determinant() << "\n";

  VERIFY_IS_MUCH_SMALLER_THAN(notInvertibleCovarianceMatrix.determinant(),
                              notInvertibleCovarianceMatrix.cwiseAbs().maxCoeff());

  VERIFY(invertibleCovarianceMatrix.inverse().exists());

  VERIFY(!notInvertibleCovarianceMatrix.inverse().exists());
}

void test_determinant()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST( nullDeterminant(Matrix<float, 30, 3>()) );
    CALL_SUBTEST( nullDeterminant(Matrix<double, 30, 3>()) );
    CALL_SUBTEST( nullDeterminant(Matrix<float, 20, 4>()) );
    CALL_SUBTEST( nullDeterminant(Matrix<double, 20, 4>()) );
//     CALL_SUBTEST( nullDeterminant(MatrixXd(20,4));
  }
}
