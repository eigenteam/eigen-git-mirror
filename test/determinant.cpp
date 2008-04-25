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

namespace Eigen {

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

  MatrixType d(rows, cols);

  // build a ill-conditionned matrix with a nul determinant
  d.col(0).setOnes();
  d.block(0,1, rows, cols-2).setRandom();
  d.col(cols-1).setOnes();

  for (int i=0 ; i<rows ; ++i)
    d.row(i).block(0,1,1,cols-2) = d.row(i).block(0,1,1,cols-2).normalized();

  SquareMatrixType covarianceMatrix = d.transpose() * d;

//   std::cout << covarianceMatrix << "\n" << covarianceMatrix.determinant() << "\n";

  VERIFY_IS_APPROX(covarianceMatrix.determinant(), Scalar(0));
}

void EigenTest::testDeterminant()
{
  for(int i = 0; i < m_repeat; i++) {
    nullDeterminant(Matrix<float, 30, 3>());
    nullDeterminant(Matrix<double, 30, 3>());
    nullDeterminant(Matrix<float, 20, 4>());
    nullDeterminant(Matrix<double, 20, 4>());
//     nullDeterminant(MatrixXd(20,4));
  }
}

} // namespace Eigen
