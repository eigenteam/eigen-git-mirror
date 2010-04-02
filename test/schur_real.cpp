// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>
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
#include <Eigen/Eigenvalues>

template<typename MatrixType> void verifyIsQuasiTriangular(const MatrixType& T)
{
  const int size = T.cols();
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;

  // The "zeros" in the real Schur decomposition are only approximately zero
  RealScalar norm = T.norm();

  // Check T is lower Hessenberg
  for(int row = 2; row < size; ++row) {
    for(int col = 0; col < row - 1; ++col) {
      VERIFY_IS_MUCH_SMALLER_THAN(T(row,col), norm);
    }
  }

  // Check that any non-zero on the subdiagonal is followed by a zero and is
  // part of a 2x2 diagonal block with imaginary eigenvalues.
  for(int row = 1; row < size; ++row) {
    if (!test_ei_isMuchSmallerThan(T(row,row-1), norm)) {
      VERIFY(row == size-1 || test_ei_isMuchSmallerThan(T(row+1,row), norm));
      Scalar tr = T(row-1,row-1) + T(row,row);
      Scalar det = T(row-1,row-1) * T(row,row) - T(row-1,row) * T(row,row-1);
      VERIFY(4 * det > tr * tr);
    }
  }
}

template<typename MatrixType> void schur(int size = MatrixType::ColsAtCompileTime)
{
  // Test basic functionality: T is quasi-triangular and A = U T U*
  for(int counter = 0; counter < g_repeat; ++counter) {
    MatrixType A = MatrixType::Random(size, size);
    RealSchur<MatrixType> schurOfA(A);
    MatrixType U = schurOfA.matrixU();
    MatrixType T = schurOfA.matrixT();
    verifyIsQuasiTriangular(T);
    VERIFY_IS_APPROX(A, U * T * U.transpose());
  }
}

void test_schur_real()
{
  CALL_SUBTEST_1(( schur<Matrix4f>() ));
  CALL_SUBTEST_2(( schur<MatrixXd>(ei_random<int>(1,50)) ));
  CALL_SUBTEST_3(( schur<Matrix<float, 1, 1> >() ));
  CALL_SUBTEST_4(( schur<Matrix<double, 3, 3, Eigen::RowMajor> >() ));
}
