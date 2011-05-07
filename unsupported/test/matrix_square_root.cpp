// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Jitse Niesen <jitse@maths.leeds.ac.uk>
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
#include <unsupported/Eigen/MatrixFunctions>

template<typename MatrixType>
void testMatrixSqrt(const MatrixType& m)
{
  typedef typename MatrixType::Index Index;
  const Index size = m.rows();
  MatrixType A = MatrixType::Random(size, size);
  MatrixSquareRoot<MatrixType> msr(A);
  MatrixType S;
  msr.compute(S);
  VERIFY_IS_APPROX(S*S, A);
}

void test_matrix_square_root()
{
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(testMatrixSqrt(Matrix3cf()));
    CALL_SUBTEST_2(testMatrixSqrt(MatrixXcd(12,12)));
  }
}
