// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@gmail.com>
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

template<typename MatrixType> void triangular(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;

  int rows = m.rows();
  int cols = m.cols();

  MatrixType m1 = MatrixType::random(rows, cols),
             m2 = MatrixType::random(rows, cols),
             m3(rows, cols),
             r1(rows, cols),
             r2(rows, cols),
             mzero = MatrixType::zero(rows, cols),
             mones = MatrixType::ones(rows, cols),
             identity = Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime>
                              ::identity(rows, rows),
             square = Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime>
                              ::random(rows, rows);
  VectorType v1 = VectorType::random(rows),
             v2 = VectorType::random(rows),
             vzero = VectorType::zero(rows);

  MatrixType m1up = m1.upper();
  MatrixType m2up = m2.upper();

  if (rows*cols>1)
  {
    VERIFY(m1up.isUpper());
    VERIFY(m2up.transpose().isLower());
    VERIFY(!m2.isLower());
  }

//   VERIFY_IS_APPROX(m1up.transpose() * m2, m1.upper().transpose().lower() * m2);

  // test overloaded operator+=
  r1.setZero();
  r2.setZero();
  r1.upper() +=  m1;
  r2 += m1up;
  VERIFY_IS_APPROX(r1,r2);

  // test overloaded operator=
  m1.setZero();
  m1.upper() = (m2.transpose() * m2).lazy();
  m3 = m2.transpose() * m2;
  VERIFY_IS_APPROX(m3.lower().transpose(), m1);

  // test overloaded operator=
  m1.setZero();
  m1.lower() = (m2.transpose() * m2).lazy();
  VERIFY_IS_APPROX(m3.lower(), m1);

  // test back and forward subsitution
  m1 = MatrixType::random(rows, cols);
  VERIFY_IS_APPROX(m1.upper() * (m1.upper().inverseProduct(m2)), m2);
  VERIFY_IS_APPROX(m1.lower() * (m1.lower().inverseProduct(m2)), m2);
  VERIFY((m1.upper() * m2.upper()).isUpper());

}

void test_triangular()
{
  for(int i = 0; i < g_repeat ; i++) {
//     triangular(Matrix<float, 1, 1>());
    CALL_SUBTEST( triangular(Matrix3d()) );
    CALL_SUBTEST( triangular(MatrixXcf(4, 4)) );
//     CALL_SUBTEST( triangular(Matrix<std::complex<float>,8, 8>()) );
  }
}
