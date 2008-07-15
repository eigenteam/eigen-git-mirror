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

template<typename MatrixType> void inverse(const MatrixType& m)
{
  /* this test covers the following files:
     Inverse.h
  */
  int rows = m.rows();
  int cols = m.cols();

  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, 1> VectorType;

  MatrixType m1 = MatrixType::random(rows, cols),
             m2 = MatrixType::random(rows, cols),
             mzero = MatrixType::zero(rows, cols),
             identity = MatrixType::identity(rows, rows);

  m2 = m1.inverse();
  VERIFY_IS_APPROX(m1, m2.inverse() );

  m1.computeInverse(&m2);
  VERIFY_IS_APPROX(m1, m2.inverse() );

  VERIFY_IS_APPROX((Scalar(2)*m2).inverse(), m2.inverse()*Scalar(0.5));

  VERIFY_IS_APPROX(identity, m1.inverse() * m1 );
  VERIFY_IS_APPROX(identity, m1 * m1.inverse() );

  VERIFY_IS_APPROX(m1, m1.inverse().inverse() );
}

void test_inverse()
{
  for(int i = 0; i < 1; i++) {
    CALL_SUBTEST( inverse(Matrix<double,1,1>()) );
    CALL_SUBTEST( inverse(Matrix2d()) );
    CALL_SUBTEST( inverse(Matrix3f()) );
    CALL_SUBTEST( inverse(Matrix4f()) );
    CALL_SUBTEST( inverse(MatrixXcd(7,7)) );
  }
}
