// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
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

template<typename MatrixType> void product(const MatrixType& m)
{
  /* this test covers the following files:
     Identity.h Product.h
  */

  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> SquareMatrixType;

  int rows = m.rows();
  int cols = m.cols();

  // this test relies a lot on Random.h, and there's not much more that we can do
  // to test it, hence I consider that we will have tested Random.h
  MatrixType m1 = MatrixType::random(rows, cols),
             m2 = MatrixType::random(rows, cols),
             m3(rows, cols),
             mzero = MatrixType::zero(rows, cols);
  SquareMatrixType
             identity = Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime>
                              ::identity(rows, rows),
             square = Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime>
                              ::random(rows, rows);
  VectorType v1 = VectorType::random(rows),
             v2 = VectorType::random(rows),
             vzero = VectorType::zero(rows);

  Scalar s1 = ei_random<Scalar>();

  int r = ei_random<int>(0, rows-1),
      c = ei_random<int>(0, cols-1);

  // begin testing Product.h: only associativity for now
  // (we use Transpose.h but this doesn't count as a test for it)
  VERIFY_IS_APPROX((m1*m1.transpose())*m2,  m1*(m1.transpose()*m2));
  m3 = m1;
  m3 *= (m1.transpose() * m2);
  VERIFY_IS_APPROX(m3,                      m1 * (m1.transpose()*m2));
  VERIFY_IS_APPROX(m3,                      m1.lazy() * (m1.transpose()*m2));

  // continue testing Product.h: distributivity
  VERIFY_IS_APPROX(square*(m1 + m2),        square*m1+square*m2);
  VERIFY_IS_APPROX(square*(m1 - m2),        square*m1-square*m2);

  // continue testing Product.h: compatibility with ScalarMultiple.h
  VERIFY_IS_APPROX(s1*(square*m1),          (s1*square)*m1);
  VERIFY_IS_APPROX(s1*(square*m1),          square*(m1*s1));

  // continue testing Product.h: lazy product
  VERIFY_IS_APPROX(square.lazy() * m1,  square*m1);
  VERIFY_IS_APPROX(square * m1.lazy(),  square*m1);
  // again, test operator() to check const-qualification
  s1 += (square.lazy() * m1)(r,c);

  // test Product.h together with Identity.h
  VERIFY_IS_APPROX(m1,                      identity*m1);
  VERIFY_IS_APPROX(v1,                      identity*v1);
  // again, test operator() to check const-qualification
  VERIFY_IS_APPROX(MatrixType::identity(rows, cols)(r,c), static_cast<Scalar>(r==c));

  if (rows!=cols)
    VERIFY_RAISES_ASSERT(m3 = m1*m1);
}

void test_product()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST( product(Matrix<float, 1, 1>()) );
    CALL_SUBTEST( product(Matrix<float, 3, 3>()) );
    CALL_SUBTEST( product(Matrix<float, 4, 2>()) );
    CALL_SUBTEST( product(Matrix4d()) );
  }
  for(int i = 0; i < g_repeat; i++) {
    int rows = ei_random<int>(1,320);
    int cols = ei_random<int>(1,320);
    CALL_SUBTEST( product(MatrixXf(rows, cols)) );
    CALL_SUBTEST( product(MatrixXd(rows, cols)) );
    CALL_SUBTEST( product(MatrixXi(rows, cols)) );
    CALL_SUBTEST( product(MatrixXcf(rows, cols)) );
    CALL_SUBTEST( product(MatrixXcd(rows, cols)) );
  }
}
