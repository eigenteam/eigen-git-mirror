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
#include <Eigen/Array>

template<typename MatrixType> void scalarAdd(const MatrixType& m)
{
  /* this test covers the following files:
     Array.cpp
  */

  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;

  int rows = m.rows();
  int cols = m.cols();

  MatrixType m1 = test_random_matrix<MatrixType>(rows, cols),
             m2 = test_random_matrix<MatrixType>(rows, cols),
             m3(rows, cols);

  Scalar  s1 = test_random<Scalar>(),
          s2 = test_random<Scalar>();

  VERIFY_IS_APPROX(m1.cwise() + s1, s1 + m1.cwise());
  VERIFY_IS_APPROX(m1.cwise() + s1, MatrixType::Constant(rows,cols,s1) + m1);
  VERIFY_IS_APPROX((m1*Scalar(2)).cwise() - s2, (m1+m1) - MatrixType::Constant(rows,cols,s2) );
  m3 = m1;
  m3.cwise() += s2;
  VERIFY_IS_APPROX(m3, m1.cwise() + s2);
  m3 = m1;
  m3.cwise() -= s1;
  VERIFY_IS_APPROX(m3, m1.cwise() - s1);

  VERIFY_IS_APPROX(m1.colwise().sum().sum(), m1.sum());
  VERIFY_IS_APPROX(m1.rowwise().sum().sum(), m1.sum());
  if (!ei_isApprox(m1.sum(), (m1+m2).sum()))
    VERIFY_IS_NOT_APPROX(((m1+m2).rowwise().sum()).sum(), m1.sum());
  VERIFY_IS_APPROX(m1.colwise().sum(), m1.colwise().redux(ei_scalar_sum_op<Scalar>()));
}

template<typename MatrixType> void comparisons(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;

  int rows = m.rows();
  int cols = m.cols();

  int r = ei_random<int>(0, rows-1),
      c = ei_random<int>(0, cols-1);

  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols),
             m3(rows, cols);

  VERIFY(((m1.cwise() + Scalar(1)).cwise() > m1).all());
  VERIFY(((m1.cwise() - Scalar(1)).cwise() < m1).all());
  if (rows*cols>1)
  {
    m3 = m1;
    m3(r,c) += 1;
    VERIFY(! (m1.cwise() < m3).all() );
    VERIFY(! (m1.cwise() > m3).all() );
  }
}

void test_array()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST( scalarAdd(Matrix<float, 1, 1>()) );
    CALL_SUBTEST( scalarAdd(Matrix2f()) );
    CALL_SUBTEST( scalarAdd(Matrix4d()) );
    CALL_SUBTEST( scalarAdd(MatrixXcf(3, 3)) );
    CALL_SUBTEST( scalarAdd(MatrixXf(8, 12)) );
    CALL_SUBTEST( scalarAdd(MatrixXi(8, 12)) );
  }
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST( comparisons(Matrix<float, 1, 1>()) );
    CALL_SUBTEST( comparisons(Matrix2f()) );
    CALL_SUBTEST( comparisons(Matrix4d()) );
    CALL_SUBTEST( comparisons(MatrixXf(8, 12)) );
    CALL_SUBTEST( comparisons(MatrixXi(8, 12)) );
  }
}
