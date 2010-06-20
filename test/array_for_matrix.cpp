// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <g.gael@free.fr>
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

template<typename MatrixType> void array_for_matrix(const MatrixType& m)
{
  typedef typename MatrixType::Index Index;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> ColVectorType;
  typedef Matrix<Scalar, 1, MatrixType::ColsAtCompileTime> RowVectorType;

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols),
             m3(rows, cols);

  ColVectorType cv1 = ColVectorType::Random(rows);
  RowVectorType rv1 = RowVectorType::Random(cols);

  Scalar  s1 = ei_random<Scalar>(),
          s2 = ei_random<Scalar>();

  // scalar addition
  VERIFY_IS_APPROX(m1.array() + s1, s1 + m1.array());
  VERIFY_IS_APPROX((m1.array() + s1).matrix(), MatrixType::Constant(rows,cols,s1) + m1);
  VERIFY_IS_APPROX(((m1*Scalar(2)).array() - s2).matrix(), (m1+m1) - MatrixType::Constant(rows,cols,s2) );
  m3 = m1;
  m3.array() += s2;
  VERIFY_IS_APPROX(m3, (m1.array() + s2).matrix());
  m3 = m1;
  m3.array() -= s1;
  VERIFY_IS_APPROX(m3, (m1.array() - s1).matrix());

  // reductions
  VERIFY_IS_APPROX(m1.colwise().sum().sum(), m1.sum());
  VERIFY_IS_APPROX(m1.rowwise().sum().sum(), m1.sum());
  if (!ei_isApprox(m1.sum(), (m1+m2).sum()))
    VERIFY_IS_NOT_APPROX(((m1+m2).rowwise().sum()).sum(), m1.sum());
  VERIFY_IS_APPROX(m1.colwise().sum(), m1.colwise().redux(ei_scalar_sum_op<Scalar>()));

  // vector-wise ops
  m3 = m1;
  VERIFY_IS_APPROX(m3.colwise() += cv1, m1.colwise() + cv1);
  m3 = m1;
  VERIFY_IS_APPROX(m3.colwise() -= cv1, m1.colwise() - cv1);
  m3 = m1;
  VERIFY_IS_APPROX(m3.rowwise() += rv1, m1.rowwise() + rv1);
  m3 = m1;
  VERIFY_IS_APPROX(m3.rowwise() -= rv1, m1.rowwise() - rv1);
}

template<typename MatrixType> void comparisons(const MatrixType& m)
{
  typedef typename MatrixType::Index Index;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;

  Index rows = m.rows();
  Index cols = m.cols();

  Index r = ei_random<Index>(0, rows-1),
        c = ei_random<Index>(0, cols-1);

  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols),
             m3(rows, cols);

  VERIFY(((m1.array() + Scalar(1)) > m1.array()).all());
  VERIFY(((m1.array() - Scalar(1)) < m1.array()).all());
  if (rows*cols>1)
  {
    m3 = m1;
    m3(r,c) += 1;
    VERIFY(! (m1.array() < m3.array()).all() );
    VERIFY(! (m1.array() > m3.array()).all() );
  }

  // comparisons to scalar
  VERIFY( (m1.array() != (m1(r,c)+1) ).any() );
  VERIFY( (m1.array() > (m1(r,c)-1) ).any() );
  VERIFY( (m1.array() < (m1(r,c)+1) ).any() );
  VERIFY( (m1.array() == m1(r,c) ).any() );

  // test Select
  VERIFY_IS_APPROX( (m1.array()<m2.array()).select(m1,m2), m1.cwiseMin(m2) );
  VERIFY_IS_APPROX( (m1.array()>m2.array()).select(m1,m2), m1.cwiseMax(m2) );
  Scalar mid = (m1.cwiseAbs().minCoeff() + m1.cwiseAbs().maxCoeff())/Scalar(2);
  for (int j=0; j<cols; ++j)
  for (int i=0; i<rows; ++i)
    m3(i,j) = ei_abs(m1(i,j))<mid ? 0 : m1(i,j);
  VERIFY_IS_APPROX( (m1.array().abs()<MatrixType::Constant(rows,cols,mid).array())
                        .select(MatrixType::Zero(rows,cols),m1), m3);
  // shorter versions:
  VERIFY_IS_APPROX( (m1.array().abs()<MatrixType::Constant(rows,cols,mid).array())
                        .select(0,m1), m3);
  VERIFY_IS_APPROX( (m1.array().abs()>=MatrixType::Constant(rows,cols,mid).array())
                        .select(m1,0), m3);
  // even shorter version:
  VERIFY_IS_APPROX( (m1.array().abs()<mid).select(0,m1), m3);

  // count
  VERIFY(((m1.array().abs()+1)>RealScalar(0.1)).count() == rows*cols);

  typedef Matrix<typename MatrixType::Index, Dynamic, 1> VectorOfIndices;

  // TODO allows colwise/rowwise for array
  VERIFY_IS_APPROX(((m1.array().abs()+1)>RealScalar(0.1)).matrix().colwise().count(), VectorOfIndices::Constant(cols,rows).transpose());
  VERIFY_IS_APPROX(((m1.array().abs()+1)>RealScalar(0.1)).matrix().rowwise().count(), VectorOfIndices::Constant(rows, cols));
}

template<typename VectorType> void lpNorm(const VectorType& v)
{
  VectorType u = VectorType::Random(v.size());

  VERIFY_IS_APPROX(u.template lpNorm<Infinity>(), u.cwiseAbs().maxCoeff());
  VERIFY_IS_APPROX(u.template lpNorm<1>(), u.cwiseAbs().sum());
  VERIFY_IS_APPROX(u.template lpNorm<2>(), ei_sqrt(u.array().abs().square().sum()));
  VERIFY_IS_APPROX(ei_pow(u.template lpNorm<5>(), typename VectorType::RealScalar(5)), u.array().abs().pow(5).sum());
}

void test_array_for_matrix()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( array_for_matrix(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_2( array_for_matrix(Matrix2f()) );
    CALL_SUBTEST_3( array_for_matrix(Matrix4d()) );
    CALL_SUBTEST_4( array_for_matrix(MatrixXcf(3, 3)) );
    CALL_SUBTEST_5( array_for_matrix(MatrixXf(8, 12)) );
    CALL_SUBTEST_6( array_for_matrix(MatrixXi(8, 12)) );
  }
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( comparisons(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_2( comparisons(Matrix2f()) );
    CALL_SUBTEST_3( comparisons(Matrix4d()) );
    CALL_SUBTEST_5( comparisons(MatrixXf(8, 12)) );
    CALL_SUBTEST_6( comparisons(MatrixXi(8, 12)) );
  }
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( lpNorm(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_2( lpNorm(Vector2f()) );
    CALL_SUBTEST_7( lpNorm(Vector3d()) );
    CALL_SUBTEST_8( lpNorm(Vector4f()) );
    CALL_SUBTEST_5( lpNorm(VectorXf(16)) );
    CALL_SUBTEST_4( lpNorm(VectorXcf(10)) );
  }
}
