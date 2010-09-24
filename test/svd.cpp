// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
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
#include <Eigen/SVD>
#include <Eigen/LU>

template<typename MatrixType> void svd(const MatrixType& m)
{
  /* this test covers the following files:
     SVD.h
  */
  typename MatrixType::Index rows = m.rows();
  typename MatrixType::Index cols = m.cols();

  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  MatrixType a = MatrixType::Random(rows,cols);
  Matrix<Scalar, MatrixType::ColsAtCompileTime, 1> x(cols,1), x2(cols,1);

  {
    SVD<MatrixType> svd(a);
    MatrixType sigma = MatrixType::Zero(rows,cols);
    MatrixType matU  = MatrixType::Zero(rows,rows);
    MatrixType matV  = MatrixType::Zero(cols,cols);

    sigma.diagonal() = svd.singularValues();
    matU = svd.matrixU();
    VERIFY_IS_UNITARY(matU);
    matV = svd.matrixV();
    VERIFY_IS_UNITARY(matV);
    VERIFY_IS_APPROX(a, matU * sigma * matV.transpose());
  }


  if (rows>=cols)
  {
    SVD<MatrixType> svd(a);
    Matrix<Scalar, MatrixType::ColsAtCompileTime, 1> b = Matrix<Scalar, MatrixType::ColsAtCompileTime, 1>::Random(rows,1);
    Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> x = svd.solve(b);
    // evaluate normal equation which works also for least-squares solutions
    VERIFY_IS_APPROX(a.adjoint()*a*x,a.adjoint()*b);
  }


  if(rows==cols)
  {
    SVD<MatrixType> svd(a);
    MatrixType unitary, positive;
    svd.computeUnitaryPositive(&unitary, &positive);
    VERIFY_IS_APPROX(unitary * unitary.adjoint(), MatrixType::Identity(unitary.rows(),unitary.rows()));
    VERIFY_IS_APPROX(positive, positive.adjoint());
    for(int i = 0; i < rows; i++) VERIFY(positive.diagonal()[i] >= 0); // cheap necessary (not sufficient) condition for positivity
    VERIFY_IS_APPROX(unitary*positive, a);

    svd.computePositiveUnitary(&positive, &unitary);
    VERIFY_IS_APPROX(unitary * unitary.adjoint(), MatrixType::Identity(unitary.rows(),unitary.rows()));
    VERIFY_IS_APPROX(positive, positive.adjoint());
    for(int i = 0; i < rows; i++) VERIFY(positive.diagonal()[i] >= 0); // cheap necessary (not sufficient) condition for positivity
    VERIFY_IS_APPROX(positive*unitary, a);
  }
}

template<typename MatrixType> void svd_verify_assert()
{
  MatrixType tmp;

  SVD<MatrixType> svd;
  VERIFY_RAISES_ASSERT(svd.solve(tmp))
  VERIFY_RAISES_ASSERT(svd.matrixU())
  VERIFY_RAISES_ASSERT(svd.singularValues())
  VERIFY_RAISES_ASSERT(svd.matrixV())
  VERIFY_RAISES_ASSERT(svd.computeUnitaryPositive(&tmp,&tmp))
  VERIFY_RAISES_ASSERT(svd.computePositiveUnitary(&tmp,&tmp))
  VERIFY_RAISES_ASSERT(svd.computeRotationScaling(&tmp,&tmp))
  VERIFY_RAISES_ASSERT(svd.computeScalingRotation(&tmp,&tmp))
}

void test_svd()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( svd(Matrix3f()) );
    CALL_SUBTEST_2( svd(Matrix4d()) );
    int cols = ei_random<int>(2,50);
    int rows = cols + ei_random<int>(0,50);
    
    CALL_SUBTEST_3( svd(MatrixXf(rows,cols)) );
    CALL_SUBTEST_4( svd(MatrixXd(rows,cols)) );
    // complex are not implemented yet
//     CALL_SUBTEST(svd(MatrixXcd(6,6)) );
//     CALL_SUBTEST(svd(MatrixXcf(3,3)) );
  }

  CALL_SUBTEST_1( svd_verify_assert<Matrix3f>() );
  CALL_SUBTEST_2( svd_verify_assert<Matrix4d>() );
  CALL_SUBTEST_3( svd_verify_assert<MatrixXf>() );
  CALL_SUBTEST_4( svd_verify_assert<MatrixXd>() );

  // Test problem size constructors
  CALL_SUBTEST_9( SVD<MatrixXf>(10, 20) );
}
