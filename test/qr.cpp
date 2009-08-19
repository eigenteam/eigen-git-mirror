// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
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
#include <Eigen/QR>

template<typename MatrixType> void qr(const MatrixType& m)
{
  /* this test covers the following files: QR.h */
  int rows = m.rows();
  int cols = m.cols();

  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, MatrixType::ColsAtCompileTime> SquareMatrixType;
  typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, 1> VectorType;

  MatrixType a = MatrixType::Random(rows,cols);
  HouseholderQR<MatrixType> qrOfA(a);
  VERIFY_IS_APPROX(a, qrOfA.matrixQ() * qrOfA.matrixR().toDense());
  VERIFY_IS_NOT_APPROX(a+MatrixType::Identity(rows, cols), qrOfA.matrixQ() * qrOfA.matrixR().toDense());

  SquareMatrixType b = a.adjoint() * a;

  // check tridiagonalization
  Tridiagonalization<SquareMatrixType> tridiag(b);
  VERIFY_IS_APPROX(b, tridiag.matrixQ() * tridiag.matrixT() * tridiag.matrixQ().adjoint());

  // check hessenberg decomposition
  HessenbergDecomposition<SquareMatrixType> hess(b);
  VERIFY_IS_APPROX(b, hess.matrixQ() * hess.matrixH() * hess.matrixQ().adjoint());
  VERIFY_IS_APPROX(tridiag.matrixT(), hess.matrixH());
  b = SquareMatrixType::Random(cols,cols);
  hess.compute(b);
  VERIFY_IS_APPROX(b, hess.matrixQ() * hess.matrixH() * hess.matrixQ().adjoint());
}

template<typename MatrixType> void qr_invertible()
{
  /* this test covers the following files: QR.h */
  typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
  int size = ei_random<int>(10,50);

  MatrixType m1(size, size), m2(size, size), m3(size, size);
  m1 = MatrixType::Random(size,size);

  if (ei_is_same_type<RealScalar,float>::ret)
  {
    // let's build a matrix more stable to inverse
    MatrixType a = MatrixType::Random(size,size*2);
    m1 += a * a.adjoint();
  }

  HouseholderQR<MatrixType> qr(m1);
  m3 = MatrixType::Random(size,size);
  qr.solve(m3, &m2);
  VERIFY_IS_APPROX(m3, m1*m2);
}

template<typename MatrixType> void qr_verify_assert()
{
  MatrixType tmp;

  HouseholderQR<MatrixType> qr;
  VERIFY_RAISES_ASSERT(qr.matrixR())
  VERIFY_RAISES_ASSERT(qr.solve(tmp,&tmp))
  VERIFY_RAISES_ASSERT(qr.matrixQ())
}

void test_qr()
{
  for(int i = 0; i < 1; i++) {
    // FIXME : very weird bug here
//     CALL_SUBTEST( qr(Matrix2f()) );
    CALL_SUBTEST( qr(Matrix4d()) );
    CALL_SUBTEST( qr(MatrixXf(47,40)) );
    CALL_SUBTEST( qr(MatrixXcd(17,7)) );
  }

  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST( qr_invertible<MatrixXf>() );
    CALL_SUBTEST( qr_invertible<MatrixXd>() );
    CALL_SUBTEST( qr_invertible<MatrixXcf>() );
    CALL_SUBTEST( qr_invertible<MatrixXcd>() );
  }

  CALL_SUBTEST(qr_verify_assert<Matrix3f>());
  CALL_SUBTEST(qr_verify_assert<Matrix3d>());
  CALL_SUBTEST(qr_verify_assert<MatrixXf>());
  CALL_SUBTEST(qr_verify_assert<MatrixXd>());
  CALL_SUBTEST(qr_verify_assert<MatrixXcf>());
  CALL_SUBTEST(qr_verify_assert<MatrixXcd>());
}
