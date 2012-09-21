// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "matrix_functions.h"

template<typename T>
void test2dRotation(double tol)
{
  Matrix<T,2,2> A, B, C;
  T angle, c, s;

  A << 0, 1, -1, 0;
  MatrixPower<Matrix<T,2,2> > Apow(A);

  for (int i=0; i<=20; ++i) {
    angle = pow(10, (i-10) / 5.);
    c = std::cos(angle);
    s = std::sin(angle);
    B << c, s, -s, c;

    C = Apow(std::ldexp(angle,1) / M_PI);
    std::cout << "test2dRotation: i = " << i << "   error powerm = " << relerr(C,B) << '\n';
    VERIFY(C.isApprox(B, static_cast<T>(tol)));
  }
}

template<typename T>
void test2dHyperbolicRotation(double tol)
{
  Matrix<std::complex<T>,2,2> A, B, C;
  T angle, ch = std::cosh(1);
  std::complex<T> ish(0, std::sinh(1));

  A << ch, ish, -ish, ch;
  MatrixPower<Matrix<std::complex<T>,2,2> > Apow(A);

  for (int i=0; i<=20; ++i) {
    angle = std::ldexp(static_cast<T>(i-10), -1);
    ch = std::cosh(angle);
    ish = std::complex<T>(0, std::sinh(angle));
    B << ch, ish, -ish, ch;

    C = Apow(angle);
    std::cout << "test2dHyperbolicRotation: i = " << i << "   error powerm = " << relerr(C,B) << '\n';
    VERIFY(C.isApprox(B, static_cast<T>(tol)));
  }
}

template<typename MatrixType>
void testExponentLaws(const MatrixType& m, double tol)
{
  typedef typename MatrixType::RealScalar RealScalar;
  MatrixType m1, m2, m3, m4, m5;
  RealScalar x, y;

  for (int i=0; i<g_repeat; ++i) {
    generateTestMatrix<MatrixType>::run(m1, m.rows());
    MatrixPower<MatrixType> mpow(m1);

    x = internal::random<RealScalar>();
    y = internal::random<RealScalar>();
    m2 = mpow(x);
    m3 = mpow(y);

    m4 = mpow(x+y);
    m5.noalias() = m2 * m3;

    std::cout << "testExponentLaws: error powerm = " << relerr(m4, m5);
    VERIFY(m4.isApprox(m5, static_cast<RealScalar>(tol)));

    m4 = mpow(x*y);
    m5 = m2.pow(y);
    std::cout << "   " << relerr(m4, m5);
    VERIFY(m4.isApprox(m5, static_cast<RealScalar>(tol)));

    m4 = (std::abs(x) * m1).pow(y);
    m5 = std::pow(std::abs(x), y) * m3;
    std::cout << "   " << relerr(m4, m5) << '\n';
    VERIFY(m4.isApprox(m5, static_cast<RealScalar>(tol)));
  }
}

template<typename MatrixType, typename VectorType>
void testMatrixVectorProduct(const MatrixType& m, const VectorType& v, double tol)
{
  typedef typename MatrixType::RealScalar RealScalar;
  MatrixType m1;
  VectorType v1, v2, v3;
  RealScalar p;

  for (int i=0; i<g_repeat; ++i) {
    generateTestMatrix<MatrixType>::run(m1, m.rows());
    MatrixPower<MatrixType> mpow(m1);

    v1 = VectorType::Random(v.rows(), v.cols());
    p = internal::random<RealScalar>();

    v2.noalias() = mpow(p) * v1;
    v3.noalias() = mpow(p).eval() * v1;
    std::cout << "testMatrixVectorProduct: error powerm = " << relerr(v2, v3) << '\n';
    VERIFY(v2.isApprox(v3, static_cast<RealScalar>(tol)));
  }
}

void test_matrix_power()
{
  typedef Matrix<long double,Dynamic,Dynamic> MatrixXe;

  CALL_SUBTEST_2(test2dRotation<double>(1e-13));
  CALL_SUBTEST_1(test2dRotation<float>(2e-5));  // was 1e-5, relaxed for clang 2.8 / linux / x86-64
  CALL_SUBTEST_9(test2dRotation<long double>(1e-13)); 
  CALL_SUBTEST_2(test2dHyperbolicRotation<double>(1e-14));
  CALL_SUBTEST_1(test2dHyperbolicRotation<float>(1e-5));
  CALL_SUBTEST_9(test2dHyperbolicRotation<long double>(1e-14));

  CALL_SUBTEST_2(testExponentLaws(Matrix2d(), 1e-13));
  CALL_SUBTEST_7(testExponentLaws(Matrix<double,3,3,RowMajor>(), 1e-13));
  CALL_SUBTEST_3(testExponentLaws(Matrix4cd(), 1e-13));
  CALL_SUBTEST_4(testExponentLaws(MatrixXd(8,8), 1e-13));
  CALL_SUBTEST_1(testExponentLaws(Matrix2f(), 1e-4));
  CALL_SUBTEST_5(testExponentLaws(Matrix3cf(), 1e-4));
  CALL_SUBTEST_8(testExponentLaws(Matrix4f(), 1e-4));
  CALL_SUBTEST_6(testExponentLaws(MatrixXf(8,8), 1e-4));

  CALL_SUBTEST_2(testMatrixVectorProduct(Matrix2d(), Vector2d(), 1e-13));
  CALL_SUBTEST_7(testMatrixVectorProduct(Matrix<double,3,3,RowMajor>(), Vector3d(), 1e-13));
  CALL_SUBTEST_3(testMatrixVectorProduct(Matrix4cd(), Vector4cd(), 1e-13));
  CALL_SUBTEST_4(testMatrixVectorProduct(MatrixXd(8,8), MatrixXd(8,2), 1e-13));
  CALL_SUBTEST_1(testMatrixVectorProduct(Matrix2f(), Vector2f(), 1e-4));
  CALL_SUBTEST_5(testMatrixVectorProduct(Matrix3cf(), Vector3cf(), 1e-4));
  CALL_SUBTEST_8(testMatrixVectorProduct(Matrix4f(), Vector4f(), 1e-4));
  CALL_SUBTEST_6(testMatrixVectorProduct(MatrixXf(8,8), VectorXf(8), 1e-4));
  CALL_SUBTEST_9(testMatrixVectorProduct(MatrixXe(7,7), MatrixXe(7,9), 1e-13));
}
