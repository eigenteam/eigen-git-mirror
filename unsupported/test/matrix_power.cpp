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
  T angle, ch = std::cosh((T)1);
  std::complex<T> ish(0, std::sinh((T)1));

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
    VERIFY(m4.isApprox(m5, static_cast<RealScalar>(tol)));

    m4 = mpow(x*y);
    m5 = m2.pow(y);
    VERIFY(m4.isApprox(m5, static_cast<RealScalar>(tol)));

    m4 = (std::abs(x) * m1).pow(y);
    m5 = std::pow(std::abs(x), y) * m3;
    VERIFY(m4.isApprox(m5, static_cast<RealScalar>(tol)));
  }
}

template<typename MatrixType, typename VectorType>
void testProduct(const MatrixType& m, const VectorType& v, double tol)
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

template<typename MatrixType, typename VectorType>
void testMatrixVector(const MatrixType& m, const VectorType& v, double tol)
{
  testExponentLaws(m,tol);
  testProduct(m,v,tol);
}

void test_matrix_power()
{
  typedef Matrix<double,3,3,RowMajor>         Matrix3dRowMajor;
  typedef Matrix<long double,Dynamic,Dynamic> MatrixXe;
  typedef Matrix<long double,Dynamic,1>       VectorXe;

  CALL_SUBTEST_2(test2dRotation<double>(1e-13));
  CALL_SUBTEST_1(test2dRotation<float>(2e-5));  // was 1e-5, relaxed for clang 2.8 / linux / x86-64
  CALL_SUBTEST_9(test2dRotation<long double>(1e-13)); 
  CALL_SUBTEST_2(test2dHyperbolicRotation<double>(1e-14));
  CALL_SUBTEST_1(test2dHyperbolicRotation<float>(1e-5));
  CALL_SUBTEST_9(test2dHyperbolicRotation<long double>(1e-14));

  CALL_SUBTEST_2(testMatrixVector(Matrix2d(),         Vector2d(),    1e-13));
  CALL_SUBTEST_7(testMatrixVector(Matrix3dRowMajor(), MatrixXd(3,5), 1e-13));
  CALL_SUBTEST_3(testMatrixVector(Matrix4cd(),        Vector4cd(),   1e-13));
  CALL_SUBTEST_4(testMatrixVector(MatrixXd(8,8),      VectorXd(8),   1e-13));
  CALL_SUBTEST_1(testMatrixVector(Matrix2f(),         Vector2f(),    1e-4));
  CALL_SUBTEST_5(testMatrixVector(Matrix3cf(),        Vector3cf(),   1e-4));
  CALL_SUBTEST_8(testMatrixVector(Matrix4f(),         Vector4f(),    1e-4));
  CALL_SUBTEST_6(testMatrixVector(MatrixXf(8,8),      VectorXf(8),   1e-4));
  CALL_SUBTEST_9(testMatrixVector(MatrixXe(7,7),      VectorXe(7),   1e-13));
}
