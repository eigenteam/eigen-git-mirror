// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "matrix_functions.h"

template <typename T>
void test2dRotation(double tol)
{
  Matrix<T,2,2> A, B, C;
  T angle, c, s;

  A << 0, 1, -1, 0;
  for (int i = 0; i <= 20; i++) {
    angle = pow(10, (i-10) / 5.);
    c = std::cos(angle);
    s = std::sin(angle);
    B << c, s, -s, c;

    C = A.pow(std::ldexp(angle, 1) / M_PI);
    std::cout << "test2dRotation: i = " << i << "   error powerm = " << relerr(C, B) << "\n";
    VERIFY(C.isApprox(B, T(tol)));
  }
}

template <typename T>
void test2dHyperbolicRotation(double tol)
{
  Matrix<std::complex<T>,2,2> A, B, C;
  T angle, ch = std::cosh(1);
  std::complex<T> ish(0, std::sinh(1));

  A << ch, ish, -ish, ch;
  for (int i = 0; i <= 20; i++) {
    angle = std::ldexp(T(i-10), -1);
    ch = std::cosh(angle);
    ish = std::complex<T>(0, std::sinh(angle));
    B << ch, ish, -ish, ch;

    C = A.pow(angle);
    std::cout << "test2dHyperbolicRotation: i = " << i << "   error powerm = " << relerr(C, B) << "\n";
    VERIFY(C.isApprox(B, T(tol)));
  }
}

template <typename MatrixType>
void testExponentLaws(const MatrixType& m, double tol)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;

  typename MatrixType::Index rows = m.rows();
  typename MatrixType::Index cols = m.cols();
  MatrixType m1, m1x, m1y, m2, m3;
  RealScalar x = internal::random<RealScalar>(), y = internal::random<RealScalar>();
  double err[3];

  for(int i = 0; i < g_repeat; i++) {
    generateTestMatrix<MatrixType>::run(m1, m.rows());
    m1x = m1.pow(x);
    m1y = m1.pow(y);

    m2 = m1.pow(x + y);
    m3 = m1x * m1y;
    err[0] = relerr(m2, m3);
    VERIFY(m2.isApprox(m3, static_cast<RealScalar>(tol)));

    m2 = m1.pow(x * y);
    m3 = m1x.pow(y);
    err[1] = relerr(m2, m3);
    VERIFY(m2.isApprox(m3, static_cast<RealScalar>(tol)));

    m2 = (std::abs(x) * m1).pow(y);
    m3 = std::pow(std::abs(x), y) * m1y;
    err[2] = relerr(m2, m3);
    VERIFY(m2.isApprox(m3, static_cast<RealScalar>(tol)));

    std::cout << "testExponentLaws: error powerm = " << err[0] << "  " << err[1] << "  " << err[2] << "\n";
  }
}

void test_matrix_power()
{
  CALL_SUBTEST_2(test2dRotation<double>(1e-13));
  CALL_SUBTEST_1(test2dRotation<float>(2e-5));  // was 1e-5, relaxed for clang 2.8 / linux / x86-64
  CALL_SUBTEST_8(test2dRotation<long double>(1e-13)); 
  CALL_SUBTEST_2(test2dHyperbolicRotation<double>(1e-14));
  CALL_SUBTEST_1(test2dHyperbolicRotation<float>(1e-5));
  CALL_SUBTEST_8(test2dHyperbolicRotation<long double>(1e-14));
  CALL_SUBTEST_2(testExponentLaws(Matrix2d(), 1e-13));
  CALL_SUBTEST_7(testExponentLaws(Matrix<double,3,3,RowMajor>(), 1e-13));
  CALL_SUBTEST_3(testExponentLaws(Matrix4cd(), 1e-13));
  CALL_SUBTEST_4(testExponentLaws(MatrixXd(8,8), 1e-13));
  CALL_SUBTEST_1(testExponentLaws(Matrix2f(), 1e-4));
  CALL_SUBTEST_5(testExponentLaws(Matrix3cf(), 1e-4));
  CALL_SUBTEST_1(testExponentLaws(Matrix4f(), 1e-4));
  CALL_SUBTEST_6(testExponentLaws(MatrixXf(8,8), 1e-4));
  CALL_SUBTEST_9(testExponentLaws(Matrix<long double,Dynamic,Dynamic>(7,7), 1e-13));
}
