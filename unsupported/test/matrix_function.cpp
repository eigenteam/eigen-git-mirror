// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>
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
#include <unsupported/Eigen/MatrixFunctions>

// Returns a matrix with eigenvalues clustered around 0, 1 and 2.
template<typename MatrixType>
MatrixType randomMatrixWithRealEivals(const int size)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  MatrixType diag = MatrixType::Zero(size, size);
  for (int i = 0; i < size; ++i) {
    diag(i, i) = Scalar(RealScalar(ei_random<int>(0,2)))
      + ei_random<Scalar>() * Scalar(RealScalar(0.01));
  }
  MatrixType A = MatrixType::Random(size, size);
  return A.inverse() * diag * A;
}

template<typename MatrixType>
void testMatrixExponential(const MatrixType& A)
{
  typedef typename ei_traits<MatrixType>::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef std::complex<RealScalar> ComplexScalar;

  for (int i = 0; i < g_repeat; i++) {
    MatrixType expA1, expA2;
    ei_matrix_exponential(A, &expA1);
    ei_matrix_function(A, StdStemFunctions<ComplexScalar>::exp, &expA2);
    VERIFY_IS_APPROX(expA1, expA2);
  }
}

template<typename MatrixType>
void testHyperbolicFunctions(const MatrixType& A)
{
  for (int i = 0; i < g_repeat; i++) {
    MatrixType sinhA, coshA, expA;
    ei_matrix_sinh(A, &sinhA);
    ei_matrix_cosh(A, &coshA);
    ei_matrix_exponential(A, &expA);
    VERIFY_IS_APPROX(sinhA, (expA - expA.inverse())/2);
    VERIFY_IS_APPROX(coshA, (expA + expA.inverse())/2);
  }
}

template<typename MatrixType>
void testGonioFunctions(const MatrixType& A)
{
  typedef ei_traits<MatrixType> Traits;
  typedef typename Traits::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef std::complex<RealScalar> ComplexScalar;
  typedef Matrix<ComplexScalar, Traits::RowsAtCompileTime, 
                 Traits::ColsAtCompileTime, MatrixType::Options> ComplexMatrix;

  ComplexScalar imagUnit(0,1);
  ComplexScalar two(2,0);

  for (int i = 0; i < g_repeat; i++) {
    ComplexMatrix Ac = A.template cast<ComplexScalar>();

    ComplexMatrix exp_iA;
    ei_matrix_exponential(imagUnit * Ac, &exp_iA);

    MatrixType sinA;
    ei_matrix_sin(A, &sinA);
    ComplexMatrix sinAc = sinA.template cast<ComplexScalar>();
    VERIFY_IS_APPROX(sinAc, (exp_iA - exp_iA.inverse()) / (two*imagUnit));

    MatrixType cosA;
    ei_matrix_cos(A, &cosA);
    ComplexMatrix cosAc = cosA.template cast<ComplexScalar>();
    VERIFY_IS_APPROX(cosAc, (exp_iA + exp_iA.inverse()) / 2);
  }
}

template<typename MatrixType>
void testMatrix(const MatrixType& A)
{
  testMatrixExponential(A);
  testHyperbolicFunctions(A);
  testGonioFunctions(A);
}

template<typename MatrixType>
void testMatrixType(const MatrixType& m)
{
  // Matrices with clustered eigenvalue lead to different code paths
  // in MatrixFunction.h and are thus useful for testing.

  const int size = m.rows();
  for (int i = 0; i < g_repeat; i++) {
    testMatrix(MatrixType::Random(size, size).eval());
    testMatrix(randomMatrixWithRealEivals<MatrixType>(size));
  }
}

void test_matrix_function()
{
  CALL_SUBTEST_1(testMatrixType(Matrix<float,1,1>()));
  CALL_SUBTEST_2(testMatrixType(Matrix3cf()));
  CALL_SUBTEST_3(testMatrixType(MatrixXf(8,8)));
  CALL_SUBTEST_4(testMatrixType(Matrix2d()));
  CALL_SUBTEST_5(testMatrixType(Matrix<double,5,5,RowMajor>()));
  CALL_SUBTEST_6(testMatrixType(Matrix4cd()));
  CALL_SUBTEST_7(testMatrixType(MatrixXd(13,13)));
}
