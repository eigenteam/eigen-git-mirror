// This file is triangularView of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@gmail.com>
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



template<typename MatrixType> void triangular_square(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;

  RealScalar largerEps = 10*test_precision<RealScalar>();

  int rows = m.rows();
  int cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols),
             m3(rows, cols),
             m4(rows, cols),
             r1(rows, cols),
             r2(rows, cols),
             mzero = MatrixType::Zero(rows, cols),
             mones = MatrixType::Ones(rows, cols),
             identity = Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime>
                              ::Identity(rows, rows),
             square = Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime>
                              ::Random(rows, rows);
  VectorType v1 = VectorType::Random(rows),
             v2 = VectorType::Random(rows),
             vzero = VectorType::Zero(rows);

  MatrixType m1up = m1.template triangularView<UpperTriangular>();
  MatrixType m2up = m2.template triangularView<UpperTriangular>();

  if (rows*cols>1)
  {
    VERIFY(m1up.isUpperTriangular());
    VERIFY(m2up.transpose().isLowerTriangular());
    VERIFY(!m2.isLowerTriangular());
  }

//   VERIFY_IS_APPROX(m1up.transpose() * m2, m1.upper().transpose().lower() * m2);

  // test overloaded operator+=
  r1.setZero();
  r2.setZero();
  r1.template triangularView<UpperTriangular>() +=  m1;
  r2 += m1up;
  VERIFY_IS_APPROX(r1,r2);

  // test overloaded operator=
  m1.setZero();
  m1.template triangularView<UpperTriangular>() = m2.transpose() + m2;
  m3 = m2.transpose() + m2;
  VERIFY_IS_APPROX(m3.template triangularView<LowerTriangular>().transpose().toDenseMatrix(), m1);

  // test overloaded operator=
  m1.setZero();
  m1.template triangularView<LowerTriangular>() = m2.transpose() + m2;
  VERIFY_IS_APPROX(m3.template triangularView<LowerTriangular>().toDenseMatrix(), m1);

  m1 = MatrixType::Random(rows, cols);
  for (int i=0; i<rows; ++i)
    while (ei_abs2(m1(i,i))<1e-1) m1(i,i) = ei_random<Scalar>();

  Transpose<MatrixType> trm4(m4);
  // test back and forward subsitution with a vector as the rhs
  m3 = m1.template triangularView<UpperTriangular>();
  VERIFY(v2.isApprox(m3.adjoint() * (m1.adjoint().template triangularView<LowerTriangular>().solve(v2)), largerEps));
  m3 = m1.template triangularView<LowerTriangular>();
  VERIFY(v2.isApprox(m3.transpose() * (m1.transpose().template triangularView<UpperTriangular>().solve(v2)), largerEps));
  m3 = m1.template triangularView<UpperTriangular>();
  VERIFY(v2.isApprox(m3 * (m1.template triangularView<UpperTriangular>().solve(v2)), largerEps));
  m3 = m1.template triangularView<LowerTriangular>();
  VERIFY(v2.isApprox(m3.conjugate() * (m1.conjugate().template triangularView<LowerTriangular>().solve(v2)), largerEps));

  // test back and forward subsitution with a matrix as the rhs
  m3 = m1.template triangularView<UpperTriangular>();
  VERIFY(m2.isApprox(m3.adjoint() * (m1.adjoint().template triangularView<LowerTriangular>().solve(m2)), largerEps));
  m3 = m1.template triangularView<LowerTriangular>();
  VERIFY(m2.isApprox(m3.transpose() * (m1.transpose().template triangularView<UpperTriangular>().solve(m2)), largerEps));
  m3 = m1.template triangularView<UpperTriangular>();
  VERIFY(m2.isApprox(m3 * (m1.template triangularView<UpperTriangular>().solve(m2)), largerEps));
  m3 = m1.template triangularView<LowerTriangular>();
  VERIFY(m2.isApprox(m3.conjugate() * (m1.conjugate().template triangularView<LowerTriangular>().solve(m2)), largerEps));

  // check M * inv(L) using in place API
  m4 = m3;
  m3.transpose().template triangularView<UpperTriangular>().solveInPlace(trm4);
  VERIFY(m4.cwise().abs().isIdentity(test_precision<RealScalar>()));

  // check M * inv(U) using in place API
  m3 = m1.template triangularView<UpperTriangular>();
  m4 = m3;
  m3.transpose().template triangularView<LowerTriangular>().solveInPlace(trm4);
  VERIFY(m4.cwise().abs().isIdentity(test_precision<RealScalar>()));

  // check solve with unit diagonal
  m3 = m1.template triangularView<UnitUpperTriangular>();
  VERIFY(m2.isApprox(m3 * (m1.template triangularView<UnitUpperTriangular>().solve(m2)), largerEps));

//   VERIFY((  m1.template triangularView<UpperTriangular>()
//           * m2.template triangularView<UpperTriangular>()).isUpperTriangular());

  // test swap
  m1.setOnes();
  m2.setZero();
  m2.template triangularView<UpperTriangular>().swap(m1);
  m3.setZero();
  m3.template triangularView<UpperTriangular>().setOnes();
  VERIFY_IS_APPROX(m2,m3);

}


template<typename MatrixType> void triangular_rect(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  enum { Rows =  MatrixType::RowsAtCompileTime, Cols =  MatrixType::ColsAtCompileTime };
  typedef Matrix<Scalar, Rows, 1> VectorType;
  typedef Matrix<Scalar, Rows, Rows> RMatrixType;
  

  int rows = m.rows();
  int cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols),
             m3(rows, cols),
             m4(rows, cols),
             r1(rows, cols),
             r2(rows, cols),
             mzero = MatrixType::Zero(rows, cols),
             mones = MatrixType::Ones(rows, cols);
  RMatrixType identity = Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime>
                              ::Identity(rows, rows),
              square = Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime>
                              ::Random(rows, rows);
  VectorType v1 = VectorType::Random(rows),
             v2 = VectorType::Random(rows),
             vzero = VectorType::Zero(rows);

  MatrixType m1up = m1.template triangularView<UpperTriangular>();
  MatrixType m2up = m2.template triangularView<UpperTriangular>();

  if (rows*cols>1)
  {
    VERIFY(m1up.isUpperTriangular());
    VERIFY(m2up.transpose().isLowerTriangular());
    VERIFY(!m2.isLowerTriangular());
  }

  // test overloaded operator+=
  r1.setZero();
  r2.setZero();
  r1.template triangularView<UpperTriangular>() +=  m1;
  r2 += m1up;
  VERIFY_IS_APPROX(r1,r2);

  // test overloaded operator=
  m1.setZero();
  m1.template triangularView<UpperTriangular>() = 3 * m2;
  m3 = 3 * m2;
  VERIFY_IS_APPROX(m3.template triangularView<UpperTriangular>().toDenseMatrix(), m1);


  m1.setZero();
  m1.template triangularView<LowerTriangular>() = 3 * m2;
  VERIFY_IS_APPROX(m3.template triangularView<LowerTriangular>().toDenseMatrix(), m1);

  m1.setZero();
  m1.template triangularView<StrictlyUpperTriangular>() = 3 * m2;
  VERIFY_IS_APPROX(m3.template triangularView<StrictlyUpperTriangular>().toDenseMatrix(), m1);


  m1.setZero();
  m1.template triangularView<StrictlyLowerTriangular>() = 3 * m2;
  VERIFY_IS_APPROX(m3.template triangularView<StrictlyLowerTriangular>().toDenseMatrix(), m1);
  m1.setRandom();
  m2 = m1.template triangularView<UpperTriangular>();
  VERIFY(m2.isUpperTriangular());
  VERIFY(!m2.isLowerTriangular());
  m2 = m1.template triangularView<StrictlyUpperTriangular>();
  VERIFY(m2.isUpperTriangular());
  VERIFY(m2.diagonal().isMuchSmallerThan(RealScalar(1)));
  m2 = m1.template triangularView<UnitUpperTriangular>();
  VERIFY(m2.isUpperTriangular());
  m2.diagonal().cwise() -= Scalar(1);
  VERIFY(m2.diagonal().isMuchSmallerThan(RealScalar(1)));
  m2 = m1.template triangularView<LowerTriangular>();
  VERIFY(m2.isLowerTriangular());
  VERIFY(!m2.isUpperTriangular());
  m2 = m1.template triangularView<StrictlyLowerTriangular>();
  VERIFY(m2.isLowerTriangular());
  VERIFY(m2.diagonal().isMuchSmallerThan(RealScalar(1)));
  m2 = m1.template triangularView<UnitLowerTriangular>();
  VERIFY(m2.isLowerTriangular());
  m2.diagonal().cwise() -= Scalar(1);
  VERIFY(m2.diagonal().isMuchSmallerThan(RealScalar(1)));
  // test swap
  m1.setOnes();
  m2.setZero();
  m2.template triangularView<UpperTriangular>().swap(m1);
  m3.setZero();
  m3.template triangularView<UpperTriangular>().setOnes();
  VERIFY_IS_APPROX(m2,m3);
}

void test_triangular()
{
  for(int i = 0; i < g_repeat ; i++)
  {
    EIGEN_UNUSED int r = ei_random<int>(2,20);
    EIGEN_UNUSED int c = ei_random<int>(2,20);

    CALL_SUBTEST_1( triangular_square(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_2( triangular_square(Matrix<float, 2, 2>()) );
    CALL_SUBTEST_3( triangular_square(Matrix3d()) );
    CALL_SUBTEST_4( triangular_square(Matrix<std::complex<float>,8, 8>()) );
    CALL_SUBTEST_5( triangular_square(MatrixXcd(r,r)) );
    CALL_SUBTEST_6( triangular_square(Matrix<float,Dynamic,Dynamic,RowMajor>(r, r)) );

    CALL_SUBTEST_7( triangular_rect(Matrix<float, 4, 5>()) );
    CALL_SUBTEST_8( triangular_rect(Matrix<double, 6, 2>()) );
    CALL_SUBTEST_9( triangular_rect(MatrixXcf(r, c)) );
    CALL_SUBTEST_5( triangular_rect(MatrixXcd(r, c)) );
    CALL_SUBTEST_6( triangular_rect(Matrix<float,Dynamic,Dynamic,RowMajor>(r, c)) );
  }
}
