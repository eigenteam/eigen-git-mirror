// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
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

template<typename MatrixType> void product_extra(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::FloatingPoint FloatingPoint;
  typedef Matrix<Scalar, 1, Dynamic> RowVectorType;
  typedef Matrix<Scalar, Dynamic, 1> ColVectorType;
  typedef Matrix<Scalar, Dynamic, Dynamic,
                         MatrixType::Flags&RowMajorBit> OtherMajorMatrixType;

  int rows = m.rows();
  int cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols),
             m3(rows, cols),
             mzero = MatrixType::Zero(rows, cols),
             identity = MatrixType::Identity(rows, rows),
             square = MatrixType::Random(rows, rows),
             res = MatrixType::Random(rows, rows),
             square2 = MatrixType::Random(cols, cols),
             res2 = MatrixType::Random(cols, cols);
  RowVectorType v1 = RowVectorType::Random(rows),
             v2 = RowVectorType::Random(rows),
             vzero = RowVectorType::Zero(rows);
  ColVectorType vc2 = ColVectorType::Random(cols), vcres(cols);
  OtherMajorMatrixType tm1 = m1;

  Scalar s1 = ei_random<Scalar>(),
         s2 = ei_random<Scalar>(),
         s3 = ei_random<Scalar>();

  // all the expressions in this test should be compiled as a single matrix product
  // TODO: add internal checks to verify that
  
  VERIFY_IS_APPROX(m1 * m2.adjoint(),  m1 * m2.adjoint().eval());
  VERIFY_IS_APPROX(m1.adjoint() * square.adjoint(),  m1.adjoint().eval() * square.adjoint().eval());
  VERIFY_IS_APPROX(m1.adjoint() * m2,  m1.adjoint().eval() * m2);
  VERIFY_IS_APPROX( (s1 * m1.adjoint()) * m2,  (s1 * m1.adjoint()).eval() * m2);
  VERIFY_IS_APPROX( (- m1.adjoint() * s1) * (s3 * m2),  (- m1.adjoint()  * s1).eval() * (s3 * m2).eval());
  VERIFY_IS_APPROX( (s2 * m1.adjoint() * s1) * m2,  (s2 * m1.adjoint()  * s1).eval() * m2);
  VERIFY_IS_APPROX( (-m1*s2) * s1*m2.adjoint(), (-m1*s2).eval() * (s1*m2.adjoint()).eval());
  // a very tricky case where a scale factor has to be automatically conjugated:
  VERIFY_IS_APPROX( m1.adjoint() * (s1*m2).conjugate(), (m1.adjoint()).eval() * ((s1*m2).conjugate()).eval());


  // test all possible conjugate combinations for the four matrix-vector product cases:
  
//   std::cerr << "a\n";
  VERIFY_IS_APPROX((-m1.conjugate() * s2) * (s1 * vc2),
                   (-m1.conjugate()*s2).eval() * (s1 * vc2).eval());
  VERIFY_IS_APPROX((-m1 * s2) * (s1 * vc2.conjugate()),
                   (-m1*s2).eval() * (s1 * vc2.conjugate()).eval());
  VERIFY_IS_APPROX((-m1.conjugate() * s2) * (s1 * vc2.conjugate()),
                   (-m1.conjugate()*s2).eval() * (s1 * vc2.conjugate()).eval());

//   std::cerr << "b\n";
  VERIFY_IS_APPROX((s1 * vc2.transpose()) * (-m1.adjoint() * s2),
                   (s1 * vc2.transpose()).eval() * (-m1.adjoint()*s2).eval());
  VERIFY_IS_APPROX((s1 * vc2.adjoint()) * (-m1.transpose() * s2),
                   (s1 * vc2.adjoint()).eval() * (-m1.transpose()*s2).eval());
  VERIFY_IS_APPROX((s1 * vc2.adjoint()) * (-m1.adjoint() * s2),
                   (s1 * vc2.adjoint()).eval() * (-m1.adjoint()*s2).eval());

//   std::cerr << "c\n";
  VERIFY_IS_APPROX((-m1.adjoint() * s2) * (s1 * v1.transpose()),
                   (-m1.adjoint()*s2).eval() * (s1 * v1.transpose()).eval());
  VERIFY_IS_APPROX((-m1.transpose() * s2) * (s1 * v1.adjoint()),
                   (-m1.transpose()*s2).eval() * (s1 * v1.adjoint()).eval());
  VERIFY_IS_APPROX((-m1.adjoint() * s2) * (s1 * v1.adjoint()),
                   (-m1.adjoint()*s2).eval() * (s1 * v1.adjoint()).eval());

//   std::cerr << "d\n";
  VERIFY_IS_APPROX((s1 * v1) * (-m1.conjugate() * s2),
                   (s1 * v1).eval() * (-m1.conjugate()*s2).eval());
  VERIFY_IS_APPROX((s1 * v1.conjugate()) * (-m1 * s2),
                   (s1 * v1.conjugate()).eval() * (-m1*s2).eval());
  VERIFY_IS_APPROX((s1 * v1.conjugate()) * (-m1.conjugate() * s2),
                   (s1 * v1.conjugate()).eval() * (-m1.conjugate()*s2).eval());

  VERIFY_IS_APPROX((-m1.adjoint() * s2) * (s1 * v1.adjoint()),
                   (-m1.adjoint()*s2).eval() * (s1 * v1.adjoint()).eval());
}

void test_product_extra()
{
//   for(int i = 0; i < g_repeat; i++) {
//     CALL_SUBTEST( product_extra(MatrixXf(ei_random<int>(1,320), ei_random<int>(1,320))) );
//     CALL_SUBTEST( product(MatrixXd(ei_random<int>(1,320), ei_random<int>(1,320))) );
//     CALL_SUBTEST( product(MatrixXi(ei_random<int>(1,320), ei_random<int>(1,320))) );
    CALL_SUBTEST( product_extra(MatrixXcf(ei_random<int>(50,50), ei_random<int>(50,50))) );
//     CALL_SUBTEST( product(Matrix<float,Dynamic,Dynamic,RowMajor>(ei_random<int>(1,320), ei_random<int>(1,320))) );
//   }
}
