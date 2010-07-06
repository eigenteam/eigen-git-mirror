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

template<typename MatrixType> void product_extra(const MatrixType& m)
{
  typedef typename MatrixType::Index Index;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::NonInteger NonInteger;
  typedef Matrix<Scalar, 1, Dynamic> RowVectorType;
  typedef Matrix<Scalar, Dynamic, 1> ColVectorType;
  typedef Matrix<Scalar, Dynamic, Dynamic,
                         MatrixType::Flags&RowMajorBit> OtherMajorMatrixType;

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols),
             m3(rows, cols),
             mzero = MatrixType::Zero(rows, cols),
             identity = MatrixType::Identity(rows, rows),
             square = MatrixType::Random(rows, rows),
             res = MatrixType::Random(rows, rows),
             square2 = MatrixType::Random(cols, cols),
             res2 = MatrixType::Random(cols, cols);
  RowVectorType v1 = RowVectorType::Random(rows), vrres(rows);
  ColVectorType vc2 = ColVectorType::Random(cols), vcres(cols);
  OtherMajorMatrixType tm1 = m1;

  Scalar s1 = ei_random<Scalar>(),
         s2 = ei_random<Scalar>(),
         s3 = ei_random<Scalar>();

  VERIFY_IS_APPROX(m3.noalias() = m1 * m2.adjoint(),                 m1 * m2.adjoint().eval());
  VERIFY_IS_APPROX(m3.noalias() = m1.adjoint() * square.adjoint(),   m1.adjoint().eval() * square.adjoint().eval());
  VERIFY_IS_APPROX(m3.noalias() = m1.adjoint() * m2,                 m1.adjoint().eval() * m2);
  VERIFY_IS_APPROX(m3.noalias() = (s1 * m1.adjoint()) * m2,          (s1 * m1.adjoint()).eval() * m2);
  VERIFY_IS_APPROX(m3.noalias() = ((s1 * m1).adjoint()) * m2,        (ei_conj(s1) * m1.adjoint()).eval() * m2);
  VERIFY_IS_APPROX(m3.noalias() = (- m1.adjoint() * s1) * (s3 * m2), (- m1.adjoint()  * s1).eval() * (s3 * m2).eval());
  VERIFY_IS_APPROX(m3.noalias() = (s2 * m1.adjoint() * s1) * m2,     (s2 * m1.adjoint()  * s1).eval() * m2);
  VERIFY_IS_APPROX(m3.noalias() = (-m1*s2) * s1*m2.adjoint(),        (-m1*s2).eval() * (s1*m2.adjoint()).eval());

  // a very tricky case where a scale factor has to be automatically conjugated:
  VERIFY_IS_APPROX( m1.adjoint() * (s1*m2).conjugate(), (m1.adjoint()).eval() * ((s1*m2).conjugate()).eval());


  // test all possible conjugate combinations for the four matrix-vector product cases:

  VERIFY_IS_APPROX((-m1.conjugate() * s2) * (s1 * vc2),
                   (-m1.conjugate()*s2).eval() * (s1 * vc2).eval());
  VERIFY_IS_APPROX((-m1 * s2) * (s1 * vc2.conjugate()),
                   (-m1*s2).eval() * (s1 * vc2.conjugate()).eval());
  VERIFY_IS_APPROX((-m1.conjugate() * s2) * (s1 * vc2.conjugate()),
                   (-m1.conjugate()*s2).eval() * (s1 * vc2.conjugate()).eval());

  VERIFY_IS_APPROX((s1 * vc2.transpose()) * (-m1.adjoint() * s2),
                   (s1 * vc2.transpose()).eval() * (-m1.adjoint()*s2).eval());
  VERIFY_IS_APPROX((s1 * vc2.adjoint()) * (-m1.transpose() * s2),
                   (s1 * vc2.adjoint()).eval() * (-m1.transpose()*s2).eval());
  VERIFY_IS_APPROX((s1 * vc2.adjoint()) * (-m1.adjoint() * s2),
                   (s1 * vc2.adjoint()).eval() * (-m1.adjoint()*s2).eval());

  VERIFY_IS_APPROX((-m1.adjoint() * s2) * (s1 * v1.transpose()),
                   (-m1.adjoint()*s2).eval() * (s1 * v1.transpose()).eval());
  VERIFY_IS_APPROX((-m1.transpose() * s2) * (s1 * v1.adjoint()),
                   (-m1.transpose()*s2).eval() * (s1 * v1.adjoint()).eval());
  VERIFY_IS_APPROX((-m1.adjoint() * s2) * (s1 * v1.adjoint()),
                   (-m1.adjoint()*s2).eval() * (s1 * v1.adjoint()).eval());

  VERIFY_IS_APPROX((s1 * v1) * (-m1.conjugate() * s2),
                   (s1 * v1).eval() * (-m1.conjugate()*s2).eval());
  VERIFY_IS_APPROX((s1 * v1.conjugate()) * (-m1 * s2),
                   (s1 * v1.conjugate()).eval() * (-m1*s2).eval());
  VERIFY_IS_APPROX((s1 * v1.conjugate()) * (-m1.conjugate() * s2),
                   (s1 * v1.conjugate()).eval() * (-m1.conjugate()*s2).eval());

  VERIFY_IS_APPROX((-m1.adjoint() * s2) * (s1 * v1.adjoint()),
                   (-m1.adjoint()*s2).eval() * (s1 * v1.adjoint()).eval());

  // test the vector-matrix product with non aligned starts
  Index i = ei_random<Index>(0,m1.rows()-2);
  Index j = ei_random<Index>(0,m1.cols()-2);
  Index r = ei_random<Index>(1,m1.rows()-i);
  Index c = ei_random<Index>(1,m1.cols()-j);
  Index i2 = ei_random<Index>(0,m1.rows()-1);
  Index j2 = ei_random<Index>(0,m1.cols()-1);

  VERIFY_IS_APPROX(m1.col(j2).adjoint() * m1.block(0,j,m1.rows(),c), m1.col(j2).adjoint().eval() * m1.block(0,j,m1.rows(),c).eval());
  VERIFY_IS_APPROX(m1.block(i,0,r,m1.cols()) * m1.row(i2).adjoint(), m1.block(i,0,r,m1.cols()).eval() * m1.row(i2).adjoint().eval());
}

void test_product_extra()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( product_extra(MatrixXf(ei_random<int>(2,320), ei_random<int>(2,320))) );
    CALL_SUBTEST_2( product_extra(MatrixXcf(ei_random<int>(2,50), ei_random<int>(2,50))) );
    CALL_SUBTEST_3( product_extra(Matrix<std::complex<double>,Dynamic,Dynamic,RowMajor>(ei_random<int>(2,50), ei_random<int>(2,50))) );
  }
}
