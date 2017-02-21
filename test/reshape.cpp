// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2014 yoco <peter.xiau@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template<typename T1,typename T2>
typename internal::enable_if<internal::is_same<T1,T2>::value,bool>::type
is_same_eq(const T1& a, const T2& b)
{
  return (a.array() == b.array()).all();
}

// just test a 4x4 matrix, enumerate all combination manually
template <typename MatType>
void reshape4x4(MatType m)
{
  if((MatType::Flags&RowMajorBit)==0)
  {
    typedef Map<MatrixXi> MapMat;
    // dynamic
    VERIFY_IS_EQUAL((m.reshaped( 1, 16)), MapMat(m.data(),  1, 16));
    VERIFY_IS_EQUAL((m.reshaped( 2,  8)), MapMat(m.data(),  2,  8));
    VERIFY_IS_EQUAL((m.reshaped( 4,  4)), MapMat(m.data(),  4,  4));
    VERIFY_IS_EQUAL((m.reshaped( 8,  2)), MapMat(m.data(),  8,  2));
    VERIFY_IS_EQUAL((m.reshaped(16,  1)), MapMat(m.data(), 16,  1));

    // static
    VERIFY_IS_EQUAL(m.reshaped(fix< 1>, fix<16>), MapMat(m.data(),  1, 16));
    VERIFY_IS_EQUAL(m.reshaped(fix< 2>, fix< 8>), MapMat(m.data(),  2,  8));
    VERIFY_IS_EQUAL(m.reshaped(fix< 4>, fix< 4>), MapMat(m.data(),  4,  4));
    VERIFY_IS_EQUAL(m.reshaped(fix< 8>, fix< 2>), MapMat(m.data(),  8,  2));
    VERIFY_IS_EQUAL(m.reshaped(fix<16>, fix< 1>), MapMat(m.data(), 16,  1));

    // reshape chain
    VERIFY_IS_EQUAL(
      (m
      .reshaped( 1, 16)
      .reshaped(fix< 2>,fix< 8>)
      .reshaped(16,  1)
      .reshaped(fix< 8>,fix< 2>)
      .reshaped( 2,  8)
      .reshaped(fix< 1>,fix<16>)
      .reshaped( 4,  4)
      .reshaped(fix<16>,fix< 1>)
      .reshaped( 8,  2)
      .reshaped(fix< 4>,fix< 4>)
      ),
      MapMat(m.data(), 4,  4)
    );
  }

  VERIFY_IS_EQUAL(m.reshaped( 1, 16).data(), m.data());
  VERIFY_IS_EQUAL(m.reshaped( 1, 16).innerStride(), 1);

  VERIFY_IS_EQUAL(m.reshaped( 2, 8).data(), m.data());
  VERIFY_IS_EQUAL(m.reshaped( 2, 8).innerStride(), 1);
  VERIFY_IS_EQUAL(m.reshaped( 2, 8).outerStride(), 2);

  if((MatType::Flags&RowMajorBit)==0)
  {
    VERIFY_IS_EQUAL(m.reshaped(2,8,ColOrder),m.reshaped(2,8));
    VERIFY_IS_EQUAL(m.reshaped(2,8,ColOrder),m.reshaped(2,8,AutoOrder));
    VERIFY_IS_EQUAL(m.transpose().reshaped(2,8,RowOrder),m.transpose().reshaped(2,8,AutoOrder));
  }
  else
  {
    VERIFY_IS_EQUAL(m.reshaped(2,8,ColOrder),m.reshaped(2,8));
    VERIFY_IS_EQUAL(m.reshaped(2,8,RowOrder),m.reshaped(2,8,AutoOrder));
    VERIFY_IS_EQUAL(m.transpose().reshaped(2,8,ColOrder),m.transpose().reshaped(2,8,AutoOrder));
    VERIFY_IS_EQUAL(m.transpose().reshaped(2,8),m.transpose().reshaped(2,8,AutoOrder));
  }

  MatrixXi m28r1 = m.reshaped(2,8,RowOrder);
  MatrixXi m28r2 = m.transpose().reshaped(8,2,ColOrder).transpose();
  VERIFY_IS_EQUAL( m28r1, m28r2);

  using placeholders::all;
  VERIFY(is_same_eq(m.reshaped(fix<MatType::SizeAtCompileTime>(m.size()),fix<1>), m(all)));
  VERIFY_IS_EQUAL(m.reshaped(16,1), m(all));
  VERIFY_IS_EQUAL(m.reshaped(1,16), m(all).transpose());
  VERIFY_IS_EQUAL(m(all).reshaped(2,8), m.reshaped(2,8));
  VERIFY_IS_EQUAL(m(all).reshaped(4,4), m.reshaped(4,4));
  VERIFY_IS_EQUAL(m(all).reshaped(8,2), m.reshaped(8,2));
}

void test_reshape()
{
  typedef Matrix<int,Dynamic,Dynamic> RowMatrixXi;
  typedef Matrix<int,4,4> RowMatrix4i;
  MatrixXi mx = MatrixXi::Random(4, 4);
  Matrix4i m4 = Matrix4i::Random(4, 4);
  RowMatrixXi rmx = RowMatrixXi::Random(4, 4);
  RowMatrix4i rm4 = RowMatrix4i::Random(4, 4);

  // test dynamic-size matrix
  CALL_SUBTEST(reshape4x4(mx));
  // test static-size matrix
  CALL_SUBTEST(reshape4x4(m4));
  // test dynamic-size const matrix
  CALL_SUBTEST(reshape4x4(static_cast<const MatrixXi>(mx)));
  // test static-size const matrix
  CALL_SUBTEST(reshape4x4(static_cast<const Matrix4i>(m4)));

  CALL_SUBTEST(reshape4x4(rmx));
  CALL_SUBTEST(reshape4x4(rm4));
}
