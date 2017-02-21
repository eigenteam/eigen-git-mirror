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

template <typename MatType,typename OrderType>
void check_auto_reshape4x4(MatType m,OrderType order)
{
  internal::VariableAndFixedInt<MatType::SizeAtCompileTime==Dynamic?-1: 1>  v1( 1);
  internal::VariableAndFixedInt<MatType::SizeAtCompileTime==Dynamic?-1: 2>  v2( 2);
  internal::VariableAndFixedInt<MatType::SizeAtCompileTime==Dynamic?-1: 4>  v4( 4);
  internal::VariableAndFixedInt<MatType::SizeAtCompileTime==Dynamic?-1: 8>  v8( 8);
  internal::VariableAndFixedInt<MatType::SizeAtCompileTime==Dynamic?-1:16> v16(16);

  VERIFY(is_same_eq(m.reshaped( 1,       AutoSize, order), m.reshaped( 1, 16, order)));
  VERIFY(is_same_eq(m.reshaped(AutoSize, 16,       order), m.reshaped( 1, 16, order)));
  VERIFY(is_same_eq(m.reshaped( 2,       AutoSize, order), m.reshaped( 2,  8, order)));
  VERIFY(is_same_eq(m.reshaped(AutoSize, 8,        order), m.reshaped( 2,  8, order)));
  VERIFY(is_same_eq(m.reshaped( 4,       AutoSize, order), m.reshaped( 4,  4, order)));
  VERIFY(is_same_eq(m.reshaped(AutoSize, 4,        order), m.reshaped( 4,  4, order)));
  VERIFY(is_same_eq(m.reshaped( 8,       AutoSize, order), m.reshaped( 8,  2, order)));
  VERIFY(is_same_eq(m.reshaped(AutoSize, 2,        order), m.reshaped( 8,  2, order)));
  VERIFY(is_same_eq(m.reshaped(16,       AutoSize, order), m.reshaped(16,  1, order)));
  VERIFY(is_same_eq(m.reshaped(AutoSize, 1,       order),  m.reshaped(16,  1, order)));

  VERIFY(is_same_eq(m.reshaped(fix< 1>,   AutoSize, order),  m.reshaped(fix< 1>, v16,     order)));
  VERIFY(is_same_eq(m.reshaped(AutoSize,  fix<16>,  order),  m.reshaped( v1,     fix<16>, order)));
  VERIFY(is_same_eq(m.reshaped(fix< 2>,   AutoSize, order),  m.reshaped(fix< 2>, v8,      order)));
  VERIFY(is_same_eq(m.reshaped(AutoSize,  fix< 8>,  order),  m.reshaped( v2,     fix< 8>, order)));
  VERIFY(is_same_eq(m.reshaped(fix< 4>,   AutoSize, order),  m.reshaped(fix< 4>, v4,      order)));
  VERIFY(is_same_eq(m.reshaped(AutoSize,  fix< 4>,  order),  m.reshaped( v4,     fix< 4>, order)));
  VERIFY(is_same_eq(m.reshaped(fix< 8>,   AutoSize, order),  m.reshaped(fix< 8>, v2,      order)));
  VERIFY(is_same_eq(m.reshaped(AutoSize,  fix< 2>,  order),  m.reshaped( v8,     fix< 2>, order)));
  VERIFY(is_same_eq(m.reshaped(fix<16>,   AutoSize, order),  m.reshaped(fix<16>, v1,      order)));
  VERIFY(is_same_eq(m.reshaped(AutoSize,  fix< 1>,  order),  m.reshaped(v16,     fix< 1>, order)));
}

// just test a 4x4 matrix, enumerate all combination manually
template <typename MatType>
void reshape4x4(MatType m)
{
  internal::VariableAndFixedInt<MatType::SizeAtCompileTime==Dynamic?-1: 1>  v1( 1);
  internal::VariableAndFixedInt<MatType::SizeAtCompileTime==Dynamic?-1: 2>  v2( 2);
  internal::VariableAndFixedInt<MatType::SizeAtCompileTime==Dynamic?-1: 4>  v4( 4);
  internal::VariableAndFixedInt<MatType::SizeAtCompileTime==Dynamic?-1: 8>  v8( 8);
  internal::VariableAndFixedInt<MatType::SizeAtCompileTime==Dynamic?-1:16> v16(16);

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

  VERIFY(is_same_eq(m.reshaped( 1,       AutoSize), m.reshaped( 1, 16)));
  VERIFY(is_same_eq(m.reshaped(AutoSize, 16),       m.reshaped( 1, 16)));
  VERIFY(is_same_eq(m.reshaped( 2,       AutoSize), m.reshaped( 2,  8)));
  VERIFY(is_same_eq(m.reshaped(AutoSize, 8),        m.reshaped( 2,  8)));
  VERIFY(is_same_eq(m.reshaped( 4,       AutoSize), m.reshaped( 4,  4)));
  VERIFY(is_same_eq(m.reshaped(AutoSize, 4),        m.reshaped( 4,  4)));
  VERIFY(is_same_eq(m.reshaped( 8,       AutoSize), m.reshaped( 8,  2)));
  VERIFY(is_same_eq(m.reshaped(AutoSize, 2),        m.reshaped( 8,  2)));
  VERIFY(is_same_eq(m.reshaped(16,       AutoSize), m.reshaped(16,  1)));
  VERIFY(is_same_eq(m.reshaped(AutoSize,  1),       m.reshaped(16,  1)));

  VERIFY(is_same_eq(m.reshaped(fix< 1>,   AutoSize),  m.reshaped(fix< 1>, v16)));
  VERIFY(is_same_eq(m.reshaped(AutoSize,  fix<16>),   m.reshaped( v1,     fix<16>)));
  VERIFY(is_same_eq(m.reshaped(fix< 2>,   AutoSize),  m.reshaped(fix< 2>, v8)));
  VERIFY(is_same_eq(m.reshaped(AutoSize,  fix< 8>),   m.reshaped( v2,     fix< 8>)));
  VERIFY(is_same_eq(m.reshaped(fix< 4>,   AutoSize),  m.reshaped(fix< 4>, v4)));
  VERIFY(is_same_eq(m.reshaped(AutoSize,  fix< 4>),   m.reshaped( v4,     fix< 4>)));
  VERIFY(is_same_eq(m.reshaped(fix< 8>,   AutoSize),  m.reshaped(fix< 8>, v2)));
  VERIFY(is_same_eq(m.reshaped(AutoSize,  fix< 2>),   m.reshaped( v8,     fix< 2>)));
  VERIFY(is_same_eq(m.reshaped(fix<16>,   AutoSize),  m.reshaped(fix<16>, v1)));
  VERIFY(is_same_eq(m.reshaped(AutoSize,  fix< 1>),   m.reshaped(v16,     fix< 1>)));

  check_auto_reshape4x4(m,ColOrder);
  check_auto_reshape4x4(m,RowOrder);
  check_auto_reshape4x4(m,AutoOrder);
  check_auto_reshape4x4(m.transpose(),ColOrder);
  check_auto_reshape4x4(m.transpose(),RowOrder);
  check_auto_reshape4x4(m.transpose(),AutoOrder);

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
  VERIFY(is_same_eq(m.reshaped(v16,fix<1>), m(all)));
  VERIFY_IS_EQUAL(m.reshaped(16,1), m(all));
  VERIFY_IS_EQUAL(m.reshaped(1,16), m(all).transpose());
  VERIFY_IS_EQUAL(m(all).reshaped(2,8), m.reshaped(2,8));
  VERIFY_IS_EQUAL(m(all).reshaped(4,4), m.reshaped(4,4));
  VERIFY_IS_EQUAL(m(all).reshaped(8,2), m.reshaped(8,2));

  VERIFY(is_same_eq(m.reshaped(AutoSize,fix<1>), m(all)));
  VERIFY_IS_EQUAL(m.reshaped(fix<1>,AutoSize,RowOrder), m.transpose()(all).transpose());
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
