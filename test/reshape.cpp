// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 yoco <peter.xiau@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

using Eigen::Map;
using Eigen::MatrixXi;

// just test a 4x4 matrix, enumerate all combination manually,
// so I don't have to do template-meta-programming here.
template <typename MatType>
void reshape_all_size(MatType m) {
  typedef Eigen::Map<MatrixXi> MapMat;
  // dynamic
  VERIFY_IS_EQUAL((m.template reshape( 1, 16)), MapMat(m.data(),  1, 16));
  VERIFY_IS_EQUAL((m.template reshape( 2,  8)), MapMat(m.data(),  2,  8));
  VERIFY_IS_EQUAL((m.template reshape( 4,  4)), MapMat(m.data(),  4,  4));
  VERIFY_IS_EQUAL((m.template reshape( 8,  2)), MapMat(m.data(),  8,  2));
  VERIFY_IS_EQUAL((m.template reshape(16,  1)), MapMat(m.data(), 16,  1));

  // static
  VERIFY_IS_EQUAL((m.template reshape< 1, 16>()), MapMat(m.data(),  1, 16));
  VERIFY_IS_EQUAL((m.template reshape< 2,  8>()), MapMat(m.data(),  2,  8));
  VERIFY_IS_EQUAL((m.template reshape< 4,  4>()), MapMat(m.data(),  4,  4));
  VERIFY_IS_EQUAL((m.template reshape< 8,  2>()), MapMat(m.data(),  8,  2));
  VERIFY_IS_EQUAL((m.template reshape<16,  1>()), MapMat(m.data(), 16,  1));

  // reshape chain
  VERIFY_IS_EQUAL(
    (m
     .template reshape( 1, 16)
     .template reshape< 2,  8>()
     .template reshape(16,  1)
     .template reshape< 8,  2>()
     .template reshape( 2,  8)
     .template reshape< 1, 16>()
     .template reshape( 4,  4)
     .template reshape<16,  1>()
     .template reshape( 8,  2)
     .template reshape< 4,  4>()
    ),
    MapMat(m.data(), 4,  4)
  );
}

void test_reshape()
{
  Eigen::MatrixXi mx = Eigen::MatrixXi::Random(4, 4);
  Eigen::Matrix4i m4 = Eigen::Matrix4i::Random(4, 4);

  // test dynamic-size matrix
  CALL_SUBTEST(reshape_all_size(mx));
  // test static-size matrix
  CALL_SUBTEST(reshape_all_size(m4));
  // test dynamic-size const matrix
  CALL_SUBTEST(reshape_all_size(static_cast<const Eigen::MatrixXi>(mx)));
  // test static-size const matrix
  CALL_SUBTEST(reshape_all_size(static_cast<const Eigen::Matrix4i>(m4)));
}
