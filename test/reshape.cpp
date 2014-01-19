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

template <typename MatType>
void dynamic_reshape_all_size(MatType m, int row, int col) {
  for(int new_r = 1; new_r <= row * col; ++new_r) {
    // skip invalid shape
    if(row * col % new_r != 0)
      continue;

    // reshape and compare
    int new_c = row * col / new_r;
    VERIFY_IS_EQUAL(m.reshape(new_r, new_c),
                    Map<MatrixXi>(m.data(), new_r, new_c));
  }
}

// just test a 4x4 matrix, enumerate all combination manually,
// so I don't have to do template-meta-programming here.
template <typename MatType>
void static_reshape_all_size(MatType m) {
  VERIFY_IS_EQUAL((m.template reshape< 1, 16>()), Map<MatrixXi>(m.data(),  1, 16));
  VERIFY_IS_EQUAL((m.template reshape< 2,  8>()), Map<MatrixXi>(m.data(),  2,  8));
  VERIFY_IS_EQUAL((m.template reshape< 4,  4>()), Map<MatrixXi>(m.data(),  4,  4));
  VERIFY_IS_EQUAL((m.template reshape< 8,  2>()), Map<MatrixXi>(m.data(),  8,  2));
  VERIFY_IS_EQUAL((m.template reshape<16,  1>()), Map<MatrixXi>(m.data(), 16,  1));
}

void test_reshape()
{
  // create matrix(row, col) and filled with 0, 1, 2, ...
  // for all possible shape
  int row = 4;
  int col = 4;
  Eigen::MatrixXi mx(row, col);
  Eigen::Matrix4i m4(row, col);

  for(int r = 0; r < row; ++r) {
    for(int c = 0; c < col; ++c) {
      mx(r, c) = col * c + r;
      m4(r, c) = col * c + r;
    }
  }

  mx.reshape(8, 2).leftCols(2);

  // test dynamic-size matrix
  CALL_SUBTEST(dynamic_reshape_all_size(mx, 4, 4));
  CALL_SUBTEST(static_reshape_all_size(mx));
  // test static-size matrix
  CALL_SUBTEST(dynamic_reshape_all_size(m4, 4, 4));
  CALL_SUBTEST(static_reshape_all_size(m4));

  // test dynamic-size const matrix
  CALL_SUBTEST(dynamic_reshape_all_size(static_cast<const Eigen::MatrixXi>(mx), 4, 4));
  CALL_SUBTEST(static_reshape_all_size(static_cast<const Eigen::MatrixXi>(mx)));
  // test static-size const matrix
  CALL_SUBTEST(dynamic_reshape_all_size(static_cast<const Eigen::Matrix4i>(m4), 4, 4));
  CALL_SUBTEST(static_reshape_all_size(static_cast<const Eigen::Matrix4i>(m4)));
}
