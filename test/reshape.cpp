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

void dynamic_reshape_all_size(int row, int col) {
  // create matrix(row, col) and filled with 0, 1, 2, ...
  Eigen::MatrixXi m(row, col);
  for(int r = 0; r < row; ++r)
    for(int c = 0; c < col; ++c)
      m(r, c) = col * c + r;

  // for all possible shape
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
void static_reshape_all_size() {
  // create matrix(row, col) and filled with 0, 1, 2, ...
  int row = 4;
  int col = 4;
  Eigen::MatrixXi m(row, col);
  for(int r = 0; r < row; ++r)
    for(int c = 0; c < col; ++c)
      m(r, c) = col * c + r;

  // reshape and compare
  VERIFY_IS_EQUAL((m.reshape< 1, 16>()), Map<MatrixXi>(m.data(),  1, 16));
  VERIFY_IS_EQUAL((m.reshape< 2,  8>()), Map<MatrixXi>(m.data(),  2,  8));
  VERIFY_IS_EQUAL((m.reshape< 4,  4>()), Map<MatrixXi>(m.data(),  4,  4));
  VERIFY_IS_EQUAL((m.reshape< 8,  2>()), Map<MatrixXi>(m.data(),  8,  2));
  VERIFY_IS_EQUAL((m.reshape<16,  1>()), Map<MatrixXi>(m.data(), 16,  1));
}

void test_reshape()
{
  CALL_SUBTEST(dynamic_reshape_all_size(4, 4));
  CALL_SUBTEST(static_reshape_all_size());
}
