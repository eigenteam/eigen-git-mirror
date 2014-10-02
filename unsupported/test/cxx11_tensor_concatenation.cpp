// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;

static void test_dimension_failures()
{
  Tensor<int, 3> left(2, 3, 1);
  Tensor<int, 3> right(3, 3, 1);
  left.setRandom();
  right.setRandom();

  // Okay; other dimensions are equal.
  Tensor<int, 3> concatenation = left.concatenate(right, 0);

  // Dimension mismatches.
  VERIFY_RAISES_ASSERT(concatenation = left.concatenate(right, 1));
  VERIFY_RAISES_ASSERT(concatenation = left.concatenate(right, 2));

  // Axis > NumDims or < 0.
  VERIFY_RAISES_ASSERT(concatenation = left.concatenate(right, 3));
  VERIFY_RAISES_ASSERT(concatenation = left.concatenate(right, -1));
}

static void test_static_dimension_failure()
{
  Tensor<int, 2> left(2, 3);
  Tensor<int, 3> right(2, 3, 1);

#ifdef CXX11_TENSOR_CONCATENATION_STATIC_DIMENSION_FAILURE
  // Technically compatible, but we static assert that the inputs have same
  // NumDims.
  Tensor<int, 3> concatenation = left.concatenate(right, 0);
#endif

  // This can be worked around in this case.
  Tensor<int, 3> concatenation = left
      .reshape(Tensor<int, 3>::Dimensions{{2, 3, 1}})
      .concatenate(right, 0);
  Tensor<int, 2> alternative = left
      .concatenate(right.reshape(Tensor<int, 2>::Dimensions{{2, 3}}), 0);
}

static void test_simple_concatenation()
{
  Tensor<int, 3> left(2, 3, 1);
  Tensor<int, 3> right(2, 3, 1);
  left.setRandom();
  right.setRandom();

  Tensor<int, 3> concatenation = left.concatenate(right, 0);
  VERIFY_IS_EQUAL(concatenation.dimension(0), 4);
  VERIFY_IS_EQUAL(concatenation.dimension(1), 3);
  VERIFY_IS_EQUAL(concatenation.dimension(2), 1);
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 2; ++i) {
      VERIFY_IS_EQUAL(concatenation(i, j, 0), left(i, j, 0));
    }
    for (int i = 2; i < 4; ++i) {
      VERIFY_IS_EQUAL(concatenation(i, j, 0), right(i - 2, j, 0));
    }
  }

  concatenation = left.concatenate(right, 1);
  VERIFY_IS_EQUAL(concatenation.dimension(0), 2);
  VERIFY_IS_EQUAL(concatenation.dimension(1), 6);
  VERIFY_IS_EQUAL(concatenation.dimension(2), 1);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      VERIFY_IS_EQUAL(concatenation(i, j, 0), left(i, j, 0));
    }
    for (int j = 3; j < 6; ++j) {
      VERIFY_IS_EQUAL(concatenation(i, j, 0), right(i, j - 3, 0));
    }
  }

  concatenation = left.concatenate(right, 2);
  VERIFY_IS_EQUAL(concatenation.dimension(0), 2);
  VERIFY_IS_EQUAL(concatenation.dimension(1), 3);
  VERIFY_IS_EQUAL(concatenation.dimension(2), 2);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      VERIFY_IS_EQUAL(concatenation(i, j, 0), left(i, j, 0));
      VERIFY_IS_EQUAL(concatenation(i, j, 1), right(i, j, 0));
    }
  }
}


// TODO(phli): Add test once we have a real vectorized implementation.
// static void test_vectorized_concatenation() {}


void test_cxx11_tensor_concatenation()
{
   CALL_SUBTEST(test_dimension_failures());
   CALL_SUBTEST(test_static_dimension_failure());
   CALL_SUBTEST(test_simple_concatenation());
   // CALL_SUBTEST(test_vectorized_concatenation());
}
