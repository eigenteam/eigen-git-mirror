// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014-2015 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/CXX11/Tensor>


void test_signed_32bit()
{
  for (int32_t i = 2; i < 25000; ++i) {
    const Eigen::internal::TensorIntDivisor<int32_t> div(i);

    for (int32_t j = 0; j < 25000; ++j) {
      const int32_t fast_div = j / div;
      const int32_t slow_div = j / i;
      VERIFY_IS_EQUAL(fast_div, slow_div);
    }
  }
}


void test_unsigned_32bit()
{
  for (uint32_t i = 1; i < 25000; ++i) {
    const Eigen::internal::TensorIntDivisor<uint32_t> div(i);

    for (uint32_t j = 0; j < 25000; ++j) {
      const uint32_t fast_div = j / div;
      const uint32_t slow_div = j / i;
      VERIFY_IS_EQUAL(fast_div, slow_div);
    }
  }
}


void test_signed_64bit()
{
  for (int64_t i = 2; i < 25000; ++i) {
    const Eigen::internal::TensorIntDivisor<int64_t> div(i);

    for (int64_t j = 0; j < 25000; ++j) {
      const int64_t fast_div = j / div;
      const int64_t slow_div = j / i;
      VERIFY_IS_EQUAL(fast_div, slow_div);
    }
  }
}


void test_unsigned_64bit()
{
  for (uint64_t i = 2; i < 25000; ++i) {
    const Eigen::internal::TensorIntDivisor<uint64_t> div(i);

    for (uint64_t j = 0; j < 25000; ++j) {
      const uint64_t fast_div = j / div;
      const uint64_t slow_div = j / i;
      VERIFY_IS_EQUAL(fast_div, slow_div);
    }
  }
}


void test_specific()
{
  // A particular combination that exposed a bug in the past.
  int64_t div = 209715200;
  int64_t num = 3238002688;
  Eigen::internal::TensorIntDivisor<int64_t> divider =
      Eigen::internal::TensorIntDivisor<int64_t>(div);
  int64_t result = num/div;
  int64_t result_op = divider.divide(num);
  VERIFY_IS_EQUAL(result, result_op);
}

void test_cxx11_tensor_intdiv()
{
  CALL_SUBTEST_1(test_signed_32bit());
  CALL_SUBTEST_2(test_unsigned_32bit());
  CALL_SUBTEST_3(test_signed_64bit());
  CALL_SUBTEST_4(test_unsigned_64bit());
  CALL_SUBTEST_5(test_specific());
}
