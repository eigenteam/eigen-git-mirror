// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2018 Eugene Zhulenev <ezhulenev@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_USE_THREADS

#include "main.h"

#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;
using Eigen::RowMajor;
using Eigen::ColMajor;

// A set of tests to verify that different TensorExecutor strategies yields the
// same results for all the ops, supporting tiled execution.

template <typename Device, bool Vectorizable, bool Tileable, int Layout>
static void test_execute_binary_expr(Device d) {
  // Pick a large enough tensor size to bypass small tensor block evaluation
  // optimization.
  int d0 = internal::random<int>(100, 200);
  int d1 = internal::random<int>(100, 200);
  int d2 = internal::random<int>(100, 200);

  static constexpr int Options = 0;
  using IndexType = int;

  Tensor<float, 3, Options, IndexType> lhs(d0, d1, d2);
  Tensor<float, 3, Options, IndexType> rhs(d0, d1, d2);
  Tensor<float, 3, Options, IndexType> dst(d0, d1, d2);

  lhs.setRandom();
  rhs.setRandom();

  const auto expr = lhs + rhs;

  using Assign = TensorAssignOp<decltype(dst), const decltype(expr)>;
  using Executor =
      internal::TensorExecutor<const Assign, Device, Vectorizable, Tileable>;

  Executor::run(Assign(dst, expr), d);

  for (int i = 0; i < d0; ++i) {
    for (int j = 0; j < d1; ++j) {
      for (int k = 0; k < d2; ++k) {
        float sum = lhs(i, j, k) + rhs(i, j, k);
        VERIFY_IS_EQUAL(sum, dst(i, j, k));
      }
    }
  }
}

#define CALL_SUBTEST_COMBINATIONS(NAME)                                        \
  CALL_SUBTEST((NAME<DefaultDevice, false, false, ColMajor>(default_device))); \
  CALL_SUBTEST((NAME<DefaultDevice, false, true, ColMajor>(default_device)));  \
  CALL_SUBTEST((NAME<DefaultDevice, true, false, ColMajor>(default_device)));  \
  CALL_SUBTEST((NAME<DefaultDevice, true, true, ColMajor>(default_device)));   \
  CALL_SUBTEST((NAME<DefaultDevice, false, false, RowMajor>(default_device))); \
  CALL_SUBTEST((NAME<DefaultDevice, false, true, RowMajor>(default_device)));  \
  CALL_SUBTEST((NAME<DefaultDevice, true, false, RowMajor>(default_device)));  \
  CALL_SUBTEST((NAME<DefaultDevice, true, true, RowMajor>(default_device)));   \
  CALL_SUBTEST((NAME<ThreadPoolDevice, false, false, ColMajor>(tp_device)));   \
  CALL_SUBTEST((NAME<ThreadPoolDevice, false, true, ColMajor>(tp_device)));    \
  CALL_SUBTEST((NAME<ThreadPoolDevice, true, false, ColMajor>(tp_device)));    \
  CALL_SUBTEST((NAME<ThreadPoolDevice, true, true, ColMajor>(tp_device)));     \
  CALL_SUBTEST((NAME<ThreadPoolDevice, false, false, RowMajor>(tp_device)));   \
  CALL_SUBTEST((NAME<ThreadPoolDevice, false, true, RowMajor>(tp_device)));    \
  CALL_SUBTEST((NAME<ThreadPoolDevice, true, false, RowMajor>(tp_device)));    \
  CALL_SUBTEST((NAME<ThreadPoolDevice, true, true, RowMajor>(tp_device)))

EIGEN_DECLARE_TEST(cxx11_tensor_executor) {
  Eigen::DefaultDevice default_device;

  const auto num_threads = internal::random<int>(1, 24);
  Eigen::ThreadPool tp(num_threads);
  Eigen::ThreadPoolDevice tp_device(&tp, num_threads);

  CALL_SUBTEST_COMBINATIONS(test_execute_binary_expr);
}

#undef CALL_SUBTEST_COMBINATIONS
