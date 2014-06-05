// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_USE_THREADS


#include "main.h"
#include <Eigen/CXX11/Tensor>
#include "thread/threadpool.h"

using Eigen::Tensor;

void test_cxx11_tensor_thread_pool()
{
  Eigen::Tensor<float, 3> in1(Eigen::array<ptrdiff_t, 3>(2,3,7));
  Eigen::Tensor<float, 3> in2(Eigen::array<ptrdiff_t, 3>(2,3,7));
  Eigen::Tensor<float, 3> out(Eigen::array<ptrdiff_t, 3>(2,3,7));

  in1.setRandom();
  in2.setRandom();

  ThreadPool thread_pool(2);
  thread_pool.StartWorkers();
  Eigen::ThreadPoolDevice thread_pool_device(&thread_pool, 3);
  out.device(thread_pool_device) = in1 + in2 * 3.14f;

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        VERIFY_IS_APPROX(out(Eigen::array<ptrdiff_t, 3>(i,j,k)), in1(Eigen::array<ptrdiff_t, 3>(i,j,k)) + in2(Eigen::array<ptrdiff_t, 3>(i,j,k)) * 3.14f);
      }
    }
  }
}
