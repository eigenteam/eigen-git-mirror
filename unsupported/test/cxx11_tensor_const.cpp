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




static void test_simple_assign()
{
  Tensor<int, 3> random(2,3,7);
  random.setRandom();

  TensorMap<Tensor<const int, 3> > constant(random.data(), 2, 3, 7);
  Tensor<int, 3> result(2,3,7);
  result = constant;

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        VERIFY_IS_EQUAL((result(i,j,k)), random(i,j,k));
      }
    }
  }
}

void test_cxx11_tensor_const()
{
  CALL_SUBTEST(test_simple_assign());
}
