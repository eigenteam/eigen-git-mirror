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
using Eigen::array;

static void test_simple_cast()
{
  Tensor<float, 2> ftensor(20,30);
  ftensor.setRandom();
  Tensor<char, 2> chartensor(20,30);
  chartensor.setRandom();
  Tensor<std::complex<float>, 2> cplextensor(20,30);
  cplextensor.setRandom();

  chartensor = ftensor.cast<char>();
  cplextensor = ftensor.cast<std::complex<float>>();

  for (int i = 0; i < 20; ++i) {
    for (int j = 0; j < 30; ++j) {
      VERIFY_IS_EQUAL(chartensor(i,j), static_cast<char>(ftensor(i,j)));
      VERIFY_IS_EQUAL(cplextensor(i,j), static_cast<std::complex<float>>(ftensor(i,j)));
    }
  }
}


void test_cxx11_tensor_casts()
{
   CALL_SUBTEST(test_simple_cast());
}
