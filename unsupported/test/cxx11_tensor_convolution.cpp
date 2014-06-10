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
using Eigen::DefaultDevice;

static void test_evals()
{
  Tensor<float, 2> input(3, 3);
  Tensor<float, 1> kernel(2);

  input.setRandom();
  kernel.setRandom();

  Tensor<float, 2> result(2,3);
  result.setZero();
  Eigen::array<Tensor<float, 2>::Index, 1> dims3({0});

  typedef TensorEvaluator<decltype(input.convolve(kernel, dims3)), DefaultDevice> Evaluator;
  Evaluator eval(input.convolve(kernel, dims3), DefaultDevice());
  eval.evalTo(result.data());
  EIGEN_STATIC_ASSERT(Evaluator::NumDims==2ul, YOU_MADE_A_PROGRAMMING_MISTAKE);
  VERIFY_IS_EQUAL(eval.dimensions()[0], 2);
  VERIFY_IS_EQUAL(eval.dimensions()[1], 3);

  VERIFY_IS_APPROX(result(0,0), input(0,0)*kernel(0) + input(1,0)*kernel(1));  // index 0
  VERIFY_IS_APPROX(result(0,1), input(0,1)*kernel(0) + input(1,1)*kernel(1));  // index 2
  VERIFY_IS_APPROX(result(0,2), input(0,2)*kernel(0) + input(1,2)*kernel(1));  // index 4
  VERIFY_IS_APPROX(result(1,0), input(1,0)*kernel(0) + input(2,0)*kernel(1));  // index 1
  VERIFY_IS_APPROX(result(1,1), input(1,1)*kernel(0) + input(2,1)*kernel(1));  // index 3
  VERIFY_IS_APPROX(result(1,2), input(1,2)*kernel(0) + input(2,2)*kernel(1));  // index 5
}


static void test_expr()
{
  Tensor<float, 2> input(3, 3);
  Tensor<float, 2> kernel(2, 2);
  input.setRandom();
  kernel.setRandom();

  Tensor<float, 2> result(2,2);
  Eigen::array<ptrdiff_t, 2> dims({0, 1});
  result = input.convolve(kernel, dims);

  VERIFY_IS_APPROX(result(0,0), input(0,0)*kernel(0,0) + input(0,1)*kernel(0,1) +
                                input(1,0)*kernel(1,0) + input(1,1)*kernel(1,1));
  VERIFY_IS_APPROX(result(0,1), input(0,1)*kernel(0,0) + input(0,2)*kernel(0,1) +
                                input(1,1)*kernel(1,0) + input(1,2)*kernel(1,1));
  VERIFY_IS_APPROX(result(1,0), input(1,0)*kernel(0,0) + input(1,1)*kernel(0,1) +
                                input(2,0)*kernel(1,0) + input(2,1)*kernel(1,1));
  VERIFY_IS_APPROX(result(1,1), input(1,1)*kernel(0,0) + input(1,2)*kernel(0,1) +
                                input(2,1)*kernel(1,0) + input(2,2)*kernel(1,1));
}


void test_cxx11_tensor_convolution()
{
  CALL_SUBTEST(test_evals());
  CALL_SUBTEST(test_expr());
}
