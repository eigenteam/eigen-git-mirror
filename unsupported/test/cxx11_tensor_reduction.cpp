// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <limits>
#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;

static void test_simple_reductions()
{
  Tensor<float, 4> tensor(2,3,5,7);
  tensor.setRandom();
  array<ptrdiff_t, 2> reduction_axis;
  reduction_axis[0] = 1;
  reduction_axis[1] = 3;

  Tensor<float, 2> result = tensor.sum(reduction_axis);
  VERIFY_IS_EQUAL(result.dimension(0), 2);
  VERIFY_IS_EQUAL(result.dimension(1), 5);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 5; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 7; ++l) {
          sum += tensor(i, k, j, l);
        }
      }
      VERIFY_IS_APPROX(result(i, j), sum);
    }
  }

  reduction_axis[0] = 0;
  reduction_axis[1] = 2;
  result = tensor.maximum(reduction_axis);
  VERIFY_IS_EQUAL(result.dimension(0), 3);
  VERIFY_IS_EQUAL(result.dimension(1), 7);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 7; ++j) {
      float max_val = std::numeric_limits<float>::lowest();
      for (int k = 0; k < 2; ++k) {
        for (int l = 0; l < 5; ++l) {
          max_val = (std::max)(max_val, tensor(k, i, l, j));
        }
      }
      VERIFY_IS_APPROX(result(i, j), max_val);
    }
  }

  reduction_axis[0] = 0;
  reduction_axis[1] = 1;
  result = tensor.minimum(reduction_axis);
  VERIFY_IS_EQUAL(result.dimension(0), 5);
  VERIFY_IS_EQUAL(result.dimension(1), 7);
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 7; ++j) {
      float min_val = (std::numeric_limits<float>::max)();
      for (int k = 0; k < 2; ++k) {
        for (int l = 0; l < 3; ++l) {
          min_val = (std::min)(min_val, tensor(k,  l, i, j));
        }
      }
      VERIFY_IS_APPROX(result(i, j), min_val);
    }
  }
}


static void test_full_reductions()
{
  Tensor<float, 2> tensor(2,3);
  tensor.setRandom();
  array<ptrdiff_t, 2> reduction_axis;
  reduction_axis[0] = 0;
  reduction_axis[1] = 1;

  Tensor<float, 1> result = tensor.sum(reduction_axis);
  VERIFY_IS_EQUAL(result.dimension(0), 1);

  float sum = 0.0f;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      sum += tensor(i, j);
    }
  }
  VERIFY_IS_APPROX(result(0), sum);

  result = tensor.square().sum(reduction_axis).sqrt();
  VERIFY_IS_EQUAL(result.dimension(0), 1);

  sum = 0.0f;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      sum += tensor(i, j) * tensor(i, j);
    }
  }
  VERIFY_IS_APPROX(result(0), sqrtf(sum));
}


struct UserReducer {
  UserReducer(float offset) : offset_(offset), sum_(0.0f) {}
  void reduce(const float val) {
    sum_ += val * val;
  }
  float finalize() const {
    return 1.0f / (sum_ + offset_);
  }

 private:
  float offset_;
  float sum_;
};

static void test_user_defined_reductions()
{
  Tensor<float, 2> tensor(5,7);
  tensor.setRandom();
  array<ptrdiff_t, 1> reduction_axis;
  reduction_axis[0] = 1;

  UserReducer reducer(10.0f);
  Tensor<float, 1> result = tensor.reduce(reduction_axis, reducer);
  VERIFY_IS_EQUAL(result.dimension(0), 5);
  for (int i = 0; i < 5; ++i) {
    float expected = 10.0f;
    for (int j = 0; j < 7; ++j) {
      expected += tensor(i, j) * tensor(i, j);
    }
    expected = 1.0f / expected;
    VERIFY_IS_APPROX(result(i), expected);
  }
}


static void test_tensor_maps()
{
  int inputs[2*3*5*7];
  TensorMap<Tensor<int, 4> > tensor_map(inputs, 2,3,5,7);
  TensorMap<Tensor<const int, 4> > tensor_map_const(inputs, 2,3,5,7);
  const TensorMap<Tensor<const int, 4> > tensor_map_const_const(inputs, 2,3,5,7);

  tensor_map.setRandom();
  array<ptrdiff_t, 2> reduction_axis;
  reduction_axis[0] = 1;
  reduction_axis[1] = 3;

  Tensor<int, 2> result = tensor_map.sum(reduction_axis);
  Tensor<int, 2> result2 = tensor_map_const.sum(reduction_axis);
  Tensor<int, 2> result3 = tensor_map_const_const.sum(reduction_axis);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 5; ++j) {
      int sum = 0;
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 7; ++l) {
          sum += tensor_map(i, k, j, l);
        }
      }
      VERIFY_IS_EQUAL(result(i, j), sum);
      VERIFY_IS_EQUAL(result2(i, j), sum);
      VERIFY_IS_EQUAL(result3(i, j), sum);
    }
  }
}


void test_cxx11_tensor_reduction()
{
   CALL_SUBTEST(test_simple_reductions());
   CALL_SUBTEST(test_full_reductions());
   CALL_SUBTEST(test_user_defined_reductions());
   CALL_SUBTEST(test_tensor_maps());
}
