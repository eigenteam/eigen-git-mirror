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


static void test_static_index_list()
{
  Tensor<float, 4> tensor(2,3,5,7);
  tensor.setRandom();

  constexpr auto reduction_axis = make_index_list(0, 1, 2);
  VERIFY_IS_EQUAL(internal::array_get<0>(reduction_axis), 0);
  VERIFY_IS_EQUAL(internal::array_get<1>(reduction_axis), 1);
  VERIFY_IS_EQUAL(internal::array_get<2>(reduction_axis), 2);
  VERIFY_IS_EQUAL(static_cast<DenseIndex>(reduction_axis[0]), 0);
  VERIFY_IS_EQUAL(static_cast<DenseIndex>(reduction_axis[1]), 1);
  VERIFY_IS_EQUAL(static_cast<DenseIndex>(reduction_axis[2]), 2);

  EIGEN_STATIC_ASSERT((internal::array_get<0>(reduction_axis) == 0), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::array_get<1>(reduction_axis) == 1), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::array_get<2>(reduction_axis) == 2), YOU_MADE_A_PROGRAMMING_MISTAKE);

  Tensor<float, 1> result = tensor.sum(reduction_axis);
  for (int i = 0; i < result.size(); ++i) {
    float expected = 0.0f;
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 5; ++l) {
          expected += tensor(j,k,l,i);
        }
      }
    }
    VERIFY_IS_APPROX(result(i), expected);
  }
}


static void test_dynamic_index_list()
{
  Tensor<float, 4> tensor(2,3,5,7);
  tensor.setRandom();

  int dim1 = 2;
  int dim2 = 1;
  int dim3 = 0;

  auto reduction_axis = make_index_list(dim1, dim2, dim3);

  VERIFY_IS_EQUAL(internal::array_get<0>(reduction_axis), 2);
  VERIFY_IS_EQUAL(internal::array_get<1>(reduction_axis), 1);
  VERIFY_IS_EQUAL(internal::array_get<2>(reduction_axis), 0);
  VERIFY_IS_EQUAL(static_cast<DenseIndex>(reduction_axis[0]), 2);
  VERIFY_IS_EQUAL(static_cast<DenseIndex>(reduction_axis[1]), 1);
  VERIFY_IS_EQUAL(static_cast<DenseIndex>(reduction_axis[2]), 0);

  Tensor<float, 1> result = tensor.sum(reduction_axis);
  for (int i = 0; i < result.size(); ++i) {
    float expected = 0.0f;
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 5; ++l) {
          expected += tensor(j,k,l,i);
        }
      }
    }
    VERIFY_IS_APPROX(result(i), expected);
  }
}

static void test_mixed_index_list()
{
  Tensor<float, 4> tensor(2,3,5,7);
  tensor.setRandom();

  int dim2 = 1;
  int dim4 = 3;

  auto reduction_axis = make_index_list(0, dim2, 2, dim4);

  VERIFY_IS_EQUAL(internal::array_get<0>(reduction_axis), 0);
  VERIFY_IS_EQUAL(internal::array_get<1>(reduction_axis), 1);
  VERIFY_IS_EQUAL(internal::array_get<2>(reduction_axis), 2);
  VERIFY_IS_EQUAL(internal::array_get<3>(reduction_axis), 3);
  VERIFY_IS_EQUAL(static_cast<DenseIndex>(reduction_axis[0]), 0);
  VERIFY_IS_EQUAL(static_cast<DenseIndex>(reduction_axis[1]), 1);
  VERIFY_IS_EQUAL(static_cast<DenseIndex>(reduction_axis[2]), 2);
  VERIFY_IS_EQUAL(static_cast<DenseIndex>(reduction_axis[3]), 3);

  typedef IndexList<type2index<0>, int, type2index<2>, int> ReductionIndices;
  ReductionIndices reduction_indices;
  reduction_indices.set(1, 1);
  reduction_indices.set(3, 3);
  EIGEN_STATIC_ASSERT((internal::array_get<0>(reduction_indices) == 0), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::array_get<2>(reduction_indices) == 2), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::index_known_statically<ReductionIndices>()(0) == true), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::index_known_statically<ReductionIndices>()(2) == true), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::index_statically_eq<ReductionIndices>()(0, 0) == true), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::index_statically_eq<ReductionIndices>()(2, 2) == true), YOU_MADE_A_PROGRAMMING_MISTAKE);


  Tensor<float, 1> result1 = tensor.sum(reduction_axis);
  Tensor<float, 1> result2 = tensor.sum(reduction_indices);

  float expected = 0.0f;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 7; ++l) {
          expected += tensor(i,j,k,l);
        }
      }
    }
  }
  VERIFY_IS_APPROX(result1(0), expected);
  VERIFY_IS_APPROX(result2(0), expected);
}


void test_cxx11_tensor_index_list()
{
  CALL_SUBTEST(test_static_index_list());
  CALL_SUBTEST(test_dynamic_index_list());
  CALL_SUBTEST(test_mixed_index_list());
}
