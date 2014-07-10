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

static void test_simple_reshape()
{
  Tensor<float, 5> tensor1(2,3,1,7,1);
  tensor1.setRandom();

  Tensor<float, 3> tensor2(2,3,7);
  Tensor<float, 2> tensor3(6,7);
  Tensor<float, 2> tensor4(2,21);

  Tensor<float, 3>::Dimensions dim1{{2,3,7}};
  tensor2 = tensor1.reshape(dim1);
  Tensor<float, 2>::Dimensions dim2{{6,7}};
  tensor3 = tensor1.reshape(dim2);
  Tensor<float, 2>::Dimensions dim3{{2,21}};
  tensor4 = tensor1.reshape(dim1).reshape(dim3);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        VERIFY_IS_EQUAL(tensor1(i,j,0,k,0), tensor2(i,j,k));
        VERIFY_IS_EQUAL(tensor1(i,j,0,k,0), tensor3(i+2*j,k));
        VERIFY_IS_EQUAL(tensor1(i,j,0,k,0), tensor4(i,j+3*k));
      }
    }
  }
}


static void test_reshape_in_expr() {
  MatrixXf m1(2,3*5*7*11);
  MatrixXf m2(3*5*7*11,13);
  m1.setRandom();
  m2.setRandom();
  MatrixXf m3 = m1 * m2;

  TensorMap<Tensor<float, 5>> tensor1(m1.data(), 2,3,5,7,11);
  TensorMap<Tensor<float, 5>> tensor2(m2.data(), 3,5,7,11,13);
  Tensor<float, 2>::Dimensions newDims1{{2,3*5*7*11}};
  Tensor<float, 2>::Dimensions newDims2{{3*5*7*11,13}};
  array<Tensor<float, 1>::DimensionPair, 1> contract_along{{1, 0}};
  Tensor<float, 2> tensor3(2,13);
  tensor3 = tensor1.reshape(newDims1).contract(tensor2.reshape(newDims2), contract_along);

  Map<MatrixXf> res(tensor3.data(), 2, 13);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 13; ++j) {
      VERIFY_IS_APPROX(res(i,j), m3(i,j));
    }
  }
}


static void test_reshape_as_lvalue()
{
  Tensor<float, 3> tensor(2,3,7);
  tensor.setRandom();

  Tensor<float, 2> tensor2d(6,7);
  Tensor<float, 3>::Dimensions dim{{2,3,7}};
  tensor2d.reshape(dim) = tensor;

  Tensor<float, 5> tensor5d(2,3,1,7,1);
  tensor5d.reshape(dim).device(Eigen::DefaultDevice()) = tensor;

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        VERIFY_IS_EQUAL(tensor2d(i+2*j,k), tensor(i,j,k));
        VERIFY_IS_EQUAL(tensor5d(i,j,0,k,0), tensor(i,j,k));
      }
    }
  }
}


static void test_simple_slice()
{
  Tensor<float, 5> tensor(2,3,5,7,11);
  tensor.setRandom();

  Tensor<float, 5> slice1(1,1,1,1,1);
  Eigen::DSizes<ptrdiff_t, 5> indices(Eigen::array<ptrdiff_t, 5>(1,2,3,4,5));
  Eigen::DSizes<ptrdiff_t, 5> sizes(Eigen::array<ptrdiff_t, 5>(1,1,1,1,1));
  slice1 = tensor.slice(indices, sizes);
  VERIFY_IS_EQUAL(slice1(0,0,0,0,0), tensor(1,2,3,4,5));

  Tensor<float, 5> slice2(1,1,2,2,3);
  Eigen::DSizes<ptrdiff_t, 5> indices2(Eigen::array<ptrdiff_t, 5>(1,1,3,4,5));
  Eigen::DSizes<ptrdiff_t, 5> sizes2(Eigen::array<ptrdiff_t, 5>(1,1,2,2,3));
  slice2 = tensor.slice(indices2, sizes2);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 3; ++k) {
        VERIFY_IS_EQUAL(slice2(0,0,i,j,k), tensor(1,1,3+i,4+j,5+k));
      }
    }
  }
}


static void test_slice_in_expr() {
  MatrixXf m1(7,7);
  MatrixXf m2(3,3);
  m1.setRandom();
  m2.setRandom();

  MatrixXf m3 = m1.block(1, 2, 3, 3) * m2.block(0, 2, 3, 1);

  TensorMap<Tensor<float, 2>> tensor1(m1.data(), 7, 7);
  TensorMap<Tensor<float, 2>> tensor2(m2.data(), 3, 3);
  Tensor<float, 2> tensor3(3,1);
  array<Tensor<float, 1>::DimensionPair, 1> contract_along{{1, 0}};

  Eigen::DSizes<ptrdiff_t, 2> indices1(Eigen::array<ptrdiff_t, 2>(1,2));
  Eigen::DSizes<ptrdiff_t, 2> sizes1(Eigen::array<ptrdiff_t, 2>(3,3));
  Eigen::DSizes<ptrdiff_t, 2> indices2(Eigen::array<ptrdiff_t, 2>(0,2));
  Eigen::DSizes<ptrdiff_t, 2> sizes2(Eigen::array<ptrdiff_t, 2>(3,1));
  tensor3 = tensor1.slice(indices1, sizes1).contract(tensor2.slice(indices2, sizes2), contract_along);

  Map<MatrixXf> res(tensor3.data(), 3, 1);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 1; ++j) {
      VERIFY_IS_APPROX(res(i,j), m3(i,j));
    }
  }
}


static void test_slice_as_lvalue()
{
  Tensor<float, 3> tensor1(2,2,7);
  tensor1.setRandom();
  Tensor<float, 3> tensor2(2,2,7);
  tensor2.setRandom();
  Tensor<float, 3> tensor3(4,3,5);
  tensor3.setRandom();
  Tensor<float, 3> tensor4(4,3,2);
  tensor4.setRandom();

  Tensor<float, 3> result(4,5,7);
  Eigen::DSizes<ptrdiff_t, 3> sizes12(Eigen::array<ptrdiff_t, 3>(2,2,7));
  Eigen::DSizes<ptrdiff_t, 3> first_slice(Eigen::array<ptrdiff_t, 3>(0,0,0));
  result.slice(first_slice, sizes12) = tensor1;
  Eigen::DSizes<ptrdiff_t, 3> second_slice(Eigen::array<ptrdiff_t, 3>(2,0,0));
  result.slice(second_slice, sizes12).device(Eigen::DefaultDevice()) = tensor2;

  Eigen::DSizes<ptrdiff_t, 3> sizes3(Eigen::array<ptrdiff_t, 3>(4,3,5));
  Eigen::DSizes<ptrdiff_t, 3> third_slice(Eigen::array<ptrdiff_t, 3>(0,2,0));
  result.slice(third_slice, sizes3) = tensor3;

  Eigen::DSizes<ptrdiff_t, 3> sizes4(Eigen::array<ptrdiff_t, 3>(4,3,2));
  Eigen::DSizes<ptrdiff_t, 3> fourth_slice(Eigen::array<ptrdiff_t, 3>(0,2,5));
  result.slice(fourth_slice, sizes4) = tensor4;

  for (int j = 0; j < 2; ++j) {
    for (int k = 0; k < 7; ++k) {
      for (int i = 0; i < 2; ++i) {
        VERIFY_IS_EQUAL(result(i,j,k), tensor1(i,j,k));
        VERIFY_IS_EQUAL(result(i+2,j,k), tensor2(i,j,k));
      }
    }
  }
  for (int i = 0; i < 4; ++i) {
    for (int j = 2; j < 5; ++j) {
      for (int k = 0; k < 5; ++k) {
        VERIFY_IS_EQUAL(result(i,j,k), tensor3(i,j-2,k));
      }
      for (int k = 5; k < 7; ++k) {
        VERIFY_IS_EQUAL(result(i,j,k), tensor4(i,j-2,k-5));
      }
    }
  }
}


void test_cxx11_tensor_morphing()
{
  CALL_SUBTEST(test_simple_reshape());
  CALL_SUBTEST(test_reshape_in_expr());
  CALL_SUBTEST(test_reshape_as_lvalue());

  CALL_SUBTEST(test_simple_slice());
  CALL_SUBTEST(test_slice_in_expr());
  CALL_SUBTEST(test_slice_as_lvalue());
}
