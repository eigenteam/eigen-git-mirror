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
  typedef Tensor<float, 1>::DimensionPair DimPair;
  array<DimPair, 1> contract_along{{DimPair(1, 0)}};
  Tensor<float, 2> tensor3(2,13);
  tensor3 = tensor1.reshape(newDims1).contract(tensor2.reshape(newDims2), contract_along);

  Map<MatrixXf> res(tensor3.data(), 2, 13);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 13; ++j) {
      VERIFY_IS_APPROX(res(i,j), m3(i,j));
    }
  }
}

void test_cxx11_tensor_morphing()
{
  CALL_SUBTEST(test_simple_reshape());
  CALL_SUBTEST(test_reshape_in_expr());
}
