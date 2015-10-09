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

#include <Eigen/Dense>
#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;


template <int DataLayout>
static void test_customIndex() {
  
  Tensor<float, 4, DataLayout> tensor(2, 3, 5, 7);
  
  using NormalIndex = DSizes<ptrdiff_t, 4>;
  using CustomIndex = Matrix<unsigned int , 4, 1>;
  
  tensor.setRandom();

  CustomIndex coeffC(1,2,4,1);
  NormalIndex coeff(1,2,4,1);
  
  VERIFY_IS_EQUAL(tensor.coeff( coeffC ), tensor.coeff( coeff ));
  VERIFY_IS_EQUAL(tensor.coeffRef( coeffC ), tensor.coeffRef( coeff ));
  
}


void test_cxx11_tensor_customIndex() {
  CALL_SUBTEST(test_customIndex<ColMajor>());
  CALL_SUBTEST(test_customIndex<RowMajor>());
}
