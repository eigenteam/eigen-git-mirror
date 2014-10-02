// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/Core>
#include <Eigen/CXX11/Tensor>

using Eigen::MatrixXf;
using Eigen::Tensor;

static void test_simple()
{
  MatrixXf m1(3,3);
  MatrixXf m2(3,3);
  m1.setRandom();
  m2.setRandom();

  TensorMap<Tensor<float, 2>> mat1(m1.data(), 3,3);
  TensorMap<Tensor<float, 2>> mat2(m2.data(), 3,3);

  Tensor<float, 2> mat3(3,3);
  mat3 = mat1;

  typedef Tensor<float, 1>::DimensionPair DimPair;
  Eigen::array<DimPair, 1> dims({{DimPair(1, 0)}});

  mat3 = mat3.contract(mat2, dims).eval();

  VERIFY_IS_APPROX(mat3(0, 0), (m1*m2).eval()(0,0));
  VERIFY_IS_APPROX(mat3(0, 1), (m1*m2).eval()(0,1));
  VERIFY_IS_APPROX(mat3(0, 2), (m1*m2).eval()(0,2));
  VERIFY_IS_APPROX(mat3(1, 0), (m1*m2).eval()(1,0));
  VERIFY_IS_APPROX(mat3(1, 1), (m1*m2).eval()(1,1));
  VERIFY_IS_APPROX(mat3(1, 2), (m1*m2).eval()(1,2));
  VERIFY_IS_APPROX(mat3(2, 0), (m1*m2).eval()(2,0));
  VERIFY_IS_APPROX(mat3(2, 1), (m1*m2).eval()(2,1));
  VERIFY_IS_APPROX(mat3(2, 2), (m1*m2).eval()(2,2));
}


void test_cxx11_tensor_forced_eval()
{
  CALL_SUBTEST(test_simple());
}
