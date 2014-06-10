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

using Eigen::DefaultDevice;
using Eigen::Tensor;

typedef Tensor<float, 1>::DimensionPair DimPair;


static void test_evals()
{
  Tensor<float, 2> mat1(2, 3);
  Tensor<float, 2> mat2(2, 3);
  Tensor<float, 2> mat3(3, 2);

  mat1.setRandom();
  mat2.setRandom();
  mat3.setRandom();

  Tensor<float, 2> mat4(3,3);
  mat4.setZero();
  Eigen::array<DimPair, 1> dims3({{DimPair(0, 0)}});
  typedef TensorEvaluator<decltype(mat1.contract(mat2, dims3)), DefaultDevice> Evaluator;
  Evaluator eval(mat1.contract(mat2, dims3), DefaultDevice());
  eval.evalTo(mat4.data());
  EIGEN_STATIC_ASSERT(Evaluator::NumDims==2ul, YOU_MADE_A_PROGRAMMING_MISTAKE);
  VERIFY_IS_EQUAL(eval.dimensions()[0], 3);
  VERIFY_IS_EQUAL(eval.dimensions()[1], 3);

  VERIFY_IS_APPROX(mat4(0,0), mat1(0,0)*mat2(0,0) + mat1(1,0)*mat2(1,0));
  VERIFY_IS_APPROX(mat4(0,1), mat1(0,0)*mat2(0,1) + mat1(1,0)*mat2(1,1));
  VERIFY_IS_APPROX(mat4(0,2), mat1(0,0)*mat2(0,2) + mat1(1,0)*mat2(1,2));
  VERIFY_IS_APPROX(mat4(1,0), mat1(0,1)*mat2(0,0) + mat1(1,1)*mat2(1,0));
  VERIFY_IS_APPROX(mat4(1,1), mat1(0,1)*mat2(0,1) + mat1(1,1)*mat2(1,1));
  VERIFY_IS_APPROX(mat4(1,2), mat1(0,1)*mat2(0,2) + mat1(1,1)*mat2(1,2));
  VERIFY_IS_APPROX(mat4(2,0), mat1(0,2)*mat2(0,0) + mat1(1,2)*mat2(1,0));
  VERIFY_IS_APPROX(mat4(2,1), mat1(0,2)*mat2(0,1) + mat1(1,2)*mat2(1,1));
  VERIFY_IS_APPROX(mat4(2,2), mat1(0,2)*mat2(0,2) + mat1(1,2)*mat2(1,2));

  Tensor<float, 2> mat5(2,2);
  mat5.setZero();
  Eigen::array<DimPair, 1> dims4({{DimPair(1, 1)}});
  typedef TensorEvaluator<decltype(mat1.contract(mat2, dims4)), DefaultDevice> Evaluator2;
  Evaluator2 eval2(mat1.contract(mat2, dims4), DefaultDevice());
  eval2.evalTo(mat5.data());
  EIGEN_STATIC_ASSERT(Evaluator2::NumDims==2ul, YOU_MADE_A_PROGRAMMING_MISTAKE);
  VERIFY_IS_EQUAL(eval2.dimensions()[0], 2);
  VERIFY_IS_EQUAL(eval2.dimensions()[1], 2);

  VERIFY_IS_APPROX(mat5(0,0), mat1(0,0)*mat2(0,0) + mat1(0,1)*mat2(0,1) + mat1(0,2)*mat2(0,2));
  VERIFY_IS_APPROX(mat5(0,1), mat1(0,0)*mat2(1,0) + mat1(0,1)*mat2(1,1) + mat1(0,2)*mat2(1,2));
  VERIFY_IS_APPROX(mat5(1,0), mat1(1,0)*mat2(0,0) + mat1(1,1)*mat2(0,1) + mat1(1,2)*mat2(0,2));
  VERIFY_IS_APPROX(mat5(1,1), mat1(1,0)*mat2(1,0) + mat1(1,1)*mat2(1,1) + mat1(1,2)*mat2(1,2));

  Tensor<float, 2> mat6(2,2);
  mat6.setZero();
  Eigen::array<DimPair, 1> dims6({{DimPair(1, 0)}});
  typedef TensorEvaluator<decltype(mat1.contract(mat3, dims6)), DefaultDevice> Evaluator3;
  Evaluator3 eval3(mat1.contract(mat3, dims6), DefaultDevice());
  eval3.evalTo(mat6.data());
  EIGEN_STATIC_ASSERT(Evaluator3::NumDims==2ul, YOU_MADE_A_PROGRAMMING_MISTAKE);
  VERIFY_IS_EQUAL(eval3.dimensions()[0], 2);
  VERIFY_IS_EQUAL(eval3.dimensions()[1], 2);

  VERIFY_IS_APPROX(mat6(0,0), mat1(0,0)*mat3(0,0) + mat1(0,1)*mat3(1,0) + mat1(0,2)*mat3(2,0));
  VERIFY_IS_APPROX(mat6(0,1), mat1(0,0)*mat3(0,1) + mat1(0,1)*mat3(1,1) + mat1(0,2)*mat3(2,1));
  VERIFY_IS_APPROX(mat6(1,0), mat1(1,0)*mat3(0,0) + mat1(1,1)*mat3(1,0) + mat1(1,2)*mat3(2,0));
  VERIFY_IS_APPROX(mat6(1,1), mat1(1,0)*mat3(0,1) + mat1(1,1)*mat3(1,1) + mat1(1,2)*mat3(2,1));
}


static void test_scalar()
{
  Tensor<float, 1> vec1({6});
  Tensor<float, 1> vec2({6});

  vec1.setRandom();
  vec2.setRandom();

  Tensor<float, 1> scalar(1);
  scalar.setZero();
  Eigen::array<DimPair, 1> dims({{DimPair(0, 0)}});
  typedef TensorEvaluator<decltype(vec1.contract(vec2, dims)), DefaultDevice> Evaluator;
  Evaluator eval(vec1.contract(vec2, dims), DefaultDevice());
  eval.evalTo(scalar.data());
  EIGEN_STATIC_ASSERT(Evaluator::NumDims==1ul, YOU_MADE_A_PROGRAMMING_MISTAKE);

  float expected = 0.0f;
  for (int i = 0; i < 6; ++i) {
    expected += vec1(i) * vec2(i);
  }
  VERIFY_IS_APPROX(scalar(0), expected);
}


static void test_multidims()
{
  Tensor<float, 3> mat1(2, 2, 2);
  Tensor<float, 4> mat2(2, 2, 2, 2);

  mat1.setRandom();
  mat2.setRandom();

  Tensor<float, 3> mat3(2, 2, 2);
  mat3.setZero();
  Eigen::array<DimPair, 2> dims({{DimPair(1, 2), DimPair(2, 3)}});
  typedef TensorEvaluator<decltype(mat1.contract(mat2, dims)), DefaultDevice> Evaluator;
  Evaluator eval(mat1.contract(mat2, dims), DefaultDevice());
  eval.evalTo(mat3.data());
  EIGEN_STATIC_ASSERT(Evaluator::NumDims==3ul, YOU_MADE_A_PROGRAMMING_MISTAKE);
  VERIFY_IS_EQUAL(eval.dimensions()[0], 2);
  VERIFY_IS_EQUAL(eval.dimensions()[1], 2);
  VERIFY_IS_EQUAL(eval.dimensions()[2], 2);

  VERIFY_IS_APPROX(mat3(0,0,0), mat1(0,0,0)*mat2(0,0,0,0) + mat1(0,1,0)*mat2(0,0,1,0) +
                                mat1(0,0,1)*mat2(0,0,0,1) + mat1(0,1,1)*mat2(0,0,1,1));
  VERIFY_IS_APPROX(mat3(0,0,1), mat1(0,0,0)*mat2(0,1,0,0) + mat1(0,1,0)*mat2(0,1,1,0) +
                                mat1(0,0,1)*mat2(0,1,0,1) + mat1(0,1,1)*mat2(0,1,1,1));
  VERIFY_IS_APPROX(mat3(0,1,0), mat1(0,0,0)*mat2(1,0,0,0) + mat1(0,1,0)*mat2(1,0,1,0) +
                                mat1(0,0,1)*mat2(1,0,0,1) + mat1(0,1,1)*mat2(1,0,1,1));
  VERIFY_IS_APPROX(mat3(0,1,1), mat1(0,0,0)*mat2(1,1,0,0) + mat1(0,1,0)*mat2(1,1,1,0) +
                                mat1(0,0,1)*mat2(1,1,0,1) + mat1(0,1,1)*mat2(1,1,1,1));
  VERIFY_IS_APPROX(mat3(1,0,0), mat1(1,0,0)*mat2(0,0,0,0) + mat1(1,1,0)*mat2(0,0,1,0) +
                                mat1(1,0,1)*mat2(0,0,0,1) + mat1(1,1,1)*mat2(0,0,1,1));
  VERIFY_IS_APPROX(mat3(1,0,1), mat1(1,0,0)*mat2(0,1,0,0) + mat1(1,1,0)*mat2(0,1,1,0) +
                                mat1(1,0,1)*mat2(0,1,0,1) + mat1(1,1,1)*mat2(0,1,1,1));
  VERIFY_IS_APPROX(mat3(1,1,0), mat1(1,0,0)*mat2(1,0,0,0) + mat1(1,1,0)*mat2(1,0,1,0) +
                                mat1(1,0,1)*mat2(1,0,0,1) + mat1(1,1,1)*mat2(1,0,1,1));
  VERIFY_IS_APPROX(mat3(1,1,1), mat1(1,0,0)*mat2(1,1,0,0) + mat1(1,1,0)*mat2(1,1,1,0) +
                                mat1(1,0,1)*mat2(1,1,0,1) + mat1(1,1,1)*mat2(1,1,1,1));
}


static void test_expr()
{
  Tensor<float, 2> mat1(2, 3);
  Tensor<float, 2> mat2(3, 2);
  mat1.setRandom();
  mat2.setRandom();

  Tensor<float, 2> mat3(2,2);

  Eigen::array<DimPair, 1> dims({{DimPair(1, 0)}});
  mat3 = mat1.contract(mat2, dims);

  VERIFY_IS_APPROX(mat3(0,0), mat1(0,0)*mat2(0,0) + mat1(0,1)*mat2(1,0) + mat1(0,2)*mat2(2,0));
  VERIFY_IS_APPROX(mat3(0,1), mat1(0,0)*mat2(0,1) + mat1(0,1)*mat2(1,1) + mat1(0,2)*mat2(2,1));
  VERIFY_IS_APPROX(mat3(1,0), mat1(1,0)*mat2(0,0) + mat1(1,1)*mat2(1,0) + mat1(1,2)*mat2(2,0));
  VERIFY_IS_APPROX(mat3(1,1), mat1(1,0)*mat2(0,1) + mat1(1,1)*mat2(1,1) + mat1(1,2)*mat2(2,1));
}


void test_cxx11_tensor_contraction()
{
  CALL_SUBTEST(test_evals());
  CALL_SUBTEST(test_scalar());
  CALL_SUBTEST(test_multidims());
  CALL_SUBTEST(test_expr());
}
