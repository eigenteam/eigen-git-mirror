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


static void test_holes() {
  Tensor<float, 4> t1(2, 5, 7, 3);
  Tensor<float, 5> t2(2, 7, 11, 13, 3);
  t1.setRandom();
  t2.setRandom();

  Eigen::array<DimPair, 2> dims({{DimPair(0, 0), DimPair(3, 4)}});
  Tensor<float, 5> result = t1.contract(t2, dims);
  VERIFY_IS_EQUAL(result.dimension(0), 5);
  VERIFY_IS_EQUAL(result.dimension(1), 7);
  VERIFY_IS_EQUAL(result.dimension(2), 7);
  VERIFY_IS_EQUAL(result.dimension(3), 11);
  VERIFY_IS_EQUAL(result.dimension(4), 13);

  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 5; ++l) {
          for (int m = 0; m < 5; ++m) {
            VERIFY_IS_APPROX(result(i, j, k, l, m),
                             t1(0, i, j, 0) * t2(0, k, l, m, 0) +
                             t1(1, i, j, 0) * t2(1, k, l, m, 0) +
                             t1(0, i, j, 1) * t2(0, k, l, m, 1) +
                             t1(1, i, j, 1) * t2(1, k, l, m, 1) +
                             t1(0, i, j, 2) * t2(0, k, l, m, 2) +
                             t1(1, i, j, 2) * t2(1, k, l, m, 2));
          }
        }
      }
    }
  }
}


static void test_full_redux()
{
  Tensor<float, 2> t1(2, 2);
  Tensor<float, 3> t2(2, 2, 2);
  t1.setRandom();
  t2.setRandom();

  Eigen::array<DimPair, 2> dims({{DimPair(0, 0), DimPair(1, 1)}});
  Tensor<float, 1> result = t1.contract(t2, dims);
  VERIFY_IS_EQUAL(result.dimension(0), 2);
  VERIFY_IS_APPROX(result(0), t1(0, 0) * t2(0, 0, 0) +  t1(1, 0) * t2(1, 0, 0)
                            + t1(0, 1) * t2(0, 1, 0) +  t1(1, 1) * t2(1, 1, 0));
  VERIFY_IS_APPROX(result(1), t1(0, 0) * t2(0, 0, 1) +  t1(1, 0) * t2(1, 0, 1)
                            + t1(0, 1) * t2(0, 1, 1) +  t1(1, 1) * t2(1, 1, 1));

  dims[0] = DimPair(1, 0);
  dims[1] = DimPair(2, 1);
  result = t2.contract(t1, dims);
  VERIFY_IS_EQUAL(result.dimension(0), 2);
  VERIFY_IS_APPROX(result(0), t1(0, 0) * t2(0, 0, 0) +  t1(1, 0) * t2(0, 1, 0)
                            + t1(0, 1) * t2(0, 0, 1) +  t1(1, 1) * t2(0, 1, 1));
  VERIFY_IS_APPROX(result(1), t1(0, 0) * t2(1, 0, 0) +  t1(1, 0) * t2(1, 1, 0)
                            + t1(0, 1) * t2(1, 0, 1) +  t1(1, 1) * t2(1, 1, 1));
}


static void test_contraction_of_contraction()
{
  Tensor<float, 2> t1(2, 2);
  Tensor<float, 2> t2(2, 2);
  Tensor<float, 2> t3(2, 2);
  Tensor<float, 2> t4(2, 2);
  t1.setRandom();
  t2.setRandom();
  t3.setRandom();
  t4.setRandom();

  Eigen::array<DimPair, 1> dims({{DimPair(1, 0)}});
  auto contract1 = t1.contract(t2, dims);
  auto diff = t3 - contract1;
  auto contract2 = t1.contract(t4, dims);
  Tensor<float, 2> result = contract2.contract(diff, dims);
  VERIFY_IS_EQUAL(result.dimension(0), 2);
  VERIFY_IS_EQUAL(result.dimension(1), 2);

  Eigen::Map<MatrixXf> m1(t1.data(), 2, 2);
  Eigen::Map<MatrixXf> m2(t2.data(), 2, 2);
  Eigen::Map<MatrixXf> m3(t3.data(), 2, 2);
  Eigen::Map<MatrixXf> m4(t4.data(), 2, 2);
  Eigen::MatrixXf expected = (m1 * m4) * (m3 - m1 * m2);
  VERIFY_IS_APPROX(result(0, 0), expected(0, 0));
  VERIFY_IS_APPROX(result(0, 1), expected(0, 1));
  VERIFY_IS_APPROX(result(1, 0), expected(1, 0));
  VERIFY_IS_APPROX(result(1, 1), expected(1, 1));
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


static void test_out_of_order_contraction()
{
  Tensor<float, 3> mat1(2, 2, 2);
  Tensor<float, 3> mat2(2, 2, 2);

  mat1.setRandom();
  mat2.setRandom();

  Tensor<float, 2> mat3(2, 2);

  Eigen::array<DimPair, 2> dims({{DimPair(2, 0), DimPair(0, 2)}});
  mat3 = mat1.contract(mat2, dims);

  VERIFY_IS_APPROX(mat3(0, 0),
                   mat1(0,0,0)*mat2(0,0,0) + mat1(1,0,0)*mat2(0,0,1) +
                   mat1(0,0,1)*mat2(1,0,0) + mat1(1,0,1)*mat2(1,0,1));
  VERIFY_IS_APPROX(mat3(1, 0),
                   mat1(0,1,0)*mat2(0,0,0) + mat1(1,1,0)*mat2(0,0,1) +
                   mat1(0,1,1)*mat2(1,0,0) + mat1(1,1,1)*mat2(1,0,1));
  VERIFY_IS_APPROX(mat3(0, 1),
                   mat1(0,0,0)*mat2(0,1,0) + mat1(1,0,0)*mat2(0,1,1) +
                   mat1(0,0,1)*mat2(1,1,0) + mat1(1,0,1)*mat2(1,1,1));
  VERIFY_IS_APPROX(mat3(1, 1),
                   mat1(0,1,0)*mat2(0,1,0) + mat1(1,1,0)*mat2(0,1,1) +
                   mat1(0,1,1)*mat2(1,1,0) + mat1(1,1,1)*mat2(1,1,1));

  Eigen::array<DimPair, 2> dims2({{DimPair(0, 2), DimPair(2, 0)}});
  mat3 = mat1.contract(mat2, dims2);

  VERIFY_IS_APPROX(mat3(0, 0),
                   mat1(0,0,0)*mat2(0,0,0) + mat1(1,0,0)*mat2(0,0,1) +
                   mat1(0,0,1)*mat2(1,0,0) + mat1(1,0,1)*mat2(1,0,1));
  VERIFY_IS_APPROX(mat3(1, 0),
                   mat1(0,1,0)*mat2(0,0,0) + mat1(1,1,0)*mat2(0,0,1) +
                   mat1(0,1,1)*mat2(1,0,0) + mat1(1,1,1)*mat2(1,0,1));
  VERIFY_IS_APPROX(mat3(0, 1),
                   mat1(0,0,0)*mat2(0,1,0) + mat1(1,0,0)*mat2(0,1,1) +
                   mat1(0,0,1)*mat2(1,1,0) + mat1(1,0,1)*mat2(1,1,1));
  VERIFY_IS_APPROX(mat3(1, 1),
                   mat1(0,1,0)*mat2(0,1,0) + mat1(1,1,0)*mat2(0,1,1) +
                   mat1(0,1,1)*mat2(1,1,0) + mat1(1,1,1)*mat2(1,1,1));

}


static void test_consistency()
{
  // this does something like testing (A*B)^T = (B^T * A^T)

  Tensor<float, 3> mat1(4, 3, 5);
  Tensor<float, 5> mat2(3, 2, 1, 5, 4);
  mat1.setRandom();
  mat2.setRandom();

  Tensor<float, 4> mat3(5, 2, 1, 5);
  Tensor<float, 4> mat4(2, 1, 5, 5);

  // contract on dimensions of size 4 and 3
  Eigen::array<DimPair, 2> dims1({{DimPair(0, 4), DimPair(1, 0)}});
  Eigen::array<DimPair, 2> dims2({{DimPair(4, 0), DimPair(0, 1)}});

  mat3 = mat1.contract(mat2, dims1);
  mat4 = mat2.contract(mat1, dims2);

  // check that these are equal except for ordering of dimensions
  for (size_t i = 0; i < 5; i++) {
    for (size_t j = 0; j < 10; j++) {
      VERIFY_IS_APPROX(mat3.data()[i + 5 * j], mat4.data()[j + 10 * i]);
    }
  }
}


static void test_large_contraction()
{
  Tensor<float, 4> t_left(30, 50, 8, 31);
  Tensor<float, 5> t_right(8, 31, 7, 20, 10);
  Tensor<float, 5> t_result(30, 50, 7, 20, 10);

  t_left.setRandom();
  t_right.setRandom();

  typedef Map<MatrixXf> MapXf;
  MapXf m_left(t_left.data(), 1500, 248);
  MapXf m_right(t_right.data(), 248, 1400);
  MatrixXf m_result(1500, 1400);

  // this contraction should be equivalent to a single matrix multiplication
  Eigen::array<DimPair, 2> dims({{DimPair(2, 0), DimPair(3, 1)}});

  // compute results by separate methods
  t_result = t_left.contract(t_right, dims);
  m_result = m_left * m_right;

  for (size_t i = 0; i < t_result.dimensions().TotalSize(); i++) {
    VERIFY(&t_result.data()[i] != &m_result.data()[i]);
    VERIFY_IS_APPROX(t_result.data()[i], m_result.data()[i]);
  }
}


void test_cxx11_tensor_contraction()
{
  CALL_SUBTEST(test_evals());
  CALL_SUBTEST(test_scalar());
  CALL_SUBTEST(test_multidims());
  CALL_SUBTEST(test_holes());
  CALL_SUBTEST(test_full_redux());
  CALL_SUBTEST(test_contraction_of_contraction());
  CALL_SUBTEST(test_expr());
  CALL_SUBTEST(test_out_of_order_contraction());
  CALL_SUBTEST(test_consistency());
  CALL_SUBTEST(test_large_contraction());
}
