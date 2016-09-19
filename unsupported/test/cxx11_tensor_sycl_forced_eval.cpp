// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016
// Mehdi Goli    Codeplay Software Ltd.
// Ralph Potter  Codeplay Software Ltd.
// Luke Iwanski  Codeplay Software Ltd.
// Contact: <eigen@codeplay.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX
#define EIGEN_TEST_FUNC cxx11_tensor_sycl_forced_eval
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_SYCL

#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::Tensor;

void test_sycl_gpu() {
	cl::sycl::gpu_selector s;
  cl::sycl::queue q(s, [=](cl::sycl::exception_list l) {
    for (const auto& e : l) {
      try {
        std::rethrow_exception(e);
      } catch (cl::sycl::exception e) {
        std::cout << e.what() << std::endl;
      }
    }
  });
	SyclDevice sycl_device(q);

  int sizeDim1 = 100;
  int sizeDim2 = 200;
  int sizeDim3 = 200;
  Eigen::array<int, 3> tensorRange = {{sizeDim1, sizeDim2, sizeDim3}};
  Eigen::Tensor<float, 3> in1(tensorRange);
  Eigen::Tensor<float, 3> in2(tensorRange);
  Eigen::Tensor<float, 3> out(tensorRange);

  in1 = in1.random() + in1.constant(10.0f);
  in2 = in2.random() + in2.constant(10.0f);

	// creating TensorMap from tensor
  Eigen::TensorMap<Eigen::Tensor<float, 3>> gpu_in1(in1.data(), tensorRange);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> gpu_in2(in2.data(), tensorRange);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> gpu_out(out.data(), tensorRange);

  /// c=(a+b)*b
	gpu_out.device(sycl_device) =(gpu_in1 + gpu_in2).eval() * gpu_in2;
	sycl_device.deallocate(out.data());
  for (int i = 0; i < sizeDim1; ++i) {
    for (int j = 0; j < sizeDim2; ++j) {
      for (int k = 0; k < sizeDim3; ++k) {
        VERIFY_IS_APPROX(out(i, j, k),
                         (in1(i, j, k) + in2(i, j, k)) * in2(i, j, k));
      }
    }
  }
	printf("(a+b)*b Test Passed\n");
}

void test_cxx11_tensor_sycl_forced_eval() { CALL_SUBTEST(test_sycl_gpu()); }
