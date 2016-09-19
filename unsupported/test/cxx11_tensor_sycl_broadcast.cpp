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
#define EIGEN_TEST_FUNC cxx11_tensor_sycl_broadcast
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_SYCL

#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::array;
using Eigen::SyclDevice;
using Eigen::Tensor;
using Eigen::TensorMap;

// Types used in tests:
using TestTensor = Tensor<float, 3>;
using TestTensorMap = TensorMap<Tensor<float, 3>>;
static void test_sycl_broadcast(){

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
	// BROADCAST test:
	array<int, 4> in_range   = {{2, 3, 5, 7}};
	array<int, in_range.size()> broadcasts = {{2, 3, 1, 4}};
	array<int, in_range.size()> out_range;  // = in_range * broadcasts
	for (size_t i = 0; i < out_range.size(); ++i)
		out_range[i] = in_range[i] * broadcasts[i];

	Tensor<float, in_range.size()>  input(in_range);
	Tensor<float, out_range.size()> output(out_range);

	for (int i = 0; i < input.size(); ++i)
		input(i) = static_cast<float>(i);

	TensorMap<decltype(input)>  gpu_in(input.data(), in_range);
	TensorMap<decltype(output)> gpu_out(output.data(), out_range);
		gpu_out.device(sycl_device) = gpu_in.broadcast(broadcasts);
		sycl_device.deallocate(output.data());

	for (size_t i = 0; i < in_range.size(); ++i)
		VERIFY_IS_EQUAL(output.dimension(i), out_range[i]);

	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 9; ++j) {
			for (int k = 0; k < 5; ++k) {
				for (int l = 0; l < 28; ++l) {
					VERIFY_IS_APPROX(input(i%2,j%3,k%5,l%7), output(i,j,k,l));
				}
			}
		}
	}
	printf("Broadcast Test Passed\n");
}

void test_cxx11_tensor_sycl_broadcast() {
  CALL_SUBTEST(test_sycl_broadcast());
}
