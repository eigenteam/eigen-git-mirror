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
#define EIGEN_TEST_FUNC cxx11_tensor_sycl
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

void test_sycl_cpu() {
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
  int sizeDim2 = 100;
  int sizeDim3 = 100;
  array<int, 3> tensorRange = {{sizeDim1, sizeDim2, sizeDim3}};
  TestTensor in1(tensorRange);
  TestTensor in2(tensorRange);
  TestTensor in3(tensorRange);
  TestTensor out(tensorRange);
  in1 = in1.random();
  in2 = in2.random();
  in3 = in3.random();
	TestTensorMap gpu_in1(in1.data(), tensorRange);
	TestTensorMap gpu_in2(in2.data(), tensorRange);
	TestTensorMap gpu_in3(in3.data(), tensorRange);
	TestTensorMap gpu_out(out.data(), tensorRange);

	/// a=1.2f
	gpu_in1.device(sycl_device) = gpu_in1.constant(1.2f);
	sycl_device.deallocate(in1.data());
  for (int i = 0; i < sizeDim1; ++i) {
    for (int j = 0; j < sizeDim2; ++j) {
      for (int k = 0; k < sizeDim3; ++k) {
        VERIFY_IS_APPROX(in1(i,j,k), 1.2f);
      }
    }
  }
	printf("a=1.2f Test passed\n");

	/// a=b*1.2f
	gpu_out.device(sycl_device) = gpu_in1 * 1.2f;
	sycl_device.deallocate(out.data());
  for (int i = 0; i < sizeDim1; ++i) {
    for (int j = 0; j < sizeDim2; ++j) {
      for (int k = 0; k < sizeDim3; ++k) {
        VERIFY_IS_APPROX(out(i,j,k),
                         in1(i,j,k) * 1.2f);
      }
    }
  }
	printf("a=b*1.2f Test Passed\n");

	/// c=a*b
	gpu_out.device(sycl_device) = gpu_in1 * gpu_in2;
	sycl_device.deallocate(out.data());
  for (int i = 0; i < sizeDim1; ++i) {
    for (int j = 0; j < sizeDim2; ++j) {
      for (int k = 0; k < sizeDim3; ++k) {
        VERIFY_IS_APPROX(out(i,j,k),
                         in1(i,j,k) *
                             in2(i,j,k));
      }
    }
  }
	printf("c=a*b Test Passed\n");

	/// c=a+b
	gpu_out.device(sycl_device) = gpu_in1 + gpu_in2;
	sycl_device.deallocate(out.data());
  for (int i = 0; i < sizeDim1; ++i) {
    for (int j = 0; j < sizeDim2; ++j) {
      for (int k = 0; k < sizeDim3; ++k) {
        VERIFY_IS_APPROX(out(i,j,k),
                         in1(i,j,k) +
                             in2(i,j,k));
      }
    }
  }
	printf("c=a+b Test Passed\n");

	/// c=a*a
	gpu_out.device(sycl_device) = gpu_in1 * gpu_in1;
	sycl_device.deallocate(out.data());
  for (int i = 0; i < sizeDim1; ++i) {
    for (int j = 0; j < sizeDim2; ++j) {
      for (int k = 0; k < sizeDim3; ++k) {
        VERIFY_IS_APPROX(out(i,j,k),
                         in1(i,j,k) *
                             in1(i,j,k));
      }
    }
  }

	printf("c= a*a Test Passed\n");

	//a*3.14f + b*2.7f
	gpu_out.device(sycl_device) =  gpu_in1 * gpu_in1.constant(3.14f) + gpu_in2 * gpu_in2.constant(2.7f);
	sycl_device.deallocate(out.data());
  for (int i = 0; i < sizeDim1; ++i) {
    for (int j = 0; j < sizeDim2; ++j) {
      for (int k = 0; k < sizeDim3; ++k) {
        VERIFY_IS_APPROX(out(i,j,k),
                         in1(i,j,k) * 3.14f
                       + in2(i,j,k) * 2.7f);
      }
    }
  }
	printf("a*3.14f + b*2.7f Test Passed\n");

	///d= (a>0.5? b:c)
	gpu_out.device(sycl_device) =(gpu_in1 > gpu_in1.constant(0.5f)).select(gpu_in2, gpu_in3);
	sycl_device.deallocate(out.data());
	for (int i = 0; i < sizeDim1; ++i) {
	  for (int j = 0; j < sizeDim2; ++j) {
	    for (int k = 0; k < sizeDim3; ++k) {
	      VERIFY_IS_APPROX(out(i, j, k), (in1(i, j, k) > 0.5f)
	                                              ? in2(i, j, k)
	                                              : in3(i, j, k));
	    }
	  }
	}
	printf("d= (a>0.5? b:c) Test Passed\n");

}
void test_cxx11_tensor_sycl() {
  CALL_SUBTEST(test_sycl_cpu());
}
