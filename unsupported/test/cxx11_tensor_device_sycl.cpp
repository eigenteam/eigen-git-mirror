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
#define EIGEN_TEST_FUNC cxx11_tensor_device_sycl
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_SYCL

#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include<stdint.h>

void test_device_memory(const Eigen::SyclDevice &sycl_device) {
  std::cout << "Running on: "
	    << sycl_device.m_queue.get_device(). template get_info<cl::sycl::info::device::name>()
	    << std::endl;
  int sizeDim1 = 100;

  array<int, 1> tensorRange = {{sizeDim1}};
  Tensor<int, 1> in(tensorRange);
  Tensor<int, 1> in1(tensorRange);
  memset(in1.data(), 1,in1.size()*sizeof(int));
  int* gpu_in_data  = static_cast<int*>(sycl_device.allocate(in.size()*sizeof(int)));
  sycl_device.memset(gpu_in_data, 1, in.size()*sizeof(int) );
  sycl_device.memcpyDeviceToHost(in.data(), gpu_in_data, in.size()*sizeof(int) );
  for (int i=0; i<in.size(); i++) {
    VERIFY_IS_APPROX(in(i), in1(i));
  }
  sycl_device.deallocate(gpu_in_data);
}


void test_device_exceptions(const Eigen::SyclDevice &sycl_device) {
  VERIFY(sycl_device.ok());
  array<int, 1> tensorDims = {{100}};
  int* gpu_data = static_cast<int*>(sycl_device.allocate(100*sizeof(int)));
  TensorMap<Tensor<int, 1>> in(gpu_data, tensorDims);
  TensorMap<Tensor<int, 1>> out(gpu_data, tensorDims);
  out.device(sycl_device) = in / in.constant(0);
  VERIFY(!sycl_device.ok());
  sycl_device.deallocate(gpu_data);
}


void test_cxx11_tensor_device_sycl() {
  cl::sycl::gpu_selector s;
  Eigen::SyclDevice sycl_device(s);
  CALL_SUBTEST(test_device_memory(sycl_device));
  // This deadlocks
  //CALL_SUBTEST(test_device_exceptions(sycl_device));
}
