// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016
// Mehdi Goli    Codeplay Software Ltd.
// Ralph Potter  Codeplay Software Ltd.
// Luke Iwanski  Codeplay Software Ltd.
// Contact: <eigen@codeplay.com>
// Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX
#define EIGEN_TEST_FUNC cxx11_tensor_morphing_sycl
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_SYCL


#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::array;
using Eigen::SyclDevice;
using Eigen::Tensor;
using Eigen::TensorMap;


static void test_simple_slice(const Eigen::SyclDevice &sycl_device)
{
  int sizeDim1 = 2;
  int sizeDim2 = 3;
  int sizeDim3 = 5;
  int sizeDim4 = 7;
  int sizeDim5 = 11;
  array<int, 5> tensorRange = {{sizeDim1, sizeDim2, sizeDim3, sizeDim4, sizeDim5}};
  Tensor<float, 5> tensor(tensorRange);
  tensor.setRandom();
  array<int, 5> slice1_range ={{1, 1, 1, 1, 1}};
  Tensor<float, 5> slice1(slice1_range);

  float* gpu_data1  = static_cast<float*>(sycl_device.allocate(tensor.size()*sizeof(float)));
  float* gpu_data2  = static_cast<float*>(sycl_device.allocate(slice1.size()*sizeof(float)));
  TensorMap<Tensor<float, 5>> gpu1(gpu_data1, tensorRange);
  TensorMap<Tensor<float, 5>> gpu2(gpu_data2, slice1_range);
  Eigen::DSizes<ptrdiff_t, 5> indices(1,2,3,4,5);
  Eigen::DSizes<ptrdiff_t, 5> sizes(1,1,1,1,1);
  sycl_device.memcpyHostToDevice(gpu_data1, tensor.data(),(tensor.size())*sizeof(float));
  gpu2.device(sycl_device)=gpu1.slice(indices, sizes);
  sycl_device.memcpyDeviceToHost(slice1.data(), gpu_data2,(slice1.size())*sizeof(float));
  VERIFY_IS_EQUAL(slice1(0,0,0,0,0), tensor(1,2,3,4,5));


  array<int, 5> slice2_range ={{1,1,2,2,3}};
  Tensor<float, 5> slice2(slice2_range);
  float* gpu_data3  = static_cast<float*>(sycl_device.allocate(slice2.size()*sizeof(float)));
  TensorMap<Tensor<float, 5>> gpu3(gpu_data3, slice2_range);
  Eigen::DSizes<ptrdiff_t, 5> indices2(1,1,3,4,5);
  Eigen::DSizes<ptrdiff_t, 5> sizes2(1,1,2,2,3);
  gpu3.device(sycl_device)=gpu1.slice(indices2, sizes2);
  sycl_device.memcpyDeviceToHost(slice2.data(), gpu_data3,(slice2.size())*sizeof(float));
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 3; ++k) {
        VERIFY_IS_EQUAL(slice2(0,0,i,j,k), tensor(1,1,3+i,4+j,5+k));
      }
    }
  }
  sycl_device.deallocate(gpu_data1);
  sycl_device.deallocate(gpu_data2);
  sycl_device.deallocate(gpu_data3);
}

void test_cxx11_tensor_morphing_sycl()
{
  /// Currentlly it only works on cpu. Adding GPU cause LLVM ERROR in cunstructing OpenCL Kernel at runtime.
  cl::sycl::cpu_selector s;
  Eigen::SyclDevice sycl_device(s);
  CALL_SUBTEST(test_simple_slice(sycl_device));

}
