// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX
#define EIGEN_TEST_FUNC cxx11_tensor_of_float16_cuda
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU


#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::Tensor;

#ifdef EIGEN_HAS_CUDA_FP16

void test_cuda_conversion() {
  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);
  int num_elem = 101;
  
  float* d_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
  half* d_half = (half*)gpu_device.allocate(num_elem * sizeof(half));
  float* d_conv = (float*)gpu_device.allocate(num_elem * sizeof(float));

  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float(
      d_float, num_elem);
  Eigen::TensorMap<Eigen::Tensor<half, 1>, Eigen::Aligned> gpu_half(
      d_half, num_elem);
  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_conv(
      d_conv, num_elem);

  gpu_float.device(gpu_device) = gpu_float.random();
  gpu_half.device(gpu_device) = gpu_float.cast<half>();
  gpu_conv.device(gpu_device) = gpu_half.cast<float>();

  Tensor<float, 1> initial(num_elem);
  Tensor<float, 1> final(num_elem);
  gpu_device.memcpyDeviceToHost(initial.data(), d_float, num_elem*sizeof(float));
  gpu_device.memcpyDeviceToHost(final.data(), d_conv, num_elem*sizeof(float));

  for (int i = 0; i < num_elem; ++i) {
    VERIFY_IS_APPROX(initial(i), final(i));
  }

  gpu_device.deallocate(d_float);
  gpu_device.deallocate(d_half);
  gpu_device.deallocate(d_conv);
}

void test_cuda_elementwise() {
  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);
  int num_elem = 101;

  float* d_float1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
  float* d_float2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
  float* d_res_half = (float*)gpu_device.allocate(num_elem * sizeof(float));
  float* d_res_float = (float*)gpu_device.allocate(num_elem * sizeof(float));

  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float1(
      d_float1, num_elem);
  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float2(
      d_float2, num_elem);
  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_half(
      d_res_half, num_elem);
  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_float(
      d_res_float, num_elem);

  gpu_float1.device(gpu_device) = gpu_float1.random();
  gpu_float2.device(gpu_device) = gpu_float2.random();
  gpu_res_float.device(gpu_device) = (gpu_float1 + gpu_float2) * gpu_float1;
  gpu_res_half.device(gpu_device) = ((gpu_float1.cast<half>() + gpu_float2.cast<half>()) * gpu_float1.cast<half>()).cast<float>();

  Tensor<float, 1> half_prec(num_elem);
  Tensor<float, 1> full_prec(num_elem);
  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, num_elem*sizeof(float));
  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, num_elem*sizeof(float));

  for (int i = 0; i < num_elem; ++i) {
    VERIFY_IS_APPROX(full_prec(i), half_prec(i));
  }

  gpu_device.deallocate(d_float1);
  gpu_device.deallocate(d_float2);
  gpu_device.deallocate(d_res_half);
  gpu_device.deallocate(d_res_float);
}

#endif


void test_cxx11_tensor_of_float16_cuda()
{
#ifdef EIGEN_HAS_CUDA_FP16
  CALL_SUBTEST_1(test_cuda_conversion());
  CALL_SUBTEST_1(test_cuda_element_wise());
#endif
}
