// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_FUNC cxx11_tensor_complex
#define EIGEN_USE_GPU

#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
#include <cuda_fp16.h>
#endif
#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::Tensor;

void test_cuda_nullary() {
  Tensor<std::complex<float>, 1, 0, int> in1(2);
  Tensor<std::complex<float>, 1, 0, int> in2(2);
  in1.setRandom();
  in2.setRandom();

  std::size_t float_bytes = in1.size() * sizeof(float);
  std::size_t complex_bytes = in1.size() * sizeof(std::complex<float>);

  std::complex<float>* d_in1;
  std::complex<float>* d_in2;
  float* d_out2;
  cudaMalloc((void**)(&d_in1), complex_bytes);
  cudaMalloc((void**)(&d_in2), complex_bytes);
  cudaMalloc((void**)(&d_out2), float_bytes);
  cudaMemcpy(d_in1, in1.data(), complex_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_in2, in2.data(), complex_bytes, cudaMemcpyHostToDevice);

  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<std::complex<float>, 1, 0, int>, Eigen::Aligned> gpu_in1(
      d_in1, 2);
  Eigen::TensorMap<Eigen::Tensor<std::complex<float>, 1, 0, int>, Eigen::Aligned> gpu_in2(
      d_in2, 2);
  Eigen::TensorMap<Eigen::Tensor<float, 1, 0, int>, Eigen::Aligned> gpu_out2(
      d_out2, 2);

  gpu_in1.device(gpu_device) = gpu_in1.constant(std::complex<float>(3.14f, 2.7f));
  gpu_out2.device(gpu_device) = gpu_in2.abs();

  Tensor<std::complex<float>, 1, 0, int> new1(2);
  Tensor<float, 1, 0, int> new2(2);

  assert(cudaMemcpyAsync(new1.data(), d_in1, complex_bytes, cudaMemcpyDeviceToHost,
                         gpu_device.stream()) == cudaSuccess);
  assert(cudaMemcpyAsync(new2.data(), d_out2, float_bytes, cudaMemcpyDeviceToHost,
                         gpu_device.stream()) == cudaSuccess);

  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);

  for (int i = 0; i < 2; ++i) {
    VERIFY_IS_APPROX(new1(i), std::complex<float>(3.14f, 2.7f));
    VERIFY_IS_APPROX(new2(i), std::abs(in2(i)));
  }

  cudaFree(d_in1);
  cudaFree(d_in2);
  cudaFree(d_out2);
}



void test_cxx11_tensor_complex()
{
  CALL_SUBTEST(test_cuda_nullary());
}
