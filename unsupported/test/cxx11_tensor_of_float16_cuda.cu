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
  Eigen::half* d_half = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
  float* d_conv = (float*)gpu_device.allocate(num_elem * sizeof(float));

  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float(
      d_float, num_elem);
  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_half(
      d_half, num_elem);
  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_conv(
      d_conv, num_elem);

  gpu_float.device(gpu_device) = gpu_float.random();
  gpu_half.device(gpu_device) = gpu_float.cast<Eigen::half>();
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


void test_cuda_unary() {
  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);
  int num_elem = 101;

  float* d_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
  float* d_res_half = (float*)gpu_device.allocate(num_elem * sizeof(float));
  float* d_res_float = (float*)gpu_device.allocate(num_elem * sizeof(float));

  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float(
      d_float, num_elem);
  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_half(
      d_res_half, num_elem);
  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_float(
      d_res_float, num_elem);

  gpu_float.device(gpu_device) = gpu_float.random() - gpu_float.constant(0.5f);
  gpu_res_float.device(gpu_device) = gpu_float.abs();
  gpu_res_half.device(gpu_device) = gpu_float.cast<Eigen::half>().abs().cast<float>();

  Tensor<float, 1> half_prec(num_elem);
  Tensor<float, 1> full_prec(num_elem);
  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, num_elem*sizeof(float));
  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, num_elem*sizeof(float));
  gpu_device.synchronize();

  for (int i = 0; i < num_elem; ++i) {
    std::cout << "Checking unary " << i << std::endl;
    VERIFY_IS_APPROX(full_prec(i), half_prec(i));
  }

  gpu_device.deallocate(d_float);
  gpu_device.deallocate(d_res_half);
  gpu_device.deallocate(d_res_float);
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
  gpu_res_half.device(gpu_device) = ((gpu_float1.cast<Eigen::half>() + gpu_float2.cast<Eigen::half>()) * gpu_float1.cast<Eigen::half>()).cast<float>();

  Tensor<float, 1> half_prec(num_elem);
  Tensor<float, 1> full_prec(num_elem);
  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, num_elem*sizeof(float));
  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, num_elem*sizeof(float));
  gpu_device.synchronize();

  for (int i = 0; i < num_elem; ++i) {
    std::cout << "Checking elemwise " << i << std::endl;
    VERIFY_IS_APPROX(full_prec(i), half_prec(i));
  }

  gpu_device.deallocate(d_float1);
  gpu_device.deallocate(d_float2);
  gpu_device.deallocate(d_res_half);
  gpu_device.deallocate(d_res_float);
}


void test_cuda_contractions() {
  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);
  int rows = 23;
  int cols = 23;
  int num_elem = rows*cols;

  float* d_float1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
  float* d_float2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
  float* d_res_half = (float*)gpu_device.allocate(num_elem * sizeof(float));
  float* d_res_float = (float*)gpu_device.allocate(num_elem * sizeof(float));

  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float1(
      d_float1, rows, cols);
  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float2(
      d_float2, rows, cols);
  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_res_half(
      d_res_half, rows, cols);
  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_res_float(
      d_res_float, rows, cols);

  gpu_float1.device(gpu_device) = gpu_float1.random() - gpu_float1.constant(0.5f);
  gpu_float2.device(gpu_device) = gpu_float2.random() - gpu_float1.constant(0.5f);

  typedef Tensor<float, 2>::DimensionPair DimPair;
  Eigen::array<DimPair, 1> dims(DimPair(1, 0));
  gpu_res_float.device(gpu_device) = gpu_float1.contract(gpu_float2, dims);
  gpu_res_half.device(gpu_device) = gpu_float1.cast<Eigen::half>().contract(gpu_float2.cast<Eigen::half>(), dims).cast<float>();

  Tensor<float, 2> half_prec(rows, cols);
  Tensor<float, 2> full_prec(rows, cols);
  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, num_elem*sizeof(float));
  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, num_elem*sizeof(float));
  gpu_device.synchronize();

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      std::cout << "Checking contract " << i << " " << j << std::endl;
      VERIFY_IS_APPROX(full_prec(i, j), half_prec(i, j));
    }
  }

  gpu_device.deallocate(d_float1);
  gpu_device.deallocate(d_float2);
  gpu_device.deallocate(d_res_half);
  gpu_device.deallocate(d_res_float);
}


void test_cuda_reductions() {
  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);
  int size = 13;
  int num_elem = size*size;

  float* d_float1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
  float* d_float2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
  float* d_res_half = (float*)gpu_device.allocate(size * sizeof(float));
  float* d_res_float = (float*)gpu_device.allocate(size * sizeof(float));

  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float1(
      d_float1, size, size);
  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float2(
      d_float2, size, size);
  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_half(
      d_res_half, size);
  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_float(
      d_res_float, size);

  gpu_float1.device(gpu_device) = gpu_float1.random();
  gpu_float2.device(gpu_device) = gpu_float2.random();

  Eigen::array<int, 1> redux_dim = {{0}};
  gpu_res_float.device(gpu_device) = gpu_float1.sum(redux_dim);
  gpu_res_half.device(gpu_device) = gpu_float1.cast<Eigen::half>().sum(redux_dim).cast<float>();

  Tensor<float, 1> half_prec(size);
  Tensor<float, 1> full_prec(size);
  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, size*sizeof(float));
  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, size*sizeof(float));
  gpu_device.synchronize();

  for (int i = 0; i < size; ++i) {
    std::cout << "Checking redux " << i << std::endl;
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
  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice device(&stream);
  if (device.majorDeviceVersion() > 5 ||
      (device.majorDeviceVersion() == 5 && device.minorDeviceVersion() >= 3)) {
    std::cout << "Running test on device with capability " << device.majorDeviceVersion() << "." << device.minorDeviceVersion() << std::endl;

    CALL_SUBTEST_1(test_cuda_conversion());
    CALL_SUBTEST_1(test_cuda_unary());
    CALL_SUBTEST_1(test_cuda_elementwise());
    CALL_SUBTEST_2(test_cuda_contractions());
    CALL_SUBTEST_3(test_cuda_reductions());
  }
  else {
   std::cout << "Half floats require compute capability of at least 5.3. This device only supports " << device.majorDeviceVersion() << "." << device.minorDeviceVersion() << ". Skipping the test" << std::endl;
  }
#else
  std::cout << "Half floats are not supported by this version of cuda: skipping the test" << std::endl;
#endif
}
