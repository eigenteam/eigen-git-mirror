// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX
#define EIGEN_TEST_FUNC cxx11_tensor_reduction_cuda
#define EIGEN_USE_GPU

#include <cuda_fp16.h>
#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>


template<typename Type, int DataLayout>
static void test_full_reductions() {

  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  const int num_rows = internal::random<int>(1024, 5*1024);
  const int num_cols = internal::random<int>(1024, 5*1024);

  Tensor<Type, 2, DataLayout> in(num_rows, num_cols);
  in.setRandom();

  Tensor<Type, 0, DataLayout> full_redux;
  full_redux = in.sum();

  std::size_t in_bytes = in.size() * sizeof(Type);
  std::size_t out_bytes = full_redux.size() * sizeof(Type);
  Type* gpu_in_ptr = static_cast<Type*>(gpu_device.allocate(in_bytes));
  Type* gpu_out_ptr = static_cast<Type*>(gpu_device.allocate(out_bytes));
  gpu_device.memcpyHostToDevice(gpu_in_ptr, in.data(), in_bytes);

  TensorMap<Tensor<Type, 2, DataLayout> > in_gpu(gpu_in_ptr, num_rows, num_cols);
  TensorMap<Tensor<Type, 0, DataLayout> > out_gpu(gpu_out_ptr);

  out_gpu.device(gpu_device) = in_gpu.sum();

  Tensor<Type, 0, DataLayout> full_redux_gpu;
  gpu_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_ptr, out_bytes);
  gpu_device.synchronize();

  // Check that the CPU and GPU reductions return the same result.
  VERIFY_IS_APPROX(full_redux(), full_redux_gpu());

  gpu_device.deallocate(gpu_in_ptr);
  gpu_device.deallocate(gpu_out_ptr);
}

void test_cxx11_tensor_reduction_cuda() {
  CALL_SUBTEST_1((test_full_reductions<float, ColMajor>()));
  CALL_SUBTEST_1((test_full_reductions<double, ColMajor>()));
  CALL_SUBTEST_2((test_full_reductions<float, RowMajor>()));
  CALL_SUBTEST_2((test_full_reductions<double, RowMajor>()));
}
