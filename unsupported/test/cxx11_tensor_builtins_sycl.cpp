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
#define EIGEN_TEST_FUNC cxx11_tensor_builtins_sycl
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_SYCL

#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::array;
using Eigen::SyclDevice;
using Eigen::Tensor;
using Eigen::TensorMap;

namespace std {
template <typename T> T rsqrt(T x) { return 1 / std::sqrt(x); }
template <typename T> T square(T x) { return x * x; }
template <typename T> T cube(T x) { return x * x * x; }
template <typename T> T inverse(T x) { return 1 / x; }
}

#define TEST_UNARY_BUILTINS_FOR_SCALAR(FUNC, SCALAR)                           \
  {                                                                            \
    Tensor<SCALAR, 3> in(tensorRange);                                         \
    Tensor<SCALAR, 3> out(tensorRange);                                        \
    in = in.random() + static_cast<SCALAR>(0.01);                              \
    SCALAR *gpu_data = static_cast<SCALAR *>(                                  \
        sycl_device.allocate(in.size() * sizeof(SCALAR)));                     \
    SCALAR *gpu_data_out = static_cast<SCALAR *>(                              \
        sycl_device.allocate(out.size() * sizeof(SCALAR)));                    \
    TensorMap<Tensor<SCALAR, 3>> gpu(gpu_data, tensorRange);                   \
    TensorMap<Tensor<SCALAR, 3>> gpu_out(gpu_data_out, tensorRange);           \
    sycl_device.memcpyHostToDevice(gpu_data, in.data(),                        \
                                   (in.size()) * sizeof(SCALAR));              \
    gpu_out.device(sycl_device) = gpu.FUNC();                                  \
    sycl_device.memcpyDeviceToHost(out.data(), gpu_data_out,                   \
                                   (out.size()) * sizeof(SCALAR));             \
    for (int i = 0; i < in.size(); ++i) {                                      \
      VERIFY_IS_APPROX(out(i), std::FUNC(in(i)));                              \
    }                                                                          \
    sycl_device.deallocate(gpu_data);                                          \
    sycl_device.deallocate(gpu_data_out);                                      \
  }

#define TEST_UNARY_BUILTINS(SCALAR)                                            \
  TEST_UNARY_BUILTINS_FOR_SCALAR(abs, SCALAR)                                  \
  TEST_UNARY_BUILTINS_FOR_SCALAR(sqrt, SCALAR)                                 \
  TEST_UNARY_BUILTINS_FOR_SCALAR(rsqrt, SCALAR)                                \
  TEST_UNARY_BUILTINS_FOR_SCALAR(square, SCALAR)                               \
  TEST_UNARY_BUILTINS_FOR_SCALAR(cube, SCALAR)                                 \
  TEST_UNARY_BUILTINS_FOR_SCALAR(inverse, SCALAR)                              \
  TEST_UNARY_BUILTINS_FOR_SCALAR(tanh, SCALAR)                                 \
  TEST_UNARY_BUILTINS_FOR_SCALAR(exp, SCALAR)                                  \
  TEST_UNARY_BUILTINS_FOR_SCALAR(log, SCALAR)                                  \
  TEST_UNARY_BUILTINS_FOR_SCALAR(abs, SCALAR)                                  \
  TEST_UNARY_BUILTINS_FOR_SCALAR(ceil, SCALAR)                                 \
  TEST_UNARY_BUILTINS_FOR_SCALAR(floor, SCALAR)                                \
  TEST_UNARY_BUILTINS_FOR_SCALAR(round, SCALAR)                                \
  TEST_UNARY_BUILTINS_FOR_SCALAR(log1p, SCALAR)

static void test_builtin_unary_sycl(const Eigen::SyclDevice &sycl_device) {
  int sizeDim1 = 100;
  int sizeDim2 = 100;
  int sizeDim3 = 100;
  array<int, 3> tensorRange = {{sizeDim1, sizeDim2, sizeDim3}};

  TEST_UNARY_BUILTINS(float)
  TEST_UNARY_BUILTINS(double)

}

void test_cxx11_tensor_builtins_sycl() {
  cl::sycl::gpu_selector s;
  Eigen::SyclDevice sycl_device(s);
  CALL_SUBTEST(test_builtin_unary_sycl(sycl_device));
}
