// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX
#define EIGEN_TEST_FUNC cxx11_tensor_device
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU


#include "main.h"
#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;
using Eigen::RowMajor;

// Context for evaluation on cpu
struct CPUContext {
  CPUContext(const Eigen::Tensor<float, 3>& in1, Eigen::Tensor<float, 3>& in2, Eigen::Tensor<float, 3>& out) : in1_(in1), in2_(in2), out_(out) { }

  const Eigen::Tensor<float, 3>& in1() const { return in1_; }
  const Eigen::Tensor<float, 3>& in2() const { return in2_; }
  Eigen::TensorDevice<Eigen::Tensor<float, 3>, Eigen::DefaultDevice> out() { return TensorDevice<Eigen::Tensor<float, 3>, Eigen::DefaultDevice>(cpu_device_, out_); }

 private:
  const Eigen::Tensor<float, 3>& in1_;
  const Eigen::Tensor<float, 3>& in2_;
  Eigen::Tensor<float, 3>& out_;

  Eigen::DefaultDevice cpu_device_;
};


// Context for evaluation on GPU
struct GPUContext {
  GPUContext(const Eigen::TensorMap<Eigen::Tensor<float, 3> >& in1, Eigen::TensorMap<Eigen::Tensor<float, 3> >& in2, Eigen::TensorMap<Eigen::Tensor<float, 3> >& out) : in1_(in1), in2_(in2), out_(out) { }

  const Eigen::TensorMap<Eigen::Tensor<float, 3> >& in1() const { return in1_; }
  const Eigen::TensorMap<Eigen::Tensor<float, 3> >& in2() const { return in2_; }
  Eigen::TensorDevice<Eigen::TensorMap<Eigen::Tensor<float, 3> >, Eigen::GpuDevice> out() { return TensorDevice<Eigen::TensorMap<Eigen::Tensor<float, 3> >, Eigen::GpuDevice>(gpu_device_, out_); }

 private:
  const Eigen::TensorMap<Eigen::Tensor<float, 3> >& in1_;
  const Eigen::TensorMap<Eigen::Tensor<float, 3> >& in2_;
  Eigen::TensorMap<Eigen::Tensor<float, 3> >& out_;
  Eigen::GpuDevice gpu_device_;
};


// The actual expression to evaluate
template <typename Context>
static void test_contextual_eval(Context* context)
{
  context->out() = context->in1() + context->in2() * 3.14f;
}

static void test_cpu() {
  Eigen::Tensor<float, 3> in1(Eigen::array<int, 3>(2,3,7));
  Eigen::Tensor<float, 3> in2(Eigen::array<int, 3>(2,3,7));
  Eigen::Tensor<float, 3> out(Eigen::array<int, 3>(2,3,7));

  in1.setRandom();
  in2.setRandom();
  CPUContext context(in1, in2, out);
  test_contextual_eval(&context);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        VERIFY_IS_APPROX(out(Eigen::array<int, 3>(i,j,k)), in1(Eigen::array<int, 3>(i,j,k)) + in2(Eigen::array<int, 3>(i,j,k)) * 3.14f);
      }
    }
  }
}

static void test_gpu() {
  Eigen::Tensor<float, 3> in1(Eigen::array<int, 3>(2,3,7));
  Eigen::Tensor<float, 3> in2(Eigen::array<int, 3>(2,3,7));
  Eigen::Tensor<float, 3> out(Eigen::array<int, 3>(2,3,7));
  in1.setRandom();
  in2.setRandom();

  std::size_t in1_bytes = in1.size() * sizeof(float);
  std::size_t in2_bytes = in2.size() * sizeof(float);
  std::size_t out_bytes = out.size() * sizeof(float);

  float* d_in1;
  float* d_in2;
  float* d_out;
  cudaMalloc((void**)(&d_in1), in1_bytes);
  cudaMalloc((void**)(&d_in2), in2_bytes);
  cudaMalloc((void**)(&d_out), out_bytes);

  cudaMemcpy(d_in1, in1.data(), in1_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_in2, in2.data(), in2_bytes, cudaMemcpyHostToDevice);

  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in1(d_in1, Eigen::array<int, 3>(2,3,7));
  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in2(d_in2, Eigen::array<int, 3>(2,3,7));
  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_out(d_out, Eigen::array<int, 3>(2,3,7));

  GPUContext context(gpu_in1, gpu_in2, gpu_out);
  test_contextual_eval(&context);

  cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        VERIFY_IS_APPROX(out(Eigen::array<int, 3>(i,j,k)), in1(Eigen::array<int, 3>(i,j,k)) + in2(Eigen::array<int, 3>(i,j,k)) * 3.14f);
      }
    }
  }
}



void test_cxx11_tensor_device()
{
  CALL_SUBTEST(test_cpu());
  CALL_SUBTEST(test_gpu());
}
