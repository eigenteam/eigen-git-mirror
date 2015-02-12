// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// TODO(mdevin): Free the cuda memory.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX
#define EIGEN_TEST_FUNC cxx11_tensor_cuda
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU


#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::Tensor;

void test_cuda_elementwise_small() {
  Tensor<float, 1> in1(Eigen::array<int, 1>(2));
  Tensor<float, 1> in2(Eigen::array<int, 1>(2));
  Tensor<float, 1> out(Eigen::array<int, 1>(2));
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

  cudaStream_t stream;
  assert(cudaStreamCreate(&stream) == cudaSuccess);
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_in1(
      d_in1, Eigen::array<int, 1>(2));
  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_in2(
      d_in2, Eigen::array<int, 1>(2));
  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_out(
      d_out, Eigen::array<int, 1>(2));

  gpu_out.device(gpu_device) = gpu_in1 + gpu_in2;

  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost,
                         gpu_device.stream()) == cudaSuccess);
  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);

  for (int i = 0; i < 2; ++i) {
    VERIFY_IS_APPROX(
        out(Eigen::array<int, 1>(i)),
        in1(Eigen::array<int, 1>(i)) + in2(Eigen::array<int, 1>(i)));
  }
}

void test_cuda_elementwise()
{
  Tensor<float, 3> in1(Eigen::array<int, 3>(72,53,97));
  Tensor<float, 3> in2(Eigen::array<int, 3>(72,53,97));
  Tensor<float, 3> in3(Eigen::array<int, 3>(72,53,97));
  Tensor<float, 3> out(Eigen::array<int, 3>(72,53,97));
  in1.setRandom();
  in2.setRandom();
  in3.setRandom();

  std::size_t in1_bytes = in1.size() * sizeof(float);
  std::size_t in2_bytes = in2.size() * sizeof(float);
  std::size_t in3_bytes = in3.size() * sizeof(float);
  std::size_t out_bytes = out.size() * sizeof(float);

  float* d_in1;
  float* d_in2;
  float* d_in3;
  float* d_out;
  cudaMalloc((void**)(&d_in1), in1_bytes);
  cudaMalloc((void**)(&d_in2), in2_bytes);
  cudaMalloc((void**)(&d_in3), in3_bytes);
  cudaMalloc((void**)(&d_out), out_bytes);

  cudaMemcpy(d_in1, in1.data(), in1_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_in2, in2.data(), in2_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_in3, in3.data(), in3_bytes, cudaMemcpyHostToDevice);

  cudaStream_t stream;
  assert(cudaStreamCreate(&stream) == cudaSuccess);
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in1(d_in1, Eigen::array<int, 3>(72,53,97));
  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in2(d_in2, Eigen::array<int, 3>(72,53,97));
  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in3(d_in3, Eigen::array<int, 3>(72,53,97));
  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_out(d_out, Eigen::array<int, 3>(72,53,97));

  gpu_out.device(gpu_device) = gpu_in1 + gpu_in2 * gpu_in3;

  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);

  for (int i = 0; i < 72; ++i) {
    for (int j = 0; j < 53; ++j) {
      for (int k = 0; k < 97; ++k) {
        VERIFY_IS_APPROX(out(Eigen::array<int, 3>(i,j,k)), in1(Eigen::array<int, 3>(i,j,k)) + in2(Eigen::array<int, 3>(i,j,k)) * in3(Eigen::array<int, 3>(i,j,k)));
      }
    }
  }
}


void test_cuda_reduction()
{
  Tensor<float, 4> in1(Eigen::array<int, 4>(72,53,97,113));
  Tensor<float, 2> out(Eigen::array<int, 2>(72,97));
  in1.setRandom();

  std::size_t in1_bytes = in1.size() * sizeof(float);
  std::size_t out_bytes = out.size() * sizeof(float);

  float* d_in1;
  float* d_out;
  cudaMalloc((void**)(&d_in1), in1_bytes);
  cudaMalloc((void**)(&d_out), out_bytes);

  cudaMemcpy(d_in1, in1.data(), in1_bytes, cudaMemcpyHostToDevice);

  cudaStream_t stream;
  assert(cudaStreamCreate(&stream) == cudaSuccess);
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 4> > gpu_in1(d_in1, Eigen::array<int, 4>(72,53,97,113));
  Eigen::TensorMap<Eigen::Tensor<float, 2> > gpu_out(d_out, Eigen::array<int, 2>(72,97));

  array<int, 2> reduction_axis;
  reduction_axis[0] = 1;
  reduction_axis[1] = 3;

  gpu_out.device(gpu_device) = gpu_in1.maximum(reduction_axis);

  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);

  for (int i = 0; i < 72; ++i) {
    for (int j = 0; j < 97; ++j) {
      float expected = 0;
      for (int k = 0; k < 53; ++k) {
        for (int l = 0; l < 113; ++l) {
          expected =
              std::max<float>(expected, in1(Eigen::array<int, 4>(i, k, j, l)));
        }
      }
      VERIFY_IS_APPROX(out(Eigen::array<int, 2>(i,j)), expected);
    }
  }
}

template<int DataLayout>
static void test_cuda_contraction()
{
  // with these dimensions, the output has 300 * 140 elements, which is
  // more than 30 * 1024, which is the number of threads in blocks on
  // a 15 SM GK110 GPU
  Tensor<float, 4, DataLayout> t_left(Eigen::array<int, 4>(6, 50, 3, 31));
  Tensor<float, 5, DataLayout> t_right(Eigen::array<int, 5>(3, 31, 7, 20, 1));
  Tensor<float, 5, DataLayout> t_result(Eigen::array<int, 5>(6, 50, 7, 20, 1));

  t_left.setRandom();
  t_right.setRandom();

  std::size_t t_left_bytes = t_left.size()  * sizeof(float);
  std::size_t t_right_bytes = t_right.size() * sizeof(float);
  std::size_t t_result_bytes = t_result.size() * sizeof(float);

  float* d_t_left;
  float* d_t_right;
  float* d_t_result;

  cudaMalloc((void**)(&d_t_left), t_left_bytes);
  cudaMalloc((void**)(&d_t_right), t_right_bytes);
  cudaMalloc((void**)(&d_t_result), t_result_bytes);

  cudaMemcpy(d_t_left, t_left.data(), t_left_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_t_right, t_right.data(), t_right_bytes, cudaMemcpyHostToDevice);

  cudaStream_t stream;
  assert(cudaStreamCreate(&stream) == cudaSuccess);
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout> >
      gpu_t_left(d_t_left, Eigen::array<int, 4>(6, 50, 3, 31));
  Eigen::TensorMap<Eigen::Tensor<float, 5, DataLayout> >
      gpu_t_right(d_t_right, Eigen::array<int, 5>(3, 31, 7, 20, 1));
  Eigen::TensorMap<Eigen::Tensor<float, 5, DataLayout> >
      gpu_t_result(d_t_result, Eigen::array<int, 5>(6, 50, 7, 20, 1));

  typedef Eigen::Map<Eigen::Matrix<float, Dynamic, Dynamic, DataLayout> > MapXf;
  MapXf m_left(t_left.data(), 300, 93);
  MapXf m_right(t_right.data(), 93, 140);
  Eigen::Matrix<float, Dynamic, Dynamic, DataLayout> m_result(300, 140);

  typedef Tensor<float, 1>::DimensionPair DimPair;
  Eigen::array<DimPair, 2> dims;
  dims[0] = DimPair(2, 0);
  dims[1] = DimPair(3, 1);

  m_result = m_left * m_right;
  gpu_t_result.device(gpu_device) = gpu_t_left.contract(gpu_t_right, dims);

  cudaMemcpy(t_result.data(), d_t_result, t_result_bytes, cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < t_result.dimensions().TotalSize(); i++) {
    if (fabs(t_result.data()[i] - m_result.data()[i]) >= 1e-4) {
      cout << "mismatch detected at index " << i << ": " << t_result.data()[i] << " vs " <<  m_result.data()[i] << endl;
      assert(false);
    }
  }
}

static void test_cuda_convolution_1d()
{
  Tensor<float, 4> input(Eigen::array<int, 4>(74,37,11,137));
  Tensor<float, 1> kernel(Eigen::array<int, 1>(4));
  Tensor<float, 4> out(Eigen::array<int, 4>(74,34,11,137));
  input = input.constant(10.0f) + input.random();
  kernel = kernel.constant(7.0f) + kernel.random();

  std::size_t input_bytes = input.size() * sizeof(float);
  std::size_t kernel_bytes = kernel.size() * sizeof(float);
  std::size_t out_bytes = out.size() * sizeof(float);

  float* d_input;
  float* d_kernel;
  float* d_out;
  cudaMalloc((void**)(&d_input), input_bytes);
  cudaMalloc((void**)(&d_kernel), kernel_bytes);
  cudaMalloc((void**)(&d_out), out_bytes);

  cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, kernel.data(), kernel_bytes, cudaMemcpyHostToDevice);

  cudaStream_t stream;
  assert(cudaStreamCreate(&stream) == cudaSuccess);
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 4> > gpu_input(d_input, Eigen::array<int, 4>(74,37,11,137));
  Eigen::TensorMap<Eigen::Tensor<float, 1> > gpu_kernel(d_kernel, Eigen::array<int, 1>(4));
  Eigen::TensorMap<Eigen::Tensor<float, 4> > gpu_out(d_out, Eigen::array<int, 4>(74,34,11,137));

  Eigen::array<int, 1> dims(1);
  gpu_out.device(gpu_device) = gpu_input.convolve(gpu_kernel, dims);

  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);

  for (int i = 0; i < 74; ++i) {
    for (int j = 0; j < 34; ++j) {
      for (int k = 0; k < 11; ++k) {
        for (int l = 0; l < 137; ++l) {
          const float result = out(Eigen::array<int, 4>(i,j,k,l));
          const float expected = input(Eigen::array<int, 4>(i,j+0,k,l)) * kernel(Eigen::array<int, 1>(0)) +
                                 input(Eigen::array<int, 4>(i,j+1,k,l)) * kernel(Eigen::array<int, 1>(1)) +
                                 input(Eigen::array<int, 4>(i,j+2,k,l)) * kernel(Eigen::array<int, 1>(2)) +
                                 input(Eigen::array<int, 4>(i,j+3,k,l)) * kernel(Eigen::array<int, 1>(3));
          VERIFY_IS_APPROX(result, expected);
        }
      }
    }
  }
}


static void test_cuda_convolution_2d()
{
  Tensor<float, 4> input(Eigen::array<int, 4>(74,37,11,137));
  Tensor<float, 2> kernel(Eigen::array<int, 2>(3,4));
  Tensor<float, 4> out(Eigen::array<int, 4>(74,35,8,137));
  input = input.constant(10.0f) + input.random();
  kernel = kernel.constant(7.0f) + kernel.random();

  std::size_t input_bytes = input.size() * sizeof(float);
  std::size_t kernel_bytes = kernel.size() * sizeof(float);
  std::size_t out_bytes = out.size() * sizeof(float);

  float* d_input;
  float* d_kernel;
  float* d_out;
  cudaMalloc((void**)(&d_input), input_bytes);
  cudaMalloc((void**)(&d_kernel), kernel_bytes);
  cudaMalloc((void**)(&d_out), out_bytes);

  cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, kernel.data(), kernel_bytes, cudaMemcpyHostToDevice);

  cudaStream_t stream;
  assert(cudaStreamCreate(&stream) == cudaSuccess);
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 4> > gpu_input(d_input, Eigen::array<int, 4>(74,37,11,137));
  Eigen::TensorMap<Eigen::Tensor<float, 2> > gpu_kernel(d_kernel, Eigen::array<int, 2>(3,4));
  Eigen::TensorMap<Eigen::Tensor<float, 4> > gpu_out(d_out, Eigen::array<int, 4>(74,35,8,137));

  Eigen::array<int, 2> dims(1,2);
  gpu_out.device(gpu_device) = gpu_input.convolve(gpu_kernel, dims);

  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);

  for (int i = 0; i < 74; ++i) {
    for (int j = 0; j < 35; ++j) {
      for (int k = 0; k < 8; ++k) {
        for (int l = 0; l < 137; ++l) {
          const float result = out(Eigen::array<int, 4>(i,j,k,l));
          const float expected = input(Eigen::array<int, 4>(i,j+0,k+0,l)) * kernel(Eigen::array<int, 2>(0,0)) +
                                 input(Eigen::array<int, 4>(i,j+1,k+0,l)) * kernel(Eigen::array<int, 2>(1,0)) +
                                 input(Eigen::array<int, 4>(i,j+2,k+0,l)) * kernel(Eigen::array<int, 2>(2,0)) +
                                 input(Eigen::array<int, 4>(i,j+0,k+1,l)) * kernel(Eigen::array<int, 2>(0,1)) +
                                 input(Eigen::array<int, 4>(i,j+1,k+1,l)) * kernel(Eigen::array<int, 2>(1,1)) +
                                 input(Eigen::array<int, 4>(i,j+2,k+1,l)) * kernel(Eigen::array<int, 2>(2,1)) +
                                 input(Eigen::array<int, 4>(i,j+0,k+2,l)) * kernel(Eigen::array<int, 2>(0,2)) +
                                 input(Eigen::array<int, 4>(i,j+1,k+2,l)) * kernel(Eigen::array<int, 2>(1,2)) +
                                 input(Eigen::array<int, 4>(i,j+2,k+2,l)) * kernel(Eigen::array<int, 2>(2,2)) +
                                 input(Eigen::array<int, 4>(i,j+0,k+3,l)) * kernel(Eigen::array<int, 2>(0,3)) +
                                 input(Eigen::array<int, 4>(i,j+1,k+3,l)) * kernel(Eigen::array<int, 2>(1,3)) +
                                 input(Eigen::array<int, 4>(i,j+2,k+3,l)) * kernel(Eigen::array<int, 2>(2,3));
            VERIFY_IS_APPROX(result, expected);
        }
      }
    }
  }
}


static void test_cuda_convolution_3d()
{
  Tensor<float, 5> input(Eigen::array<int, 5>(74,37,11,137,17));
  Tensor<float, 3> kernel(Eigen::array<int, 3>(3,4,2));
  Tensor<float, 5> out(Eigen::array<int, 5>(74,35,8,136,17));
  input = input.constant(10.0f) + input.random();
  kernel = kernel.constant(7.0f) + kernel.random();

  std::size_t input_bytes = input.size() * sizeof(float);
  std::size_t kernel_bytes = kernel.size() * sizeof(float);
  std::size_t out_bytes = out.size() * sizeof(float);

  float* d_input;
  float* d_kernel;
  float* d_out;
  cudaMalloc((void**)(&d_input), input_bytes);
  cudaMalloc((void**)(&d_kernel), kernel_bytes);
  cudaMalloc((void**)(&d_out), out_bytes);

  cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, kernel.data(), kernel_bytes, cudaMemcpyHostToDevice);

  cudaStream_t stream;
  assert(cudaStreamCreate(&stream) == cudaSuccess);
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 5> > gpu_input(d_input, Eigen::array<int, 5>(74,37,11,137,17));
  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_kernel(d_kernel, Eigen::array<int, 3>(3,4,2));
  Eigen::TensorMap<Eigen::Tensor<float, 5> > gpu_out(d_out, Eigen::array<int, 5>(74,35,8,136,17));

  Eigen::array<int, 3> dims(1,2,3);
  gpu_out.device(gpu_device) = gpu_input.convolve(gpu_kernel, dims);

  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);

  for (int i = 0; i < 74; ++i) {
    for (int j = 0; j < 35; ++j) {
      for (int k = 0; k < 8; ++k) {
        for (int l = 0; l < 136; ++l) {
          for (int m = 0; m < 17; ++m) {
            const float result = out(Eigen::array<int, 5>(i,j,k,l,m));
            const float expected = input(Eigen::array<int, 5>(i,j+0,k+0,l+0,m)) * kernel(Eigen::array<int, 3>(0,0,0)) +
                                   input(Eigen::array<int, 5>(i,j+1,k+0,l+0,m)) * kernel(Eigen::array<int, 3>(1,0,0)) +
                                   input(Eigen::array<int, 5>(i,j+2,k+0,l+0,m)) * kernel(Eigen::array<int, 3>(2,0,0)) +
                                   input(Eigen::array<int, 5>(i,j+0,k+1,l+0,m)) * kernel(Eigen::array<int, 3>(0,1,0)) +
                                   input(Eigen::array<int, 5>(i,j+1,k+1,l+0,m)) * kernel(Eigen::array<int, 3>(1,1,0)) +
                                   input(Eigen::array<int, 5>(i,j+2,k+1,l+0,m)) * kernel(Eigen::array<int, 3>(2,1,0)) +
                                   input(Eigen::array<int, 5>(i,j+0,k+2,l+0,m)) * kernel(Eigen::array<int, 3>(0,2,0)) +
                                   input(Eigen::array<int, 5>(i,j+1,k+2,l+0,m)) * kernel(Eigen::array<int, 3>(1,2,0)) +
                                   input(Eigen::array<int, 5>(i,j+2,k+2,l+0,m)) * kernel(Eigen::array<int, 3>(2,2,0)) +
                                   input(Eigen::array<int, 5>(i,j+0,k+3,l+0,m)) * kernel(Eigen::array<int, 3>(0,3,0)) +
                                   input(Eigen::array<int, 5>(i,j+1,k+3,l+0,m)) * kernel(Eigen::array<int, 3>(1,3,0)) +
                                   input(Eigen::array<int, 5>(i,j+2,k+3,l+0,m)) * kernel(Eigen::array<int, 3>(2,3,0)) +
                                   input(Eigen::array<int, 5>(i,j+0,k+0,l+1,m)) * kernel(Eigen::array<int, 3>(0,0,1)) +
                                   input(Eigen::array<int, 5>(i,j+1,k+0,l+1,m)) * kernel(Eigen::array<int, 3>(1,0,1)) +
                                   input(Eigen::array<int, 5>(i,j+2,k+0,l+1,m)) * kernel(Eigen::array<int, 3>(2,0,1)) +
                                   input(Eigen::array<int, 5>(i,j+0,k+1,l+1,m)) * kernel(Eigen::array<int, 3>(0,1,1)) +
                                   input(Eigen::array<int, 5>(i,j+1,k+1,l+1,m)) * kernel(Eigen::array<int, 3>(1,1,1)) +
                                   input(Eigen::array<int, 5>(i,j+2,k+1,l+1,m)) * kernel(Eigen::array<int, 3>(2,1,1)) +
                                   input(Eigen::array<int, 5>(i,j+0,k+2,l+1,m)) * kernel(Eigen::array<int, 3>(0,2,1)) +
                                   input(Eigen::array<int, 5>(i,j+1,k+2,l+1,m)) * kernel(Eigen::array<int, 3>(1,2,1)) +
                                   input(Eigen::array<int, 5>(i,j+2,k+2,l+1,m)) * kernel(Eigen::array<int, 3>(2,2,1)) +
                                   input(Eigen::array<int, 5>(i,j+0,k+3,l+1,m)) * kernel(Eigen::array<int, 3>(0,3,1)) +
                                   input(Eigen::array<int, 5>(i,j+1,k+3,l+1,m)) * kernel(Eigen::array<int, 3>(1,3,1)) +
                                   input(Eigen::array<int, 5>(i,j+2,k+3,l+1,m)) * kernel(Eigen::array<int, 3>(2,3,1));
            VERIFY_IS_APPROX(result, expected);
          }
        }
      }
    }
  }
}

static float* CudaCopyFloat(float* data, int size) {
  const int nbytes = size * sizeof(float);
  float* result = NULL;
  if (cudaMalloc((void**)(&result), nbytes) != cudaSuccess) {
    return NULL;
  } else {
    if (data != NULL) {
      cudaMemcpy(result, data, nbytes, cudaMemcpyHostToDevice);
    }
    return result;
  }
}

static void test_cuda_constant_broadcast()
{
  cudaStream_t stream;
  assert(cudaStreamCreate(&stream) == cudaSuccess);
  Eigen::GpuDevice gpu_device(&stream);

  Tensor<float, 1> t1(10);
  for (int i = 0; i < 10; ++i) {
    t1(i) = 10.0f * i;
  }
  float* t1_cuda = CudaCopyFloat(t1.data(), t1.size());
  Eigen::TensorMap<Eigen::Tensor<float, 1> > t1_gpu(t1_cuda, 10);

  Tensor<float, 1> t2(1);
  t2 = t2.constant(20.0f);
  float* t2_cuda = CudaCopyFloat(t2.data(), t2.size());
  Eigen::TensorMap<Eigen::TensorFixedSize<float, Sizes<1> > > t2_gpu(t2_cuda, 1);

  float* t3_cuda = CudaCopyFloat(NULL, 10);
  Eigen::TensorMap<Eigen::Tensor<float, 1> > t3_gpu(t3_cuda, 10);

  t3_gpu.device(gpu_device) =
      t1_gpu + t2_gpu.broadcast(Eigen::array<int, 1>(10));

  Eigen::Tensor<float, 1> t3(10);
  cudaMemcpy(t3.data(), t3_gpu.data(), 10 * sizeof(float),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < 10; ++i) {
    VERIFY_IS_APPROX(t3(i), t1(i) + t2(0));
  }
}


void test_cuda_cast()
{
  Tensor<double, 3> in(Eigen::array<int, 3>(72,53,97));
  Tensor<float, 3> out(Eigen::array<int, 3>(72,53,97));
  in.setRandom();

  std::size_t in_bytes = in.size() * sizeof(double);
  std::size_t out_bytes = out.size() * sizeof(float);

  double* d_in;
  float* d_out;
  cudaMalloc((void**)(&d_in), in_bytes);
  cudaMalloc((void**)(&d_out), out_bytes);

  cudaMemcpy(d_in, in.data(), in_bytes, cudaMemcpyHostToDevice);

  cudaStream_t stream;
  assert(cudaStreamCreate(&stream) == cudaSuccess);
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<double, 3> > gpu_in(d_in, Eigen::array<int, 3>(72,53,97));
  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_out(d_out, Eigen::array<int, 3>(72,53,97));

  gpu_out.device(gpu_device) = gpu_in.template cast<float>();

  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);

  for (int i = 0; i < 72; ++i) {
    for (int j = 0; j < 53; ++j) {
      for (int k = 0; k < 97; ++k) {
        VERIFY_IS_APPROX(out(Eigen::array<int, 3>(i,j,k)), static_cast<float>(in(Eigen::array<int, 3>(i,j,k))));
      }
    }
  }
}


void test_cxx11_tensor_cuda()
{
  CALL_SUBTEST(test_cuda_elementwise_small());
  CALL_SUBTEST(test_cuda_elementwise());
  CALL_SUBTEST(test_cuda_reduction());
  CALL_SUBTEST(test_cuda_contraction<ColMajor>());
  CALL_SUBTEST(test_cuda_contraction<RowMajor>());
  CALL_SUBTEST(test_cuda_convolution_1d());
  CALL_SUBTEST(test_cuda_convolution_2d());
  CALL_SUBTEST(test_cuda_convolution_3d());
  CALL_SUBTEST(test_cuda_constant_broadcast());
  CALL_SUBTEST(test_cuda_cast());
}
