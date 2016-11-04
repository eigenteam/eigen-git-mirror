// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015
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
#define EIGEN_TEST_FUNC cxx11_tensor_reduction_sycl
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_SYCL

#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>



static void test_full_reductions_sycl() {


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
  Eigen::SyclDevice sycl_device(q);

  const int num_rows = 452;
  const int num_cols = 765;
  array<int, 2> tensorRange = {{num_rows, num_cols}};

  Tensor<float, 2> in(tensorRange);
  in.setRandom();

  Tensor<float, 0> full_redux;
  Tensor<float, 0> full_redux_g;
  full_redux = in.sum();
  float* out_data = (float*)sycl_device.allocate(sizeof(float));
  TensorMap<Tensor<float, 2> >  in_gpu(in.data(), tensorRange);
  TensorMap<Tensor<float, 0> >  full_redux_gpu(out_data);
  full_redux_gpu.device(sycl_device) = in_gpu.sum();
  sycl_device.deallocate(out_data);
  // Check that the CPU and GPU reductions return the same result.
  VERIFY_IS_APPROX(full_redux_gpu(), full_redux());

}


static void test_first_dim_reductions_sycl() {


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
  Eigen::SyclDevice sycl_device(q);

  int dim_x = 145;
  int dim_y = 1;
  int dim_z = 67;

  array<int, 3> tensorRange = {{dim_x, dim_y, dim_z}};

  Tensor<float, 3> in(tensorRange);
  in.setRandom();
  Eigen::array<int, 1> red_axis;
  red_axis[0] = 0;
  Tensor<float, 2> redux = in.sum(red_axis);
  array<int, 2> reduced_tensorRange = {{dim_y, dim_z}};
  Tensor<float, 2> redux_g(reduced_tensorRange);
  TensorMap<Tensor<float, 3> >  in_gpu(in.data(), tensorRange);
  float* out_data = (float*)sycl_device.allocate(dim_y*dim_z*sizeof(float));
  TensorMap<Tensor<float, 2> >  redux_gpu(out_data, dim_y, dim_z );
  redux_gpu.device(sycl_device) = in_gpu.sum(red_axis);

  sycl_device.deallocate(out_data);
  // Check that the CPU and GPU reductions return the same result.
  for(int j=0; j<dim_y; j++ )
    for(int k=0; k<dim_z; k++ )
      VERIFY_IS_APPROX(redux_gpu(j,k), redux(j,k));
}


static void test_last_dim_reductions_sycl() {


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
  Eigen::SyclDevice sycl_device(q);

  int dim_x = 567;
  int dim_y = 1;
  int dim_z = 47;

  array<int, 3> tensorRange = {{dim_x, dim_y, dim_z}};

  Tensor<float, 3> in(tensorRange);
  in.setRandom();
  Eigen::array<int, 1> red_axis;
  red_axis[0] = 2;
  Tensor<float, 2> redux = in.sum(red_axis);
  array<int, 2> reduced_tensorRange = {{dim_x, dim_y}};
  Tensor<float, 2> redux_g(reduced_tensorRange);
  TensorMap<Tensor<float, 3> >  in_gpu(in.data(), tensorRange);
  float* out_data = (float*)sycl_device.allocate(dim_x*dim_y*sizeof(float));
  TensorMap<Tensor<float, 2> >  redux_gpu(out_data, dim_x, dim_y );
  redux_gpu.device(sycl_device) = in_gpu.sum(red_axis);

  sycl_device.deallocate(out_data);
  // Check that the CPU and GPU reductions return the same result.
  for(int j=0; j<dim_x; j++ )
    for(int k=0; k<dim_y; k++ )
      VERIFY_IS_APPROX(redux_gpu(j,k), redux(j,k));
}

void test_cxx11_tensor_reduction_sycl() {
  CALL_SUBTEST((test_full_reductions_sycl()));
  CALL_SUBTEST((test_first_dim_reductions_sycl()));
  CALL_SUBTEST((test_last_dim_reductions_sycl()));

}
