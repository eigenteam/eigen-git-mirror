// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2018 Eugene Zhulenev <ezhulenev@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/CXX11/Tensor>

using Eigen::internal::blas_data_mapper;
using Eigen::internal::mkldnn_gemm_kernel;
using Eigen::internal::mkldnn_gemm_pack;

template <int NumDims>
static array<Index, NumDims> RandomDims(int min_dim = 1, int max_dim = 20) {
  array<Index, NumDims> dims;
  for (int i = 0; i < NumDims; ++i) {
    dims[i] = internal::random<int>(min_dim, max_dim);
  }
  return dims;
}

// Packing with mkldnn_gemm_pack is the same as taking a slice of 2 dimensional
// Tensor.
template <typename Scalar>
static void test_mkldnn_gemm_pack() {
  static const int Options = 0 | ColMajor;

  typedef blas_data_mapper<Scalar, Index, ColMajor> DataMapper;
  typedef mkldnn_gemm_pack<Scalar, Index, DataMapper, ColMajor> MkldnnGemmPack;
  typedef Tensor<Scalar, 2, Options, Index> Tensor2d;

  array<Index, 2> dims = RandomDims<2>(1, 500);

  // Create a tensor initialized with random data.
  Tensor2d src(dims);
  src.setRandom();

  // Pick a random slice of src tensor.
  array<Index, 2> slice_start = RandomDims<2>(0, 250);
  array<Index, 2> slice_size = RandomDims<2>(100, 500);
  // Make sure that slice start + size do not overflow tensor dims.
  for (int i = 0; i < 2; ++i) {
    slice_start[i] = numext::mini(dims[i] - 1, slice_start[i]);
    slice_size[i] = numext::mini(slice_size[i], dims[i] - slice_start[i]);
  }

  // Prepare tensors for packing and slicing results.
  Tensor2d pack_dst(slice_size[0], slice_size[1]);
  Tensor2d slice_dst(slice_size[0], slice_size[1]);

  // Pack memory using mkldnn_gemm_pack.
  DataMapper data_mapper(src.data(), dims[0]);
  MkldnnGemmPack gemm_pack;
  gemm_pack(pack_dst.data(),
            data_mapper.getSubMapper(slice_start[0], slice_start[1]),
            slice_size[0], slice_size[1]);
  // Slice the source tensor.
  slice_dst = src.slice(slice_start, slice_size);

  // Verify that dst tensors are equal.
  VERIFY_IS_EQUAL(pack_dst.dimensions().TotalSize(),
                  slice_dst.dimensions().TotalSize());
  for (Index i = 0; i < pack_dst.dimensions().TotalSize(); ++i) {
    Scalar packed = pack_dst.coeff(i);
    Scalar sliced = slice_dst.coeff(i);
    VERIFY_IS_EQUAL(packed, sliced);
  }
}
template <typename Scalar>
static void test_mkldnn_gemm_kernel() {
  static const int Options = 0 | ColMajor;

  typedef Tensor<Scalar, 2, Options, Index> Tensor2d;

  int m = internal::random<int>(1, 100);
  int n = internal::random<int>(1, 100);
  int k = internal::random<int>(1, 100);

  Tensor2d lhs(m, k);
  lhs.setRandom();

  Tensor2d rhs(k, n);
  rhs.setRandom();

  // Compute matmul with mkldnn gemm kernel.
  typedef blas_data_mapper<Scalar, Index, ColMajor> OutputMapper;
  typedef mkldnn_gemm_kernel<Scalar, Index, OutputMapper, ColMajor>
      MkldnnGemmKernel;

  Tensor2d mkldnn_result(m, n);
  mkldnn_result.setZero();

  OutputMapper output_mapper(mkldnn_result.data(), m);
  MkldnnGemmKernel gemm_kernel;
  gemm_kernel(output_mapper, lhs.data(), rhs.data(), m, k, n, /*alpha*/ 1.0);

  // Compute matmul with Eigen::Matrix.
  typedef Eigen::Matrix<Scalar, Dynamic, Dynamic, ColMajor> Matrix;
  typedef Map<Eigen::Matrix<Scalar, Dynamic, Dynamic, ColMajor> > MatrixMap;

  MatrixMap lhs_mat(lhs.data(), m, k);
  MatrixMap rhs_mat(rhs.data(), k, n);
  Matrix matmul_result(m, n);
  matmul_result.setZero();

  matmul_result = lhs_mat * rhs_mat;

  static const float error_threshold = 1e-4f;

  // Verify that results are equal.
  for (Index i = 0; i < m * n; ++i) {
    Scalar gemm = mkldnn_result(i);
    Scalar matmul = matmul_result(i % m, i / m);
    if ((std::abs)(gemm) > error_threshold &&
        (std::abs)(matmul) > error_threshold) {
      if (!Eigen::internal::isApprox(gemm, matmul, error_threshold))
        std::cout << "gemm=" << gemm << " matmul=" << matmul << std::endl;
      VERIFY(Eigen::internal::isApprox(gemm, matmul, error_threshold));
    }
  }
}

EIGEN_DECLARE_TEST(cxx11_tensor_contraction_mkldnn) {
  CALL_SUBTEST(test_mkldnn_gemm_pack<float>());
  CALL_SUBTEST(test_mkldnn_gemm_pack<double>());

  // mkldnn has only sgemm (aka gemm for floats).
  CALL_SUBTEST(test_mkldnn_gemm_kernel<float>());
}
