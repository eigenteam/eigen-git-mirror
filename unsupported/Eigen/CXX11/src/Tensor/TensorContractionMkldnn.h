// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2018 Eugene Zhulenev <ezhulenev@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
#ifndef EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_MKLDNN_H
#define EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_MKLDNN_H

#if defined(EIGEN_USE_MKLDNN)
// Support for MklDnn sgemm kernel in Tensor contractions:
//
// 1. Prepare packed Lhs/Rhs blocks from tensor expressions using
//    DataMapper (see TensorContractionInputMapper).
// 2. Invoke gemm kernel with packed blocks (replacement for default
// gebp_kernel).

namespace Eigen {
namespace internal {

template <typename Scalar, typename StorageIndex, typename DataMapper,
          int StorageOrder>
struct mkldnn_gemm_pack;

// mkl_gemm_pack for ColMajor storage order.
template <typename Scalar, typename StorageIndex, typename DataMapper>
struct mkldnn_gemm_pack<Scalar, StorageIndex, DataMapper,
                        /*StorageOrder*/ ColMajor> {
  typedef typename internal::packet_traits<Scalar>::type Packet;
  typedef typename DataMapper::LinearMapper LinearMapper;

  enum { PacketSize = internal::packet_traits<Scalar>::size };

  EIGEN_DONT_INLINE
  void operator()(Scalar *block, const DataMapper &data_mapper,
                  StorageIndex rows, StorageIndex cols) {
    const StorageIndex unrolled_rows =
        (rows / (4 * PacketSize)) * (4 * PacketSize);
    const StorageIndex vectorized_rows = (rows / PacketSize) * PacketSize;

    for (StorageIndex col = 0; col < cols; ++col) {
      LinearMapper lm = data_mapper.getLinearMapper(0, col);

      // Give compiler a strong possibility to unroll the loop.
      for (StorageIndex i = 0; i < unrolled_rows; i += 4 * PacketSize) {
        for (StorageIndex j = 0; j < 4; ++j) {
          const Packet p = lm.template loadPacket<Packet>(i + j * PacketSize);
          internal::pstoreu(block + j * PacketSize, p);
        }
        block += 4 * PacketSize;
      }

      // Process remaining rows with packets.
      for (StorageIndex i = unrolled_rows; i < vectorized_rows;
           i += PacketSize) {
        const Packet p = lm.template loadPacket<Packet>(i);
        internal::pstoreu(block, p);
        block += PacketSize;
      }

      // Finalize with coefficients.
      for (StorageIndex i = vectorized_rows; i < rows; ++i) {
        *block = lm(i);
        ++block;
      }
    }
  }
};

template <typename Scalar, typename StorageIndex, typename OutputMapper,
          bool ConjugateLhs = false, bool ConjugateRhs = false>
struct mkldnn_gemm_kernel;

// mkldnn_gemm_kernel for floats defined as a thin layer on top of mkldnn_sgemm.
template <typename StorageIndex, typename OutputMapper, bool ConjugateLhs,
          bool ConjugateRhs>
struct mkldnn_gemm_kernel</*Scalar*/ float, StorageIndex, OutputMapper,
                          ConjugateLhs, ConjugateRhs> {
  EIGEN_DONT_INLINE
  void operator()(const OutputMapper &output, const float *blockA,
                  const float *blockB, const StorageIndex rows,
                  const StorageIndex depth, const StorageIndex cols,
                  float alpha) {
    static const int max_index = (std::numeric_limits<int>::max)();

    eigen_assert(max_index > rows);
    eigen_assert(max_index > cols);
    eigen_assert(max_index > depth);
    eigen_assert(max_index > output.stride());

    const int m = static_cast<int>(rows);
    const int n = static_cast<int>(cols);
    const int k = static_cast<int>(depth);

    const char transposeA = ConjugateLhs ? 'Y' : 'N';
    const char transposeB = ConjugateRhs ? 'Y' : 'N';

    const int ldA = ConjugateLhs ? k : m;
    const int ldB = ConjugateRhs ? n : k;
    const int ldC = static_cast<int>(output.stride());

    const float beta = 1.0;

    mkldnn_status_t st = mkldnn_sgemm(&transposeA, &transposeB, &m, &n, &k,
                                      &alpha, blockA, &ldA, blockB, &ldB, &beta,
                                      const_cast<float*>(output.data()), &ldC);
    eigen_assert(st == 0);
  }
};

}  // namespace internal
}  // namespace Eigen
#endif  // EIGEN_USE_MKLDNN
#endif  // EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_MKLDNN_H
