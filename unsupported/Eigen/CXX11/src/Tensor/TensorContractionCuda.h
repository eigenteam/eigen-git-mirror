// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_CUDA_H
#define EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_CUDA_H

#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)

namespace Eigen {

template<typename Scalar, typename Index, typename LhsMapper,
         typename RhsMapper, typename OutputMapper, bool needs_edge_check>
__device__ EIGEN_STRONG_INLINE void
EigenContractionKernelInternal(const LhsMapper lhs, const RhsMapper rhs,
                               const OutputMapper output, volatile Scalar* lhs_shmem, volatile Scalar* rhs_shmem,
                               const Index m_size, const Index n_size, const Index k_size) {

  const Index m_block_idx = blockIdx.x;
  const Index n_block_idx = blockIdx.y;

  const Index base_m = 64 * m_block_idx;
  const Index base_n = 64 * n_block_idx;

  // declare and initialize 64 registers for output 8x8 block

  // prefetch registers
  Scalar lhs_pf0;
  Scalar lhs_pf1;
  Scalar lhs_pf2;
  Scalar lhs_pf3;
  Scalar lhs_pf4;
  Scalar lhs_pf5;
  Scalar lhs_pf6;
  Scalar lhs_pf7;

  Scalar rhs_pf0;
  Scalar rhs_pf1;
  Scalar rhs_pf2;
  Scalar rhs_pf3;
  Scalar rhs_pf4;
  Scalar rhs_pf5;
  Scalar rhs_pf6;
  Scalar rhs_pf7;

  // shared memory is formatted
  // (contract idx in block, nocontract idx in block, block idx)
  // where block idx is column major. This transposition limits the number of
  // bank conflicts when reading the LHS. The core idea is that since the contracting
  // index is shared by both sides, then the contracting index should be in threadIdx.x.

  // On the LHS, we pad each row inside of each block with an extra element. This makes
  // each block 8 rows of 9 elements, which is 72 elements. This gives no bank conflicts
  // on writes and very few 2-way conflicts on reads. There is an 8x8 grid of these blocks.

  // On the RHS we just add 8 padding elements to the end of each block. This gives no bank
  // conflicts on writes and also none on reads.

  // storage indices
  const Index lhs_store_idx_base = threadIdx.y * 72 + threadIdx.x * 9 + threadIdx.z;
  const Index rhs_store_idx_base = threadIdx.y * 72 + threadIdx.z * 8 + threadIdx.x;

  const Index lhs_store_idx_0 = lhs_store_idx_base + 576 * 0;
  const Index lhs_store_idx_1 = lhs_store_idx_base + 576 * 1;
  const Index lhs_store_idx_2 = lhs_store_idx_base + 576 * 2;
  const Index lhs_store_idx_3 = lhs_store_idx_base + 576 * 3;
  const Index lhs_store_idx_4 = lhs_store_idx_base + 576 * 4;
  const Index lhs_store_idx_5 = lhs_store_idx_base + 576 * 5;
  const Index lhs_store_idx_6 = lhs_store_idx_base + 576 * 6;
  const Index lhs_store_idx_7 = lhs_store_idx_base + 576 * 7;

  const Index rhs_store_idx_0 = rhs_store_idx_base + 576 * 0;
  const Index rhs_store_idx_1 = rhs_store_idx_base + 576 * 1;
  const Index rhs_store_idx_2 = rhs_store_idx_base + 576 * 2;
  const Index rhs_store_idx_3 = rhs_store_idx_base + 576 * 3;
  const Index rhs_store_idx_4 = rhs_store_idx_base + 576 * 4;
  const Index rhs_store_idx_5 = rhs_store_idx_base + 576 * 5;
  const Index rhs_store_idx_6 = rhs_store_idx_base + 576 * 6;
  const Index rhs_store_idx_7 = rhs_store_idx_base + 576 * 7;

  // in the loading code, the following variables are important:
  // threadIdx.x: the vertical position in an 8x8 block
  // threadIdx.y: the vertical index of the 8x8 block in the grid
  // threadIdx.z: the horizontal position in an 8x8 block
  // k: the horizontal index of the 8x8 block in the grid
  //
  // The k parameter is implicit (it was the loop counter for a loop that went
  // from 0 to <8, but now that loop is unrolled in the below code.

  const Index load_idx_vert = threadIdx.x + 8 * threadIdx.y;
  const Index lhs_vert = base_m + load_idx_vert;

#define prefetchIntoRegisters(base_k)                             \
  {                                                               \
      lhs_pf0 = Scalar(0);                                        \
      lhs_pf1 = Scalar(0);                                        \
      lhs_pf2 = Scalar(0);                                        \
      lhs_pf3 = Scalar(0);                                        \
      lhs_pf4 = Scalar(0);                                        \
      lhs_pf5 = Scalar(0);                                        \
      lhs_pf6 = Scalar(0);                                        \
      lhs_pf7 = Scalar(0);                                        \
                                                                  \
      rhs_pf0 = Scalar(0);                                        \
      rhs_pf1 = Scalar(0);                                        \
      rhs_pf2 = Scalar(0);                                        \
      rhs_pf3 = Scalar(0);                                        \
      rhs_pf4 = Scalar(0);                                        \
      rhs_pf5 = Scalar(0);                                        \
      rhs_pf6 = Scalar(0);                                        \
      rhs_pf7 = Scalar(0);                                        \
                                                                  \
      if (!needs_edge_check || lhs_vert < m_size) {               \
        const Index lhs_horiz_0 = base_k + threadIdx.z + 0 * 8;   \
        const Index lhs_horiz_1 = base_k + threadIdx.z + 1 * 8;   \
        const Index lhs_horiz_2 = base_k + threadIdx.z + 2 * 8;   \
        const Index lhs_horiz_3 = base_k + threadIdx.z + 3 * 8;   \
        const Index lhs_horiz_4 = base_k + threadIdx.z + 4 * 8;   \
        const Index lhs_horiz_5 = base_k + threadIdx.z + 5 * 8;   \
        const Index lhs_horiz_6 = base_k + threadIdx.z + 6 * 8;   \
        const Index lhs_horiz_7 = base_k + threadIdx.z + 7 * 8;   \
                                                                  \
        if (!needs_edge_check || lhs_horiz_7 < k_size) {          \
          lhs_pf0 = lhs(lhs_vert, lhs_horiz_0);                   \
          lhs_pf1 = lhs(lhs_vert, lhs_horiz_1);                   \
          lhs_pf2 = lhs(lhs_vert, lhs_horiz_2);                   \
          lhs_pf3 = lhs(lhs_vert, lhs_horiz_3);                   \
          lhs_pf4 = lhs(lhs_vert, lhs_horiz_4);                   \
          lhs_pf5 = lhs(lhs_vert, lhs_horiz_5);                   \
          lhs_pf6 = lhs(lhs_vert, lhs_horiz_6);                   \
          lhs_pf7 = lhs(lhs_vert, lhs_horiz_7);                   \
        } else if (lhs_horiz_6 < k_size) {                        \
          lhs_pf0 = lhs(lhs_vert, lhs_horiz_0);                   \
          lhs_pf1 = lhs(lhs_vert, lhs_horiz_1);                   \
          lhs_pf2 = lhs(lhs_vert, lhs_horiz_2);                   \
          lhs_pf3 = lhs(lhs_vert, lhs_horiz_3);                   \
          lhs_pf4 = lhs(lhs_vert, lhs_horiz_4);                   \
          lhs_pf5 = lhs(lhs_vert, lhs_horiz_5);                   \
          lhs_pf6 = lhs(lhs_vert, lhs_horiz_6);                   \
        } else if (lhs_horiz_5 < k_size) {                        \
          lhs_pf0 = lhs(lhs_vert, lhs_horiz_0);                   \
          lhs_pf1 = lhs(lhs_vert, lhs_horiz_1);                   \
          lhs_pf2 = lhs(lhs_vert, lhs_horiz_2);                   \
          lhs_pf3 = lhs(lhs_vert, lhs_horiz_3);                   \
          lhs_pf4 = lhs(lhs_vert, lhs_horiz_4);                   \
          lhs_pf5 = lhs(lhs_vert, lhs_horiz_5);                   \
        } else if (lhs_horiz_4 < k_size) {                        \
          lhs_pf0 = lhs(lhs_vert, lhs_horiz_0);                   \
          lhs_pf1 = lhs(lhs_vert, lhs_horiz_1);                   \
          lhs_pf2 = lhs(lhs_vert, lhs_horiz_2);                   \
          lhs_pf3 = lhs(lhs_vert, lhs_horiz_3);                   \
          lhs_pf4 = lhs(lhs_vert, lhs_horiz_4);                   \
        } else if (lhs_horiz_3 < k_size) {                        \
          lhs_pf0 = lhs(lhs_vert, lhs_horiz_0);                   \
          lhs_pf1 = lhs(lhs_vert, lhs_horiz_1);                   \
          lhs_pf2 = lhs(lhs_vert, lhs_horiz_2);                   \
          lhs_pf3 = lhs(lhs_vert, lhs_horiz_3);                   \
        } else if (lhs_horiz_2 < k_size) {                        \
          lhs_pf0 = lhs(lhs_vert, lhs_horiz_0);                   \
          lhs_pf1 = lhs(lhs_vert, lhs_horiz_1);                   \
          lhs_pf2 = lhs(lhs_vert, lhs_horiz_2);                   \
        } else if (lhs_horiz_1 < k_size) {                        \
          lhs_pf0 = lhs(lhs_vert, lhs_horiz_0);                   \
          lhs_pf1 = lhs(lhs_vert, lhs_horiz_1);                   \
        } else if (lhs_horiz_0 < k_size) {                        \
          lhs_pf0 = lhs(lhs_vert, lhs_horiz_0);                   \
        }                                                         \
      }                                                           \
                                                                  \
      const Index rhs_vert = base_k + load_idx_vert;              \
      if (!needs_edge_check || rhs_vert < k_size) {               \
        const Index rhs_horiz_0 = base_n + threadIdx.z + 0 * 8;   \
        const Index rhs_horiz_1 = base_n + threadIdx.z + 1 * 8;   \
        const Index rhs_horiz_2 = base_n + threadIdx.z + 2 * 8;   \
        const Index rhs_horiz_3 = base_n + threadIdx.z + 3 * 8;   \
        const Index rhs_horiz_4 = base_n + threadIdx.z + 4 * 8;   \
        const Index rhs_horiz_5 = base_n + threadIdx.z + 5 * 8;   \
        const Index rhs_horiz_6 = base_n + threadIdx.z + 6 * 8;   \
        const Index rhs_horiz_7 = base_n + threadIdx.z + 7 * 8;   \
                                                                  \
        if (rhs_horiz_7 < n_size) {                               \
          rhs_pf0 = rhs(rhs_vert, rhs_horiz_0);                   \
          rhs_pf1 = rhs(rhs_vert, rhs_horiz_1);                   \
          rhs_pf2 = rhs(rhs_vert, rhs_horiz_2);                   \
          rhs_pf3 = rhs(rhs_vert, rhs_horiz_3);                   \
          rhs_pf4 = rhs(rhs_vert, rhs_horiz_4);                   \
          rhs_pf5 = rhs(rhs_vert, rhs_horiz_5);                   \
          rhs_pf6 = rhs(rhs_vert, rhs_horiz_6);                   \
          rhs_pf7 = rhs(rhs_vert, rhs_horiz_7);                   \
        } else if (rhs_horiz_6 < n_size) {                        \
          rhs_pf0 = rhs(rhs_vert, rhs_horiz_0);                   \
          rhs_pf1 = rhs(rhs_vert, rhs_horiz_1);                   \
          rhs_pf2 = rhs(rhs_vert, rhs_horiz_2);                   \
          rhs_pf3 = rhs(rhs_vert, rhs_horiz_3);                   \
          rhs_pf4 = rhs(rhs_vert, rhs_horiz_4);                   \
          rhs_pf5 = rhs(rhs_vert, rhs_horiz_5);                   \
          rhs_pf6 = rhs(rhs_vert, rhs_horiz_6);                   \
        } else if (rhs_horiz_5 < n_size) {                        \
          rhs_pf0 = rhs(rhs_vert, rhs_horiz_0);                   \
          rhs_pf1 = rhs(rhs_vert, rhs_horiz_1);                   \
          rhs_pf2 = rhs(rhs_vert, rhs_horiz_2);                   \
          rhs_pf3 = rhs(rhs_vert, rhs_horiz_3);                   \
          rhs_pf4 = rhs(rhs_vert, rhs_horiz_4);                   \
          rhs_pf5 = rhs(rhs_vert, rhs_horiz_5);                   \
        } else if (rhs_horiz_4 < n_size) {                        \
          rhs_pf0 = rhs(rhs_vert, rhs_horiz_0);                   \
          rhs_pf1 = rhs(rhs_vert, rhs_horiz_1);                   \
          rhs_pf2 = rhs(rhs_vert, rhs_horiz_2);                   \
          rhs_pf3 = rhs(rhs_vert, rhs_horiz_3);                   \
          rhs_pf4 = rhs(rhs_vert, rhs_horiz_4);                   \
        } else if (rhs_horiz_3 < n_size) {                        \
          rhs_pf0 = rhs(rhs_vert, rhs_horiz_0);                   \
          rhs_pf1 = rhs(rhs_vert, rhs_horiz_1);                   \
          rhs_pf2 = rhs(rhs_vert, rhs_horiz_2);                   \
          rhs_pf3 = rhs(rhs_vert, rhs_horiz_3);                   \
        } else if (rhs_horiz_2 < n_size) {                        \
          rhs_pf0 = rhs(rhs_vert, rhs_horiz_0);                   \
          rhs_pf1 = rhs(rhs_vert, rhs_horiz_1);                   \
          rhs_pf2 = rhs(rhs_vert, rhs_horiz_2);                   \
        } else if (rhs_horiz_1 < n_size) {                        \
          rhs_pf0 = rhs(rhs_vert, rhs_horiz_0);                   \
          rhs_pf1 = rhs(rhs_vert, rhs_horiz_1);                   \
        } else if (rhs_horiz_0 < n_size) {                        \
          rhs_pf0 = rhs(rhs_vert, rhs_horiz_0);                   \
        }                                                         \
      }                                                           \
    }                                                             \

#define writeRegToShmem(_)                      \
  lhs_shmem[lhs_store_idx_0] = lhs_pf0;         \
  rhs_shmem[rhs_store_idx_0] = rhs_pf0;         \
                                                \
  lhs_shmem[lhs_store_idx_1] = lhs_pf1;         \
  rhs_shmem[rhs_store_idx_1] = rhs_pf1;         \
                                                \
  lhs_shmem[lhs_store_idx_2] = lhs_pf2;         \
  rhs_shmem[rhs_store_idx_2] = rhs_pf2;         \
                                                \
  lhs_shmem[lhs_store_idx_3] = lhs_pf3;         \
  rhs_shmem[rhs_store_idx_3] = rhs_pf3;         \
                                                \
  lhs_shmem[lhs_store_idx_4] = lhs_pf4;         \
  rhs_shmem[rhs_store_idx_4] = rhs_pf4;         \
                                                \
  lhs_shmem[lhs_store_idx_5] = lhs_pf5;         \
  rhs_shmem[rhs_store_idx_5] = rhs_pf5;         \
                                                \
  lhs_shmem[lhs_store_idx_6] = lhs_pf6;         \
  rhs_shmem[rhs_store_idx_6] = rhs_pf6;         \
                                                \
  lhs_shmem[lhs_store_idx_7] = lhs_pf7;         \
  rhs_shmem[rhs_store_idx_7] = rhs_pf7;         \

  // declare and initialize result array
#define res(i, j) _res_##i##j
#define initResultRow(i)                        \
  Scalar res(i, 0) = Scalar(0);                 \
  Scalar res(i, 1) = Scalar(0);                 \
  Scalar res(i, 2) = Scalar(0);                 \
  Scalar res(i, 3) = Scalar(0);                 \
  Scalar res(i, 4) = Scalar(0);                 \
  Scalar res(i, 5) = Scalar(0);                 \
  Scalar res(i, 6) = Scalar(0);                 \
  Scalar res(i, 7) = Scalar(0);                 \

  initResultRow(0);
  initResultRow(1);
  initResultRow(2);
  initResultRow(3);
  initResultRow(4);
  initResultRow(5);
  initResultRow(6);
  initResultRow(7);
#undef initResultRow

  for (Index base_k = 0; base_k < k_size; base_k += 64) {
    // wait for previous iteration to finish with shmem. Despite common sense,
    // the code is a bit faster with this here then at bottom of loop
    __syncthreads();

    prefetchIntoRegisters(base_k);
    writeRegToShmem();

    #undef prefetchIntoRegisters
    #undef writeRegToShmem

    // wait for shared mem packing to be done before starting computation
    __syncthreads();

    // compute 8x8 matrix product by outer product. This involves packing one column
    // of LHS and one row of RHS into registers (takes 16 registers).

#define lcol(i) _lcol##i
    Scalar lcol(0);
    Scalar lcol(1);
    Scalar lcol(2);
    Scalar lcol(3);
    Scalar lcol(4);
    Scalar lcol(5);
    Scalar lcol(6);
    Scalar lcol(7);

#define rrow(j) _rrow##j
    Scalar rrow(0);
    Scalar rrow(1);
    Scalar rrow(2);
    Scalar rrow(3);
    Scalar rrow(4);
    Scalar rrow(5);
    Scalar rrow(6);
    Scalar rrow(7);

    // Now x corresponds to k, y to m, and z to n
    const volatile Scalar* lhs_block = &lhs_shmem[threadIdx.x + 9 * threadIdx.y];
    const volatile Scalar* rhs_block = &rhs_shmem[threadIdx.x + 8 * threadIdx.z];

#define lhs_element(i, j) lhs_block[72 * ((i) + 8 * (j))]
#define rhs_element(i, j) rhs_block[72 * ((i) + 8 * (j))]

#define loadData(i, j)                          \
                                    lcol(0) = lhs_element(0, j);               \
                                    rrow(0) = rhs_element(i, 0);               \
                                    lcol(1) = lhs_element(1, j);               \
                                    rrow(1) = rhs_element(i, 1);               \
                                    lcol(2) = lhs_element(2, j);               \
                                    rrow(2) = rhs_element(i, 2);               \
                                    lcol(3) = lhs_element(3, j);               \
                                    rrow(3) = rhs_element(i, 3);               \
                                    lcol(4) = lhs_element(4, j);               \
                                    rrow(4) = rhs_element(i, 4);               \
                                    lcol(5) = lhs_element(5, j);               \
                                    rrow(5) = rhs_element(i, 5);               \
                                    lcol(6) = lhs_element(6, j);               \
                                    rrow(6) = rhs_element(i, 6);               \
                                    lcol(7) = lhs_element(7, j);               \
                                    rrow(7) = rhs_element(i, 7);               \

#define computeCol(j)                           \
                                    res(0, j) += lcol(0) * rrow(j);             \
                                    res(1, j) += lcol(1) * rrow(j);             \
                                    res(2, j) += lcol(2) * rrow(j);             \
                                    res(3, j) += lcol(3) * rrow(j);             \
                                    res(4, j) += lcol(4) * rrow(j);             \
                                    res(5, j) += lcol(5) * rrow(j);             \
                                    res(6, j) += lcol(6) * rrow(j);             \
                                    res(7, j) += lcol(7) * rrow(j);             \

#define computePass(i)                          \
                                    loadData(i, i);                             \
                                                \
                                    computeCol(0);                              \
                                    computeCol(1);                              \
                                    computeCol(2);                              \
                                    computeCol(3);                              \
                                    computeCol(4);                              \
                                    computeCol(5);                              \
                                    computeCol(6);                              \
                                    computeCol(7);                              \

                                    computePass(0);
                                    computePass(1);
                                    computePass(2);
                                    computePass(3);
                                    computePass(4);
                                    computePass(5);
                                    computePass(6);
                                    computePass(7);

#undef lcol
#undef rrow
#undef lhs_element
#undef rhs_element
#undef loadData
#undef computeCol
#undef computePass
                                    } // end loop over k

    // we've now iterated over all of the large (ie width 64) k blocks and
    // accumulated results in registers. At this point thread (x, y, z) contains
    // the sum across all big k blocks of the product of little k block of index (x, y)
    // with block of index (y, z). To compute the final output, we need to reduce
    // the 8 threads over y by summation.
#define shuffleInc(i, j, mask) res(i, j) += __shfl_xor(res(i, j), mask)

#define reduceRow(i, mask)                      \
    shuffleInc(i, 0, mask);                       \
    shuffleInc(i, 1, mask);                       \
    shuffleInc(i, 2, mask);                       \
    shuffleInc(i, 3, mask);                       \
    shuffleInc(i, 4, mask);                       \
    shuffleInc(i, 5, mask);                       \
    shuffleInc(i, 6, mask);                       \
    shuffleInc(i, 7, mask);                       \

#define reduceMatrix(mask)                      \
    reduceRow(0, mask);                           \
    reduceRow(1, mask);                           \
    reduceRow(2, mask);                           \
    reduceRow(3, mask);                           \
    reduceRow(4, mask);                           \
    reduceRow(5, mask);                           \
    reduceRow(6, mask);                           \
    reduceRow(7, mask);                           \

    // actually perform the reduction, now each thread of index (_, y, z)
    // contains the correct values in its registers that belong in the output
    // block
    reduceMatrix(1);
    reduceMatrix(2);
    reduceMatrix(4);

#undef shuffleInc
#undef reduceRow
#undef reduceMatrix

    // now we need to copy the 64 values into main memory. We can't split work
    // among threads because all variables are in registers. There's 2 ways
    // to do this:
    // (1) have 1 thread do 64 writes from registers into global memory
    // (2) have 1 thread do 64 writes into shared memory, and then 8 threads
    //     each do 8 writes into global memory. We can just overwrite the shared
    //     memory from the problem we just solved.
    // (2) is slightly faster than (1) due to less branching and more ILP

    // TODO: won't yield much gain, but could just use currently unused shared mem
    //       and then we won't have to sync
    // wait for shared mem to be out of use
    __syncthreads();

#define writeResultShmem(i, j)                                          \
    lhs_shmem[i + 8 * threadIdx.y + 64 * threadIdx.z + 512 * j] = res(i, j); \

#define writeRow(i)                             \
    writeResultShmem(i, 0);                       \
    writeResultShmem(i, 1);                       \
    writeResultShmem(i, 2);                       \
    writeResultShmem(i, 3);                       \
    writeResultShmem(i, 4);                       \
    writeResultShmem(i, 5);                       \
    writeResultShmem(i, 6);                       \
    writeResultShmem(i, 7);                       \

    if (threadIdx.x == 0) {
      writeRow(0);
      writeRow(1);
      writeRow(2);
      writeRow(3);
      writeRow(4);
      writeRow(5);
      writeRow(6);
      writeRow(7);
    }
#undef writeResultShmem
#undef writeRow

    const int max_i_write = (min)((int)((m_size - base_m - threadIdx.y + 7) / 8), 8);
    const int max_j_write = (min)((int)((n_size - base_n - threadIdx.z + 7) / 8), 8);

    if (threadIdx.x < max_i_write) {
      if (max_j_write == 8) {
        Scalar val0 = lhs_shmem[threadIdx.x + 8 * threadIdx.y + 64 * threadIdx.z + 512 * 0];
        Scalar val1 = lhs_shmem[threadIdx.x + 8 * threadIdx.y + 64 * threadIdx.z + 512 * 1];
        Scalar val2 = lhs_shmem[threadIdx.x + 8 * threadIdx.y + 64 * threadIdx.z + 512 * 2];
        Scalar val3 = lhs_shmem[threadIdx.x + 8 * threadIdx.y + 64 * threadIdx.z + 512 * 3];
        Scalar val4 = lhs_shmem[threadIdx.x + 8 * threadIdx.y + 64 * threadIdx.z + 512 * 4];
        Scalar val5 = lhs_shmem[threadIdx.x + 8 * threadIdx.y + 64 * threadIdx.z + 512 * 5];
        Scalar val6 = lhs_shmem[threadIdx.x + 8 * threadIdx.y + 64 * threadIdx.z + 512 * 6];
        Scalar val7 = lhs_shmem[threadIdx.x + 8 * threadIdx.y + 64 * threadIdx.z + 512 * 7];

        output(base_m + threadIdx.y + 8 * threadIdx.x, base_n + threadIdx.z + 8 * 0) = val0;
        output(base_m + threadIdx.y + 8 * threadIdx.x, base_n + threadIdx.z + 8 * 1) = val1;
        output(base_m + threadIdx.y + 8 * threadIdx.x, base_n + threadIdx.z + 8 * 2) = val2;
        output(base_m + threadIdx.y + 8 * threadIdx.x, base_n + threadIdx.z + 8 * 3) = val3;
        output(base_m + threadIdx.y + 8 * threadIdx.x, base_n + threadIdx.z + 8 * 4) = val4;
        output(base_m + threadIdx.y + 8 * threadIdx.x, base_n + threadIdx.z + 8 * 5) = val5;
        output(base_m + threadIdx.y + 8 * threadIdx.x, base_n + threadIdx.z + 8 * 6) = val6;
        output(base_m + threadIdx.y + 8 * threadIdx.x, base_n + threadIdx.z + 8 * 7) = val7;
      } else {
#pragma unroll 7
        for (int j = 0; j < max_j_write; j++) {
          Scalar val = lhs_shmem[threadIdx.x + 8 * threadIdx.y + 64 * threadIdx.z + 512 * j];
          output(base_m + threadIdx.y + 8 * threadIdx.x, base_n + threadIdx.z + 8 * j) = val;
        }
      }
    }
#undef res
  }


  template<typename Scalar, typename Index, typename LhsMapper,
           typename RhsMapper, typename OutputMapper>
__global__ void
__launch_bounds__(512)
      EigenContractionKernel(const LhsMapper lhs, const RhsMapper rhs,
                             const OutputMapper output,
                             const Index m_size, const Index n_size, const Index k_size) {
    __shared__ volatile Scalar lhs_shmem[72 * 64];
    __shared__ volatile Scalar rhs_shmem[72 * 64];

    const Index m_block_idx = blockIdx.x;
    const Index n_block_idx = blockIdx.y;

    const Index base_m = 64 * m_block_idx;
    const Index base_n = 64 * n_block_idx;

    if (base_m + 63 < m_size && base_n + 63 < n_size) {
      EigenContractionKernelInternal<Scalar, Index, LhsMapper, RhsMapper, OutputMapper, false>(lhs, rhs, output, lhs_shmem, rhs_shmem, m_size, n_size, k_size);
    } else {
      EigenContractionKernelInternal<Scalar, Index, LhsMapper, RhsMapper, OutputMapper, true>(lhs, rhs, output, lhs_shmem, rhs_shmem, m_size, n_size, k_size);
    }
  }



  template<typename Index, typename LhsMapper,
           typename RhsMapper, typename OutputMapper, bool needs_edge_check>
__device__ EIGEN_STRONG_INLINE void
      EigenFloatContractionKernelInternal(const LhsMapper lhs, const RhsMapper rhs,
                                          const OutputMapper output, float4* lhs_shmem4, float2* rhs_shmem2,
                                          const Index m_size, const Index n_size, const Index k_size) {
    typedef float Scalar;

    const Index m_block_idx = blockIdx.x;
    const Index n_block_idx = blockIdx.y;

    const Index base_m = 64 * m_block_idx;
    const Index base_n = 64 * n_block_idx;

    const Index lane = threadIdx.x + 8 * (threadIdx.y % 4);

    // prefetch registers
    float4 lhs_pf0;
    float4 lhs_pf1;

    float4 rhs_pf0;
    float4 rhs_pf1;

    // shared memory is formatted
    // (contract idx in block, nocontract idx in block, block idx)
    // where block idx is column major. This transposition limits the number of
    // bank conflicts when reading the LHS. The core idea is that since the contracting
    // index is shared by both sides, then the contracting index should be in threadIdx.x.

    // all of these indices assume float4 loading
    // this thread loads the float4 starting at this index, and then also loads
    // another float4 starting 32 columns to to the right
    const Index horiz_block_idx = threadIdx.z / 2;
    const Index vert_block_idx = threadIdx.x / 2 + 4 * (threadIdx.y % 2);
    const Index horiz_idx_in_block = threadIdx.y / 2 + 4 * (threadIdx.z % 2);
    const Index vert_idx_in_block = threadIdx.x % 2;

    // there's padding in both the LHS and RHS shared memory layouts. This padding
    // allows for 0 bank conflicts on all shmem stores and loads.
    // LHS padding: 1 float4 on each 8x8 block of floats
    // RHS padding: 1 float2 on each block, and 12 additional float2s between vertical blocks
    //              3 and 4

    // storage indices
    // lhs index with respect to float4s
  const Index lhs_store_idx_base =
      136 * horiz_block_idx +
      17 * vert_block_idx +
      8 * vert_idx_in_block +
      horiz_idx_in_block;

  // rhs index with respect to floats
  const Index rhs_store_idx_base =
      552 * horiz_block_idx +
      66 * vert_block_idx +
      32 * (horiz_idx_in_block / 4) + (horiz_idx_in_block % 4) +
      16 * vert_idx_in_block +
      ((vert_block_idx < 4) ? 0 : 24);

  const Index lhs_store_idx_0 = lhs_store_idx_base + 544 * 0;
  const Index lhs_store_idx_1 = lhs_store_idx_base + 544 * 1;

  const Index rhs_store_idx_0 = (rhs_store_idx_base / 2) + ((lane < 16) ? 0 : 4);
  const Index rhs_store_idx_1 = rhs_store_idx_0 + 2;
  const Index rhs_store_idx_2 = rhs_store_idx_0 + 1104;
  const Index rhs_store_idx_3 = rhs_store_idx_1 + 1104;

  // The below diagrams show which shmem index (with respect to floats) each element
  // in an 8x8 input block gets packed into:
  // LHS:
  // 0  4  8  12  16  20  24  28
  // 1  5  9  13  17  21  25  29
  // 2  6  10 14  18  22  26  30
  // 3  7  11 15  19  23  27  31
  // 32 36 40 44  48  52  56  60
  // ... (pack as 2 rows of float4 indexed row major, each float4 is vertical)
  //
  // RHS:
  // 0  1  2  3  32  33  34  35
  // 4  5  6  7  36  37  38  39
  // ... (pack as 2 cols of float4 indexed col major, each float4 is horizontal)

  // Each thread in a warp loads 2 float4s. This happens in 2 instructions. On each of these
  // instruction, the warp loads 2 columns (2 cols * 64 elements / col = 128 elements = 32 threads
  // * 4 elements/thread). For the LHS, we're able to store the loaded float4 directly into
  // shmem (using a 128 bit store instruction). For the RHS, we need to transpose the data.
  // This is done with warp shuffles. Furthermore, we only use 64 bit stores for the RHS, because
  // 64 bits is only 2 columns (which is all we load in a warp), and the padding for the RHS
  // doesn't meet 64 bit alignment requirements (namely, the 4 consecutive floats that we want
  // to load on the RHS are 8 byte aligned, not 16 byte aligned, which is required for float4).

  const Index load_idx_vert = 4 * (threadIdx.x + 8 * (threadIdx.y % 2));
  const Index load_idx_horiz = (threadIdx.y / 2) + 4 * threadIdx.z;

  const Index lhs_vert = base_m + load_idx_vert;
  const Index rhs_horiz_0 = base_n + load_idx_horiz;
  const Index rhs_horiz_1 = base_n + load_idx_horiz + 32;

#define prefetchIntoRegisters(base_k)                                   \
  {                                                                     \
      lhs_pf0 = internal::pset1<float4>(0);                               \
      lhs_pf1 = internal::pset1<float4>(0);                               \
                                                                        \
      rhs_pf0 = internal::pset1<float4>(0);                               \
      rhs_pf1 = internal::pset1<float4>(0);                               \
                                                                        \
      const Index lhs_horiz_0 = base_k + load_idx_horiz;                  \
      const Index lhs_horiz_1 = base_k + load_idx_horiz + 32;             \
      if (!needs_edge_check || lhs_vert + 3 < m_size) {                   \
        if (lhs_horiz_1 < k_size) {                                       \
          lhs_pf0 = lhs.loadPacket(lhs_vert, lhs_horiz_0);                \
          lhs_pf1 = lhs.loadPacket(lhs_vert, lhs_horiz_1);                \
        } else if (lhs_horiz_0 < k_size) {                                \
          lhs_pf0 = lhs.loadPacket(lhs_vert, lhs_horiz_0);                \
        }                                                                 \
      } else if (lhs_vert + 2 < m_size) {                                 \
        if (lhs_horiz_1 < k_size) {                                       \
          lhs_pf0.x = lhs(lhs_vert + 0, lhs_horiz_0);                     \
          lhs_pf0.y = lhs(lhs_vert + 1, lhs_horiz_0);                     \
          lhs_pf0.z = lhs(lhs_vert + 2, lhs_horiz_0);                     \
                                                                        \
          lhs_pf1.x = lhs(lhs_vert + 0, lhs_horiz_1);                     \
          lhs_pf1.y = lhs(lhs_vert + 1, lhs_horiz_1);                     \
          lhs_pf1.z = lhs(lhs_vert + 2, lhs_horiz_1);                     \
        } else if (lhs_horiz_0 < k_size) {                                \
          lhs_pf0.x = lhs(lhs_vert + 0, lhs_horiz_0);                     \
          lhs_pf0.y = lhs(lhs_vert + 1, lhs_horiz_0);                     \
          lhs_pf0.z = lhs(lhs_vert + 2, lhs_horiz_0);                     \
        }                                                                 \
      } else if (lhs_vert + 1 < m_size) {                                 \
        if (lhs_horiz_1 < k_size) {                                       \
          lhs_pf0.x = lhs(lhs_vert + 0, lhs_horiz_0);                     \
          lhs_pf0.y = lhs(lhs_vert + 1, lhs_horiz_0);                     \
                                                                        \
          lhs_pf1.x = lhs(lhs_vert + 0, lhs_horiz_1);                     \
          lhs_pf1.y = lhs(lhs_vert + 1, lhs_horiz_1);                     \
        } else if (lhs_horiz_0 < k_size) {                                \
          lhs_pf0.x = lhs(lhs_vert + 0, lhs_horiz_0);                     \
          lhs_pf0.y = lhs(lhs_vert + 1, lhs_horiz_0);                     \
        }                                                                 \
      } else if (lhs_vert < m_size) {                                     \
        if (lhs_horiz_1 < k_size) {                                       \
          lhs_pf0.x = lhs(lhs_vert + 0, lhs_horiz_0);                     \
          lhs_pf1.x = lhs(lhs_vert + 0, lhs_horiz_1);                     \
        } else if (lhs_horiz_0 < k_size) {                                \
          lhs_pf0.x = lhs(lhs_vert + 0, lhs_horiz_0);                     \
        }                                                                 \
}                                                                   \
                                                                        \
      const Index rhs_vert = base_k + load_idx_vert;                      \
      if (rhs_vert + 3 < k_size) {                                        \
        if (!needs_edge_check || rhs_horiz_1 < n_size) {                  \
          rhs_pf0 = rhs.loadPacket(rhs_vert, rhs_horiz_0);                \
          rhs_pf1 = rhs.loadPacket(rhs_vert, rhs_horiz_1);                \
        } else if (rhs_horiz_0 < n_size) {                                \
          rhs_pf0 = rhs.loadPacket(rhs_vert, rhs_horiz_0);                \
        }                                                                 \
      } else if (rhs_vert + 2 < k_size) {                                 \
        if (!needs_edge_check || rhs_horiz_1 < n_size) {                  \
          rhs_pf0.x = rhs(rhs_vert + 0, rhs_horiz_0);                     \
          rhs_pf0.y = rhs(rhs_vert + 1, rhs_horiz_0);                     \
          rhs_pf0.z = rhs(rhs_vert + 2, rhs_horiz_0);                     \
                                                                        \
          rhs_pf1.x = rhs(rhs_vert + 0, rhs_horiz_1);                     \
          rhs_pf1.y = rhs(rhs_vert + 1, rhs_horiz_1);                     \
          rhs_pf1.z = rhs(rhs_vert + 2, rhs_horiz_1);                     \
        } else if (rhs_horiz_0 < n_size) {                                \
          rhs_pf0.x = rhs(rhs_vert + 0, rhs_horiz_0);                     \
          rhs_pf0.y = rhs(rhs_vert + 1, rhs_horiz_0);                     \
          rhs_pf0.z = rhs(rhs_vert + 2, rhs_horiz_0);                     \
        }                                                                 \
      } else if (rhs_vert + 1 < k_size) {                                 \
        if (!needs_edge_check || rhs_horiz_1 < n_size) {                  \
          rhs_pf0.x = rhs(rhs_vert + 0, rhs_horiz_0);                     \
          rhs_pf0.y = rhs(rhs_vert + 1, rhs_horiz_0);                     \
                                                                        \
          rhs_pf1.x = rhs(rhs_vert + 0, rhs_horiz_1);                     \
          rhs_pf1.y = rhs(rhs_vert + 1, rhs_horiz_1);                     \
        } else if (rhs_horiz_0 < n_size) {                                \
          rhs_pf0.x = rhs(rhs_vert + 0, rhs_horiz_0);                     \
          rhs_pf0.y = rhs(rhs_vert + 1, rhs_horiz_0);                     \
        }                                                                 \
        } else if (rhs_vert < k_size) {                                     \
        if (!needs_edge_check || rhs_horiz_1 < n_size) {                  \
          rhs_pf0.x = rhs(rhs_vert + 0, rhs_horiz_0);                     \
          rhs_pf1.x = rhs(rhs_vert + 0, rhs_horiz_1);                     \
        } else if (rhs_horiz_0 < n_size) {                                \
          rhs_pf0.x = rhs(rhs_vert + 0, rhs_horiz_0);                     \
        }                                                                 \
}                                                                   \
                                                                        \
      float swap_val0 = (lane < 16) ? rhs_pf0.z : rhs_pf0.x;              \
      float swap_val1 = (lane < 16) ? rhs_pf0.w : rhs_pf0.y;              \
      float swap_val2 = (lane < 16) ? rhs_pf1.z : rhs_pf1.x;              \
      float swap_val3 = (lane < 16) ? rhs_pf1.w : rhs_pf1.y;              \
                                                                        \
      swap_val0 = __shfl_xor(swap_val0, 16);                              \
      swap_val1 = __shfl_xor(swap_val1, 16);                              \
      swap_val2 = __shfl_xor(swap_val2, 16);                              \
      swap_val3 = __shfl_xor(swap_val3, 16);                              \
                                                                        \
      if (lane < 16) {                                                    \
        rhs_pf0.z = swap_val0;                                            \
        rhs_pf0.w = swap_val1;                                            \
        rhs_pf1.z = swap_val2;                                            \
        rhs_pf1.w = swap_val3;                                            \
      } else {                                                            \
        rhs_pf0.x = swap_val0;                                            \
        rhs_pf0.y = swap_val1;                                            \
        rhs_pf1.x = swap_val2;                                            \
        rhs_pf1.y = swap_val3;                                            \
      }                                                                   \
}                                                                     \


#define writeRegToShmem(_)                                              \
  lhs_shmem4[lhs_store_idx_0] = lhs_pf0;                                \
                                                                        \
  rhs_shmem2[rhs_store_idx_0] = make_float2(rhs_pf0.x, rhs_pf0.z);      \
  rhs_shmem2[rhs_store_idx_1] = make_float2(rhs_pf0.y, rhs_pf0.w);      \
                                                                        \
  lhs_shmem4[lhs_store_idx_1] = lhs_pf1;                                \
                                                                        \
  rhs_shmem2[rhs_store_idx_2] = make_float2(rhs_pf1.x, rhs_pf1.z);      \
  rhs_shmem2[rhs_store_idx_3] = make_float2(rhs_pf1.y, rhs_pf1.w);      \

  // declare and initialize result array
#define res(i, j) _res_##i##j
#define initResultRow(i)                        \
  Scalar res(i, 0) = Scalar(0);                 \
  Scalar res(i, 1) = Scalar(0);                 \
  Scalar res(i, 2) = Scalar(0);                 \
  Scalar res(i, 3) = Scalar(0);                 \
  Scalar res(i, 4) = Scalar(0);                 \
  Scalar res(i, 5) = Scalar(0);                 \
  Scalar res(i, 6) = Scalar(0);                 \
  Scalar res(i, 7) = Scalar(0);                 \

  initResultRow(0);
  initResultRow(1);
  initResultRow(2);
  initResultRow(3);
  initResultRow(4);
  initResultRow(5);
  initResultRow(6);
  initResultRow(7);
#undef initResultRow

  for (Index base_k = 0; base_k < k_size; base_k += 64) {
    // wait for previous iteration to finish with shmem. Despite common sense,
    // the code is a bit faster with this here then at bottom of loop
    __syncthreads();

    prefetchIntoRegisters(base_k);
    writeRegToShmem();

#undef prefetchIntoRegisters
#undef writeRegoToShmem

    // wait for shared mem packing to be done before starting computation
    __syncthreads();

    // compute 8x8 matrix product by outer product. This involves packing one column
    // of LHS and one row of RHS into registers (takes 16 registers).

    float4 _lcol0;
    float4 _lcol1;
    float2 _rrow0;
    float2 _rrow1;
    float2 _rrow2;
    float2 _rrow3;

#define lcol0 _lcol0.x
#define lcol1 _lcol0.y
#define lcol2 _lcol0.z
#define lcol3 _lcol0.w
#define lcol4 _lcol1.x
#define lcol5 _lcol1.y
#define lcol6 _lcol1.z
#define lcol7 _lcol1.w
#define rrow0 _rrow0.x
#define rrow1 _rrow0.y
#define rrow2 _rrow1.x
#define rrow3 _rrow1.y
#define rrow4 _rrow2.x
#define rrow5 _rrow2.y
#define rrow6 _rrow3.x
#define rrow7 _rrow3.y

    // Now x corresponds to k, y to m, and z to n
    const float4* lhs_block = &lhs_shmem4[threadIdx.x + 8 * (threadIdx.y % 2) + 17 * (threadIdx.y / 2)];
    const float2* rhs_block = &rhs_shmem2[2 * threadIdx.x + 16 * (threadIdx.z % 2) + 276 * (threadIdx.z / 2)];

#define lhs_element(i, k) lhs_block[68 * i + 136 * k]
#define rhs_element(k, j) rhs_block[33 * k + 1104 * j + ((k < 4) ? 0 : 12)]

#define loadData(i)                             \
                                    _lcol0 = lhs_element(0, i);                 \
                                    _rrow0 = rhs_element(i, 0);                 \
                                    _rrow1 = *(&(rhs_element(i, 0)) + 1);       \
                                    _lcol1 = lhs_element(1, i);                 \
                                    _rrow2 = rhs_element(i, 1);                 \
                                    _rrow3 = *(&(rhs_element(i, 1)) + 1);       \

#define computeCol(j)                         \
                                    res(0, j) += lcol0 * rrow##j;             \
                                    res(1, j) += lcol1 * rrow##j;             \
                                    res(2, j) += lcol2 * rrow##j;             \
                                    res(3, j) += lcol3 * rrow##j;             \
                                    res(4, j) += lcol4 * rrow##j;             \
                                    res(5, j) += lcol5 * rrow##j;             \
                                    res(6, j) += lcol6 * rrow##j;             \
                                    res(7, j) += lcol7 * rrow##j;             \

#define computePass(i)                          \
                                    loadData(i);                                \
                                                \
                                    computeCol(0);                              \
                                    computeCol(1);                              \
                                    computeCol(2);                              \
                                    computeCol(3);                              \
                                    computeCol(4);                              \
                                    computeCol(5);                              \
                                    computeCol(6);                              \
                                    computeCol(7);                              \

                                    computePass(0);
                                    computePass(1);
                                    computePass(2);
                                    computePass(3);
                                    computePass(4);
                                    computePass(5);
                                    computePass(6);
                                    computePass(7);

#undef lcol0
#undef lcol1
#undef lcol2
#undef lcol3
#undef lcol4
#undef lcol5
#undef lcol6
#undef lcol7
#undef rrow0
#undef rrow1
#undef rrow2
#undef rrow3
#undef rrow4
#undef rrow5
#undef rrow6
#undef rrow7

#undef computePass
#undef computeCol
#undef loadData
#undef lhs_element
#undef rhs_element

                                    } // end loop over k

    // we've now iterated over all of the large (ie width 64) k blocks and
    // accumulated results in registers. At this point thread (x, y, z) contains
    // the sum across all big k blocks of the product of little k block of index (x, y)
    // with block of index (y, z). To compute the final output, we need to reduce
    // the 8 threads over y by summation.
#define shuffleInc(i, j, mask) res(i, j) += __shfl_xor(res(i, j), mask)

#define reduceRow(i, mask)                      \
    shuffleInc(i, 0, mask);                       \
    shuffleInc(i, 1, mask);                       \
    shuffleInc(i, 2, mask);                       \
    shuffleInc(i, 3, mask);                       \
    shuffleInc(i, 4, mask);                       \
    shuffleInc(i, 5, mask);                       \
    shuffleInc(i, 6, mask);                       \
    shuffleInc(i, 7, mask);                       \

#define reduceMatrix(mask)                      \
    reduceRow(0, mask);                           \
    reduceRow(1, mask);                           \
    reduceRow(2, mask);                           \
    reduceRow(3, mask);                           \
    reduceRow(4, mask);                           \
    reduceRow(5, mask);                           \
    reduceRow(6, mask);                           \
    reduceRow(7, mask);                           \

    // actually perform the reduction, now each thread of index (_, y, z)
    // contains the correct values in its registers that belong in the output
    // block
    reduceMatrix(1);
    reduceMatrix(2);
    reduceMatrix(4);

#undef shuffleInc
#undef reduceRow
#undef reduceMatrix

    // now we need to copy the 64 values into main memory. We can't split work
    // among threads because all variables are in registers. There's 2 ways
    // to do this:
    // (1) have 1 thread do 64 writes from registers into global memory
    // (2) have 1 thread do 64 writes into shared memory, and then 8 threads
    //     each do 8 writes into global memory. We can just overwrite the shared
    //     memory from the problem we just solved.
    // (3) Copies the values into new registers using conditional logic.

#define makeAssignments(i)                      \
    val0 = res(i, 0);                             \
    val1 = res(i, 1);                             \
    val2 = res(i, 2);                             \
    val3 = res(i, 3);                             \
    val4 = res(i, 4);                             \
    val5 = res(i, 5);                             \
    val6 = res(i, 6);                             \
    val7 = res(i, 7);                             \

    Scalar val0;
    Scalar val1;
    Scalar val2;
    Scalar val3;
    Scalar val4;
    Scalar val5;
    Scalar val6;
    Scalar val7;

    switch (threadIdx.x) {
    case 0:
      makeAssignments(0);
      break;
    case 1:
      makeAssignments(1);
      break;
    case 2:
      makeAssignments(2);
      break;
    case 3:
      makeAssignments(3);
      break;
    case 4:
      makeAssignments(4);
      break;
    case 5:
      makeAssignments(5);
      break;
    case 6:
      makeAssignments(6);
      break;
    case 7:
      makeAssignments(7);
      break;
  }

#undef res

    const Index vert_base = base_m + 4 * threadIdx.y + (threadIdx.x % 4) + 32 * (threadIdx.x / 4);
    const Index horiz_base = base_n + 4 * threadIdx.z;

    if (!needs_edge_check || vert_base < m_size) {
    if (!needs_edge_check || horiz_base + 35 < n_size) {
    output(vert_base, horiz_base + 0) = val0;
    output(vert_base, horiz_base + 1) = val1;
    output(vert_base, horiz_base + 2) = val2;
    output(vert_base, horiz_base + 3) = val3;
    output(vert_base, horiz_base + 32) = val4;
    output(vert_base, horiz_base + 33) = val5;
    output(vert_base, horiz_base + 34) = val6;
    output(vert_base, horiz_base + 35) = val7;
  } else if (horiz_base + 34 < n_size) {
    output(vert_base, horiz_base + 0) = val0;
    output(vert_base, horiz_base + 1) = val1;
    output(vert_base, horiz_base + 2) = val2;
    output(vert_base, horiz_base + 3) = val3;
    output(vert_base, horiz_base + 32) = val4;
    output(vert_base, horiz_base + 33) = val5;
    output(vert_base, horiz_base + 34) = val6;
  } else if (horiz_base + 33 < n_size) {
    output(vert_base, horiz_base + 0) = val0;
    output(vert_base, horiz_base + 1) = val1;
    output(vert_base, horiz_base + 2) = val2;
    output(vert_base, horiz_base + 3) = val3;
    output(vert_base, horiz_base + 32) = val4;
    output(vert_base, horiz_base + 33) = val5;
  } else if (horiz_base + 32 < n_size) {
    output(vert_base, horiz_base + 0) = val0;
    output(vert_base, horiz_base + 1) = val1;
    output(vert_base, horiz_base + 2) = val2;
    output(vert_base, horiz_base + 3) = val3;
    output(vert_base, horiz_base + 32) = val4;
  } else if (horiz_base + 3 < n_size) {
    output(vert_base, horiz_base + 0) = val0;
    output(vert_base, horiz_base + 1) = val1;
    output(vert_base, horiz_base + 2) = val2;
    output(vert_base, horiz_base + 3) = val3;
  } else if (horiz_base + 2 < n_size) {
    output(vert_base, horiz_base + 0) = val0;
    output(vert_base, horiz_base + 1) = val1;
    output(vert_base, horiz_base + 2) = val2;
  } else if (horiz_base + 1 < n_size) {
    output(vert_base, horiz_base + 0) = val0;
    output(vert_base, horiz_base + 1) = val1;
  } else if (horiz_base < n_size) {
    output(vert_base, horiz_base + 0) = val0;
  }
  }
  }


    template<typename Index, typename LhsMapper,
             typename RhsMapper, typename OutputMapper>
__global__ void
        __launch_bounds__(512)
        EigenFloatContractionKernel(const LhsMapper lhs, const RhsMapper rhs,
        const OutputMapper output,
        const Index m_size, const Index n_size, const Index k_size) {
    __shared__ float4 lhs_shmem[(68 * 64) / 4];
    __shared__ float2 rhs_shmem[((66 * 8 + 24) * 8) / 2];

    const Index m_block_idx = blockIdx.x;
    const Index n_block_idx = blockIdx.y;

    const Index base_m = 64 * m_block_idx;
    const Index base_n = 64 * n_block_idx;

    if (base_m + 63 < m_size && base_n + 63 < n_size) {
    EigenFloatContractionKernelInternal<Index, LhsMapper, RhsMapper, OutputMapper, false>(lhs, rhs, output, lhs_shmem, rhs_shmem, m_size, n_size, k_size);
  } else {
    EigenFloatContractionKernelInternal<Index, LhsMapper, RhsMapper, OutputMapper, true>(lhs, rhs, output, lhs_shmem, rhs_shmem, m_size, n_size, k_size);
  }
  }


    template<typename Indices, typename LeftArgType, typename RightArgType>
        struct TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType>, GpuDevice> :
                                public TensorContractionEvaluatorBase<TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType>, GpuDevice> > {

    typedef GpuDevice Device;

      typedef TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType>, Device> Self;
      typedef TensorContractionEvaluatorBase<Self> Base;

      typedef TensorContractionOp<Indices, LeftArgType, RightArgType> XprType;
      typedef typename internal::remove_const<typename XprType::Scalar>::type Scalar;
      typedef typename XprType::Packet Packet;
      typedef typename XprType::Index Index;
      typedef typename XprType::CoeffReturnType CoeffReturnType;
      typedef typename XprType::PacketReturnType PacketReturnType;

      typedef array<Index, TensorEvaluator<LeftArgType, Device>::Dimensions::count> left_dim_mapper_t;
      typedef array<Index, TensorEvaluator<RightArgType, Device>::Dimensions::count> right_dim_mapper_t;

      typedef array<Index, internal::array_size<Indices>::value> contract_t;
      typedef array<Index, TensorEvaluator<LeftArgType, Device>::Dimensions::count - internal::array_size<Indices>::value> left_nocontract_t;
      typedef array<Index, TensorEvaluator<RightArgType, Device>::Dimensions::count - internal::array_size<Indices>::value> right_nocontract_t;

      static const int NumDims = max_n_1<TensorEvaluator<LeftArgType, Device>::Dimensions::count + TensorEvaluator<RightArgType, Device>::Dimensions::count - 2 * internal::array_size<Indices>::value>::size;

      typedef DSizes<Index, NumDims> Dimensions;

      // typedefs needed in evalTo
      typedef typename internal::remove_const<typename LeftArgType::Scalar>::type LhsScalar;
      typedef typename internal::remove_const<typename RightArgType::Scalar>::type RhsScalar;

      typedef TensorEvaluator<LeftArgType, Device> LeftEvaluator;
      typedef TensorEvaluator<RightArgType, Device> RightEvaluator;

      typedef typename LeftEvaluator::Dimensions LeftDimensions;
      typedef typename RightEvaluator::Dimensions RightDimensions;

      EIGEN_DEVICE_FUNC TensorEvaluator(const XprType& op, const Device& device) :
          Base(op, device) {}

      // We need to redefine this method to make nvcc happy
      EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(Scalar* data) {
        this->m_leftImpl.evalSubExprsIfNeeded(NULL);
        this->m_rightImpl.evalSubExprsIfNeeded(NULL);
        if (data) {
          evalTo(data);
          return false;
        } else {
          this->m_result = static_cast<Scalar *>(this->m_device.allocate(this->dimensions().TotalSize() * sizeof(Scalar)));
          evalTo(this->m_result);
          return true;
        }
      }

      void evalTo(Scalar* buffer) const {
        if (this->m_lhs_inner_dim_contiguous) {
          if (this->m_rhs_inner_dim_contiguous) {
            if (this->m_rhs_inner_dim_reordered) {
              evalTyped<true, true, true, Unaligned>(buffer);
            }
            else {
              evalTyped<true, true, false, Unaligned>(buffer);
            }
          }
          else {
            if (this->m_rhs_inner_dim_reordered) {
              evalTyped<true, false, true, Unaligned>(buffer);
            }
            else {
              evalTyped<true, false, false, Unaligned>(buffer);
            }
          }
        }
        else {
          if (this->m_rhs_inner_dim_contiguous) {
            if (this->m_rhs_inner_dim_reordered) {
              evalTyped<false, true, true, Unaligned>(buffer);
            }
            else {
              evalTyped<false, true, false, Unaligned>(buffer);
            }
          }
          else {
            if (this->m_rhs_inner_dim_reordered) {
              evalTyped<false, false, true, Unaligned>(buffer);
            }
            else {
              evalTyped<false, false, false, Unaligned>(buffer);
            }
          }
        }
      }

      template <bool lhs_inner_dim_contiguous, bool rhs_inner_dim_contiguous, bool rhs_inner_dim_reordered, int Alignment>
      void evalTyped(Scalar* buffer) const {
        // columns in left side, rows in right side
        const Index k = this->m_k_size;

        // rows in left side
        const Index m = this->m_i_size;

        // columns in right side
        const Index n = this->m_j_size;

        // zero out the result buffer (which must be of size at least m * n * sizeof(Scalar)
        this->m_device.memset(buffer, 0, m * n * sizeof(Scalar));

        typedef internal::TensorContractionInputMapper<LhsScalar, Index, internal::Lhs,
                                                       LeftEvaluator, left_nocontract_t,
                                                       contract_t, 4,
                                                       lhs_inner_dim_contiguous,
                                                       false, Unaligned> LhsMapper;

        typedef internal::TensorContractionInputMapper<RhsScalar, Index, internal::Rhs,
                                                       RightEvaluator, right_nocontract_t,
                                                       contract_t, 4,
                                                       rhs_inner_dim_contiguous,
                                                       rhs_inner_dim_reordered, Unaligned> RhsMapper;

        typedef internal::blas_data_mapper<Scalar, Index, ColMajor> OutputMapper;


        // initialize data mappers
        LhsMapper lhs(this->m_leftImpl, this->m_left_nocontract_strides, this->m_i_strides,
                      this->m_left_contracting_strides, this->m_k_strides);

        RhsMapper rhs(this->m_rightImpl, this->m_right_nocontract_strides, this->m_j_strides,
                      this->m_right_contracting_strides, this->m_k_strides);

        OutputMapper output(buffer, m);

        const Index m_blocks = (m + 63) / 64;
        const Index n_blocks = (n + 63) / 64;
        const dim3 num_blocks(m_blocks, n_blocks, 1);
        const dim3 block_size(8, 8, 8);

        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        if (internal::is_same<LhsScalar, float>::value &&
            internal::is_same<RhsScalar, float>::value) {
          EigenFloatContractionKernel<Index, LhsMapper, RhsMapper, OutputMapper>
              <<<num_blocks, block_size, 0, this->m_device.stream()>>>(lhs, rhs, output, m, n, k);
        } else {
          EigenContractionKernel<Scalar, Index, LhsMapper, RhsMapper, OutputMapper>
              <<<num_blocks, block_size, 0, this->m_device.stream()>>>(lhs, rhs, output, m, n, k);
        }

        assert(cudaGetLastError() == cudaSuccess);
      }
    };

} // end namespace Eigen

#endif // EIGEN_USE_GPU and __CUDACC__

#endif // EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_CUDA_H
