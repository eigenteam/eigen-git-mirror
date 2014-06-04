// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_ASSIGN_H
#define EIGEN_CXX11_TENSOR_TENSOR_ASSIGN_H

#ifdef EIGEN_USE_THREADS
#include <future>
#endif

namespace Eigen {

/** \class TensorAssign
  * \ingroup CXX11_Tensor_Module
  *
  * \brief The tensor assignment class.
  *
  * This class is responsible for triggering the evaluation of the expressions
  * used on the lhs and rhs of an assignment operator and copy the result of
  * the evaluation of the rhs expression at the address computed during the
  * evaluation lhs expression.
  *
  * TODO: vectorization. For now the code only uses scalars
  * TODO: parallelisation using multithreading on cpu, or kernels on gpu.
  */
namespace internal {

// Default strategy: the expressions are evaluated with a single cpu thread.
template<typename Derived1, typename Derived2, bool Vectorizable = TensorEvaluator<Derived1>::PacketAccess & TensorEvaluator<Derived2>::PacketAccess>
struct TensorAssign
{
  typedef typename Derived1::Index Index;
  EIGEN_DEVICE_FUNC
  static inline void run(Derived1& dst, const Derived2& src)
  {
    TensorEvaluator<Derived1> evalDst(dst);
    TensorEvaluator<Derived2> evalSrc(src);
    const Index size = dst.size();
    for (Index i = 0; i < size; ++i) {
      evalDst.coeffRef(i) = evalSrc.coeff(i);
    }
  }
};


template<typename Derived1, typename Derived2>
struct TensorAssign<Derived1, Derived2, true>
{
  typedef typename Derived1::Index Index;
  static inline void run(Derived1& dst, const Derived2& src)
  {
    TensorEvaluator<Derived1> evalDst(dst);
    TensorEvaluator<Derived2> evalSrc(src);
    const Index size = dst.size();

    static const int LhsStoreMode = TensorEvaluator<Derived1>::IsAligned ? Aligned : Unaligned;
    static const int RhsLoadMode = TensorEvaluator<Derived2>::IsAligned ? Aligned : Unaligned;
    static const int PacketSize = unpacket_traits<typename TensorEvaluator<Derived1>::PacketReturnType>::size;
    const int VectorizedSize = (size / PacketSize) * PacketSize;

    for (Index i = 0; i < VectorizedSize; i += PacketSize) {
      evalDst.template writePacket<LhsStoreMode>(i, evalSrc.template packet<RhsLoadMode>(i));
    }
    for (Index i = VectorizedSize; i < size; ++i) {
      evalDst.coeffRef(i) = evalSrc.coeff(i);
    }
  }
};



// Multicore strategy: the index space is partitioned and each core is assigned to a partition
#ifdef EIGEN_USE_THREADS
template <typename LhsEval, typename RhsEval, typename Index, bool Vectorizable = LhsEval::PacketAccess & RhsEval::PacketAccess>
struct EvalRange {
  static void run(LhsEval& dst, const RhsEval& src, const Index first, const Index last) {
    eigen_assert(last > first);
    for (Index i = first; i < last; ++i) {
      dst.coeffRef(i) = src.coeff(i);
    }
  }
};

template <typename LhsEval, typename RhsEval, typename Index>
struct EvalRange<LhsEval, RhsEval, Index, true> {
  static void run(LhsEval& dst, const RhsEval& src, const Index first, const Index last) {
    eigen_assert(last > first);

    Index i = first;
    static const int PacketSize = unpacket_traits<typename LhsEval::PacketReturnType>::size;
    if (last - first > PacketSize) {
      static const int LhsStoreMode = LhsEval::IsAligned ? Aligned : Unaligned;
      static const int RhsLoadMode = RhsEval::IsAligned ? Aligned : Unaligned;
      eigen_assert(first % PacketSize == 0);
      Index lastPacket = last - (last % PacketSize);
      for (; i < lastPacket; i += PacketSize) {
        dst.template writePacket<LhsStoreMode>(i, src.template packet<RhsLoadMode>(i));
      }
    }

    for (; i < last; ++i) {
      dst.coeffRef(i) = src.coeff(i);
    }
  }
};

template<typename Derived1, typename Derived2>
struct TensorAssignMultiThreaded
{
  typedef typename Derived1::Index Index;
  static inline void run(Derived1& dst, const Derived2& src, const ThreadPoolDevice& device)
  {
    TensorEvaluator<Derived1> evalDst(dst);
    TensorEvaluator<Derived2> evalSrc(src);
    const Index size = dst.size();

    static const bool Vectorizable = TensorEvaluator<Derived1>::PacketAccess & TensorEvaluator<Derived2>::PacketAccess;
    static const int PacketSize = Vectorizable ? unpacket_traits<typename TensorEvaluator<Derived1>::PacketReturnType>::size : 1;

    int blocksz = static_cast<int>(ceil(static_cast<float>(size)/device.numThreads()) + PacketSize - 1);
    const Index blocksize = std::max<Index>(PacketSize, (blocksz - (blocksz % PacketSize)));
    const Index numblocks = size / blocksize;

    Index i = 0;
    vector<std::future<void> > results;
    results.reserve(numblocks);
    for (int i = 0; i < numblocks; ++i) {
      results.push_back(std::async(std::launch::async, &EvalRange<TensorEvaluator<Derived1>, TensorEvaluator<Derived2>, Index>::run, evalDst, evalSrc, i*blocksize, (i+1)*blocksize));
    }

    for (int i = 0; i < numblocks; ++i) {
      results[i].get();
    }

    if (numblocks * blocksize < size) {
      EvalRange<TensorEvaluator<Derived1>, TensorEvaluator<Derived2>, Index>::run(evalDst, evalSrc, numblocks * blocksize, size);
    }
  }
};
#endif


// GPU: the evaluation of the expressions is offloaded to a GPU.
#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
template <typename LhsEvaluator, typename RhsEvaluator>
__global__ void EigenMetaKernelNoCheck(LhsEvaluator evalDst, const RhsEvaluator evalSrc) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  evalDst.coeffRef(index) = evalSrc.coeff(index);
}
template <typename LhsEvaluator, typename RhsEvaluator>
__global__ void EigenMetaKernelPeel(LhsEvaluator evalDst, const RhsEvaluator evalSrc, int peel_start_offset, int size) {
  const int index = peel_start_offset + blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    evalDst.coeffRef(index) = evalSrc.coeff(index);
  }
}

template<typename Derived1, typename Derived2>
struct TensorAssignGpu
{
  typedef typename Derived1::Index Index;
  static inline void run(Derived1& dst, const Derived2& src, const GpuDevice& device)
  {
    TensorEvaluator<Derived1> evalDst(dst);
    TensorEvaluator<Derived2> evalSrc(src);
    const Index size = dst.size();
    const int block_size = std::min<int>(size, 32*32);
    const int num_blocks = size / block_size;
    EigenMetaKernelNoCheck<TensorEvaluator<Derived1>, TensorEvaluator<Derived2> > <<<num_blocks, block_size, 0, device.stream()>>>(evalDst, evalSrc);

    const int remaining_items = size % block_size;
    if (remaining_items > 0) {
      const int peel_start_offset = num_blocks * block_size;
      const int peel_block_size = std::min<int>(size, 32);
      const int peel_num_blocks = (remaining_items + peel_block_size - 1) / peel_block_size;
      EigenMetaKernelPeel<TensorEvaluator<Derived1>, TensorEvaluator<Derived2> > <<<peel_num_blocks, peel_block_size, 0, device.stream()>>>(evalDst, evalSrc, peel_start_offset, size);
    }
  }
};
#endif

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_ASSIGN_H
