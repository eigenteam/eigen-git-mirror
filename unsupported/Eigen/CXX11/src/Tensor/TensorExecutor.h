// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_EXECUTOR_H
#define EIGEN_CXX11_TENSOR_TENSOR_EXECUTOR_H

namespace Eigen {

/** \class TensorExecutor
  * \ingroup CXX11_Tensor_Module
  *
  * \brief The tensor executor class.
  *
  * This class is responsible for launch the evaluation of the expression on
  * the specified computing device.
  */
namespace internal {

// Default strategy: the expression is evaluated with a single cpu thread.
template<typename Expression, typename Device, bool Vectorizable>
class TensorExecutor
{
 public:
  typedef typename Expression::Index Index;
  EIGEN_DEVICE_FUNC
  static inline void run(const Expression& expr, const Device& device = Device())
  {
    TensorEvaluator<Expression, Device> evaluator(expr, device);
    const bool needs_assign = evaluator.evalSubExprsIfNeeded(NULL);
    if (needs_assign)
    {
      const Index size = array_prod(evaluator.dimensions());
      for (Index i = 0; i < size; ++i) {
        evaluator.evalScalar(i);
      }
    }
    evaluator.cleanup();
  }
};


template<typename Expression>
class TensorExecutor<Expression, DefaultDevice, true>
{
 public:
  typedef typename Expression::Index Index;
  EIGEN_DEVICE_FUNC
  static inline void run(const Expression& expr, const DefaultDevice& device = DefaultDevice())
  {
    TensorEvaluator<Expression, DefaultDevice> evaluator(expr, device);
    const bool needs_assign = evaluator.evalSubExprsIfNeeded(NULL);
    if (needs_assign)
    {
      const Index size = array_prod(evaluator.dimensions());
      const int PacketSize = unpacket_traits<typename TensorEvaluator<Expression, DefaultDevice>::PacketReturnType>::size;
      const Index VectorizedSize = (size / PacketSize) * PacketSize;

      for (Index i = 0; i < VectorizedSize; i += PacketSize) {
        evaluator.evalPacket(i);
      }
      for (Index i = VectorizedSize; i < size; ++i) {
        evaluator.evalScalar(i);
      }
    }
    evaluator.cleanup();
  }
};



// Multicore strategy: the index space is partitioned and each partition is executed on a single core
#ifdef EIGEN_USE_THREADS
template <typename Evaluator, typename Index, bool Vectorizable>
struct EvalRange {
  static void run(Evaluator evaluator, const Index first, const Index last) {
    eigen_assert(last > first);
    for (Index i = first; i < last; ++i) {
      evaluator.evalScalar(i);
    }
  }
};

template <typename Evaluator, typename Index>
struct EvalRange<Evaluator, Index, true> {
  static void run(Evaluator evaluator, const Index first, const Index last) {
    eigen_assert(last > first);

    Index i = first;
    static const int PacketSize = unpacket_traits<typename Evaluator::PacketReturnType>::size;
    if (last - first >= PacketSize) {
      eigen_assert(first % PacketSize == 0);
      Index lastPacket = last - (last % PacketSize);
      for (; i < lastPacket; i += PacketSize) {
        evaluator.evalPacket(i);
      }
    }

    for (; i < last; ++i) {
      evaluator.evalScalar(i);
    }
  }
};

template<typename Expression, bool Vectorizable>
class TensorExecutor<Expression, ThreadPoolDevice, Vectorizable>
{
 public:
  typedef typename Expression::Index Index;
  static inline void run(const Expression& expr, const ThreadPoolDevice& device)
  {
    typedef TensorEvaluator<Expression, ThreadPoolDevice> Evaluator;
    Evaluator evaluator(expr, device);
    const bool needs_assign = evaluator.evalSubExprsIfNeeded(NULL);
    if (needs_assign)
    {
      const Index size = array_prod(evaluator.dimensions());

      static const int PacketSize = Vectorizable ? unpacket_traits<typename Evaluator::PacketReturnType>::size : 1;

      int blocksz = std::ceil<int>(static_cast<float>(size)/device.numThreads()) + PacketSize - 1;
      const Index blocksize = numext::maxi<Index>(PacketSize, (blocksz - (blocksz % PacketSize)));
      const unsigned int numblocks = static_cast<unsigned int>(size / blocksize);

      Barrier barrier(numblocks);
      for (unsigned int i = 0; i < numblocks; ++i) {
        device.enqueue_with_barrier(&barrier, &EvalRange<Evaluator, Index, Vectorizable>::run, evaluator, i*blocksize, (i+1)*blocksize);
      }

      if (static_cast<Index>(numblocks) * blocksize < size) {
        EvalRange<Evaluator, Index, Vectorizable>::run(evaluator, numblocks * blocksize, size);
      }

      barrier.Wait();
    }
    evaluator.cleanup();
  }
};
#endif


// GPU: the evaluation of the expression is offloaded to a GPU.
#if defined(EIGEN_USE_GPU)

template <typename Expression, bool Vectorizable>
class TensorExecutor<Expression, GpuDevice, Vectorizable> {
 public:
  typedef typename Expression::Index Index;
  static void run(const Expression& expr, const GpuDevice& device);
};


#if defined(__CUDACC__)
template <typename Evaluator, typename Index, bool Vectorizable>
struct EigenMetaKernelEval {
  static __device__ EIGEN_ALWAYS_INLINE
  void run(Evaluator eval, Index first, Index last, Index step_size) {
    for (Index i = first; i < last; i += step_size) {
      eval.evalScalar(i);
    }
  }
};

template <typename Evaluator, typename Index>
struct EigenMetaKernelEval<Evaluator, Index, true> {
  static __device__ EIGEN_ALWAYS_INLINE
  void run(Evaluator eval, Index first, Index last, Index step_size) {
    const Index PacketSize = unpacket_traits<typename Evaluator::PacketReturnType>::size;
    const Index vectorized_size = (last / PacketSize) * PacketSize;
    const Index vectorized_step_size = step_size * PacketSize;

    // Use the vector path
    for (Index i = first * PacketSize; i < vectorized_size;
         i += vectorized_step_size) {
      eval.evalPacket(i);
    }
    for (Index i = vectorized_size + first; i < last; i += step_size) {
      eval.evalScalar(i);
    }
  }
};

template <typename Evaluator, typename Index>
__global__ void
__launch_bounds__(1024)
EigenMetaKernel(Evaluator memcopied_eval, Index size) {

  const Index first_index = blockIdx.x * blockDim.x + threadIdx.x;
  const Index step_size = blockDim.x * gridDim.x;

  // Cuda memcopies the kernel arguments. That's fine for POD, but for more
  // complex types such as evaluators we should really conform to the C++
  // standard and call a proper copy constructor.
  Evaluator eval(memcopied_eval);

  const bool vectorizable = Evaluator::PacketAccess & Evaluator::IsAligned;
  EigenMetaKernelEval<Evaluator, Index, vectorizable>::run(eval, first_index, size, step_size);
}

/*static*/
template <typename Expression, bool Vectorizable>
inline void TensorExecutor<Expression, GpuDevice, Vectorizable>::run(
    const Expression& expr, const GpuDevice& device) {
  TensorEvaluator<Expression, GpuDevice> evaluator(expr, device);
  const bool needs_assign = evaluator.evalSubExprsIfNeeded(NULL);
  if (needs_assign) {
    const int block_size = device.maxCudaThreadsPerBlock();
    const int max_blocks = device.getNumCudaMultiProcessors() *
                           device.maxCudaThreadsPerMultiProcessor() / block_size;
    const Index size = array_prod(evaluator.dimensions());
    // Create a least one block to ensure we won't crash when tensorflow calls with tensors of size 0.
    const int num_blocks = numext::maxi<int>(numext::mini<int>(max_blocks, (size + block_size - 1) / block_size), 1);

    LAUNCH_CUDA_KERNEL(
        (EigenMetaKernel<TensorEvaluator<Expression, GpuDevice>, Index>),
        num_blocks, block_size, 0, device, evaluator, size);
  }
  evaluator.cleanup();
}

#endif  // __CUDACC__
#endif  // EIGEN_USE_GPU


} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_EXECUTOR_H
