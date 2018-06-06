// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_HIP_H
#define EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_HIP_H

#if defined(EIGEN_HIP_DEVICE_COMPILE)
#include "Eigen/src/Core/arch/HIP/hcc/math_constants.h"
#endif

#if defined(EIGEN_HIPCC)
#define HIP_WARP_SIZE 64
#endif

namespace Eigen {
namespace internal {


#if defined(EIGEN_USE_GPU) && defined(EIGEN_HIPCC)
// Full reducers for GPU, don't vectorize for now

// Reducer function that enables multiple hip thread to safely accumulate at the same
// output address. It basically reads the current value of the output variable, and
// attempts to update it with the new value. If in the meantime another hip thread
// updated the content of the output address it will try again.
template <typename T, typename R>
__device__ EIGEN_ALWAYS_INLINE void atomicReduce(T* output, T accum, R& reducer) {
#if defined(EIGEN_HIP_DEVICE_COMPILE) && defined(__HIP_ARCH_HAS_WARP_SHUFFLE__)
  if (sizeof(T) == 4)
  {
    unsigned int oldval = *reinterpret_cast<unsigned int*>(output);
    unsigned int newval = oldval;
    reducer.reduce(accum, reinterpret_cast<T*>(&newval));
    if (newval == oldval) {
      return;
    }
    unsigned int readback;
    while ((readback = atomicCAS((unsigned int*)output, oldval, newval)) != oldval) {
      oldval = readback;
      newval = oldval;
      reducer.reduce(accum, reinterpret_cast<T*>(&newval));
      if (newval == oldval) {
        return;
      }
    }
  }
  else if (sizeof(T) == 8) {
    unsigned long long oldval = *reinterpret_cast<unsigned long long*>(output);
    unsigned long long newval = oldval;
    reducer.reduce(accum, reinterpret_cast<T*>(&newval));
    if (newval == oldval) {
      return;
    }
    unsigned long long readback;
    while ((readback = atomicCAS((unsigned long long*)output, oldval, newval)) != oldval) {
      oldval = readback;
      newval = oldval;
      reducer.reduce(accum, reinterpret_cast<T*>(&newval));
      if (newval == oldval) {
        return;
      }
    }
  }
  else {
    assert(0 && "Wordsize not supported");
  }
#else
  assert(0 && "Shouldn't be called on unsupported device");
#endif
}

// We extend atomicExch to support extra data types
template <typename Type>
__device__ inline Type atomicExchCustom(Type* address, Type val) {
  return atomicExch(address, val);
}

template <>
__device__ inline double atomicExchCustom(double* address, double val) {
  unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
  return __longlong_as_double(atomicExch(address_as_ull, __double_as_longlong(val)));
}

#if defined(EIGEN_HAS_HIP_FP16)
template <template <typename T> class R>
__device__ inline void atomicReduce(half2* output, half2 accum, R<half>& reducer) {
  unsigned int oldval = *reinterpret_cast<unsigned int*>(output);
  unsigned int newval = oldval;
  reducer.reducePacket(accum, reinterpret_cast<half2*>(&newval));
  if (newval == oldval) {
    return;
  }
  unsigned int readback;
  while ((readback = atomicCAS((unsigned int*)output, oldval, newval)) != oldval) {
    oldval = readback;
    newval = oldval;
    reducer.reducePacket(accum, reinterpret_cast<half2*>(&newval));
    if (newval == oldval) {
      return;
    }
  }
}
#endif

template <>
__device__ inline void atomicReduce(float* output, float accum, SumReducer<float>&) {
#if defined(EIGEN_HIP_DEVICE_COMPILE) && (__HIP_DEVICE_COMPILE__ == 1) &&\
    defined(__HIP_ARCH_HAS_WARP_SHUFFLE__)
  atomicAdd(output, accum);
#else
  assert(0 && "Shouldn't be called on unsupported device");
#endif
}


template <typename CoeffType, typename Index>
__global__ void ReductionInitKernel(const CoeffType val, Index num_preserved_coeffs, CoeffType* output) {
  const Index thread_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  const Index num_threads = hipBlockDim_x * hipGridDim_x;
  for (Index i = thread_id; i < num_preserved_coeffs; i += num_threads) {
    output[i] = val;
  }
}


template <int BlockSize, int NumPerThread, typename Self,
          typename Reducer, typename Index>
__global__ void FullReductionKernel(const Self input, Index num_coeffs,
                                    typename Self::CoeffReturnType* output, unsigned int* semaphore, Reducer reducer) {
#if defined(EIGEN_HIP_DEVICE_COMPILE) && (__HIP_DEVICE_COMPILE__ == 1) &&\
    defined(__HIP_ARCH_HAS_WARP_SHUFFLE__)
  // Initialize the output value
  const Index first_index = hipBlockIdx_x * BlockSize * NumPerThread + hipThreadIdx_x;
  if (hipGridDim_x == 1) {
    if (first_index == 0) {
      *output = reducer.initialize();
    }
  }
  else {
    if (hipThreadIdx_x == 0) {
      unsigned int block = atomicCAS(semaphore, 0u, 1u);
      if (block == 0) {
        // We're the first block to run, initialize the output value
        atomicExchCustom(output, reducer.initialize());
        __threadfence();
        atomicExch(semaphore, 2u);
      }
      else {
        // Wait for the first block to initialize the output value.
        // Use atomicCAS here to ensure that the reads aren't cached
        unsigned int val;
        do {
          val = atomicCAS(semaphore, 2u, 2u);
        }
        while (val < 2u);
      }
    }
  }

  __syncthreads();

  eigen_assert(hipGridDim_x == 1 || *semaphore >= 2u);

  typename Self::CoeffReturnType accum = reducer.initialize();
  Index max_iter = numext::mini<Index>(num_coeffs - first_index, NumPerThread*BlockSize);
  for (Index i = 0; i < max_iter; i+=BlockSize) {
    const Index index = first_index + i;
    eigen_assert(index < num_coeffs);
    typename Self::CoeffReturnType val = input.m_impl.coeff(index);
    reducer.reduce(val, &accum);
  }

#pragma unroll
  for (int offset = HIP_WARP_SIZE/2; offset > 0; offset /= 2) {
    // XXX use std::is_floating_point to determine the type of accum
    if (std::is_floating_point<typename Self::CoeffReturnType>::value) {
      reducer.reduce(__shfl_down(static_cast<float>(accum), offset, HIP_WARP_SIZE), &accum);
    } else {
      reducer.reduce(__shfl_down(static_cast<int>(accum), offset, HIP_WARP_SIZE), &accum);
    }
  }

  if ((hipThreadIdx_x & (HIP_WARP_SIZE - 1)) == 0) {
    atomicReduce(output, accum, reducer);
  }

  if (hipGridDim_x > 1 && hipThreadIdx_x == 0) {
    // Let the last block reset the semaphore
    atomicInc(semaphore, hipGridDim_x + 1);
    __threadfence_system();
  }

#else
  assert(0 && "Shouldn't be called on unsupported device");
#endif
}


#if defined(EIGEN_HAS_HIP_FP16)
template <typename Self,
          typename Reducer, typename Index>
__global__ void ReductionInitFullReduxKernelHalfFloat(Reducer reducer, const Self input, Index num_coeffs, half2* scratch) {
  eigen_assert(hipBlockDim_x == 1);
  eigen_assert(hipGridDim_x == 1);
  if (num_coeffs % 2 != 0) {
    half last = input.m_impl.coeff(num_coeffs-1);
    *scratch = __halves2half2(last, reducer.initialize());
  } else {
    *scratch = reducer.template initializePacket<half2>();
  }
}

template <typename Self,
          typename Reducer, typename Index>
__global__ void ReductionInitKernelHalfFloat(Reducer reducer, const Self input, Index num_coeffs, half* output) {
  const Index thread_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  const Index num_threads = hipBlockDim_x * hipGridDim_x;
  const Index num_packets = num_coeffs / 2;
  for (Index i = thread_id; i < num_packets; i += num_threads) {
    ((half2*)output)[i] = reducer.template initializePacket<half2>();
  }

  if (thread_id == 0 && num_coeffs % 2 != 0) {
    output[num_coeffs-1] = reducer.initialize();
  }
}

template <int BlockSize, int NumPerThread, typename Self,
          typename Reducer, typename Index>
__global__ void FullReductionKernelHalfFloat(Reducer reducer, const Self input, Index num_coeffs,
                                    half* output, half2* scratch) {
  eigen_assert(NumPerThread % 2 == 0);

  const Index first_index = hipBlockIdx_x * BlockSize * NumPerThread + 2*hipThreadIdx_x;

  // Initialize the output value if it wasn't initialized by the ReductionInitKernel
  if (hipGridDim_x == 1 && first_index == 0) {
    if (num_coeffs % 2 != 0) {
      half last = input.m_impl.coeff(num_coeffs-1);
      *scratch = __halves2half2(last, reducer.initialize());
    } else {
      *scratch = reducer.template initializePacket<half2>();
    }
    __syncthreads();
  }

  half2 accum = reducer.template initializePacket<half2>();
  const Index max_iter = numext::mini<Index>((num_coeffs - first_index) / 2, NumPerThread*BlockSize / 2);
  for (Index i = 0; i < max_iter; i += BlockSize) {
    const Index index = first_index + 2*i;
    eigen_assert(index + 1 < num_coeffs);
    half2 val = input.m_impl.template packet<Unaligned>(index);
    reducer.reducePacket(val, &accum);
  }

#pragma unroll
  for (int offset = HIP_WARP_SIZE/2; offset > 0; offset /= 2) {
    // FIXME : remove this workaround once we have native half/half2 support for __shfl_down
    union { int i; half2 h; } wka_in, wka_out;
    wka_in.h = accum;
    wka_out.i = __shfl_down(wka_in.i, offset, HIP_WARP_SIZE);
    reducer.reducePacket(wka_out.h, &accum);
  }

  if ((hipThreadIdx_x & (HIP_WARP_SIZE - 1)) == 0) {
    atomicReduce(scratch, accum, reducer);
  }

  __syncthreads();

  if (hipGridDim_x == 1 && first_index == 0) {
    half tmp = __low2half(*scratch);
    reducer.reduce(__high2half(*scratch), &tmp);
    *output = tmp;
  }
}

template <typename Op>
__global__ void ReductionCleanupKernelHalfFloat(Op& reducer, half* output, half2* scratch) {
  eigen_assert(hipThreadIdx_x == 1);
  half tmp = __low2half(*scratch);
  reducer.reduce(__high2half(*scratch), &tmp);
  *output = tmp;
}

#endif

template <typename Self, typename Op, typename OutputType, bool PacketAccess, typename Enabled = void>
struct FullReductionLauncher {
  static void run(const Self&, Op&, const GpuDevice&, OutputType*, typename Self::Index) {
    assert(false && "Should only be called on doubles, floats and half floats");
  }
};

namespace {
  std::mutex __eigen_reduction_hip_mutex;
}

// Specialization for float and double
template <typename Self, typename Op, typename OutputType, bool PacketAccess>
struct FullReductionLauncher<
    Self, Op, OutputType, PacketAccess,
    typename internal::enable_if<
      internal::is_same<float, OutputType>::value ||
      internal::is_same<double, OutputType>::value,
    void>::type> {
  static void run(const Self& self, Op& reducer, const GpuDevice& device, OutputType* output, typename Self::Index num_coeffs) {
    // guard FullReductionLauncher with a mutex so only 1 FullReductionKernel
    // is dispatched at a time
    std::lock_guard<std::mutex> lock(__eigen_reduction_hip_mutex);

    typedef typename Self::Index Index;
    typedef typename Self::CoeffReturnType Scalar;
    const int block_size = 256;
    const int num_per_thread = 128;
    const int num_blocks = divup<int>(num_coeffs, block_size * num_per_thread);

    unsigned int* semaphore = NULL;
    if (num_blocks > 1) {
      semaphore = device.semaphore();

      unsigned int semaphore_host = 0xFF;
      hipMemcpy(&semaphore_host, semaphore, sizeof(unsigned int), hipMemcpyDeviceToHost);
      if (semaphore_host != 0) {
        std::cerr << "[WARN][EIGEN][FullReductionLauncher] incorrect semaphore value: "
                  << semaphore_host << "\n";
        // wait for all commands on the device to complete so semaphore value
        // is reset to 0
        hipDeviceSynchronize();

        // read again
        hipMemcpy(&semaphore_host, semaphore, sizeof(unsigned int), hipMemcpyDeviceToHost);
        if (semaphore_host != 0) {
          std::cerr << "[ERROR][EIGEN][FullReductionLauncher] CRITICAL incorrect semaphore value: "
                    << semaphore_host << ", apply manual override to 0\n";

          // force set semaphore value to be 0
          semaphore_host = 0;
          hipMemcpy(semaphore, &semaphore_host, sizeof(unsigned int), hipMemcpyHostToDevice);
        }
      }
    }

    hipLaunchKernelGGL(HIP_KERNEL_NAME(FullReductionKernel<block_size, num_per_thread, Self, Op, Index>),
                       dim3(num_blocks), dim3(block_size), 0, device.stream(), self, num_coeffs, output, semaphore, reducer);
  }
};

#if defined(EIGEN_HAS_HIP_FP16)
template <typename Self, typename Op>
struct FullReductionLauncher<Self, Op, Eigen::half, false> {
  static void run(const Self&, Op&, const GpuDevice&, half*, typename Self::Index) {
    assert(false && "Should not be called since there is no packet accessor");
  }
};

template <typename Self, typename Op>
struct FullReductionLauncher<Self, Op, Eigen::half, true> {
  static void run(const Self& self, Op& reducer, const GpuDevice& device, half* output, typename Self::Index num_coeffs) {
    typedef typename Self::Index Index;

    const int block_size = 256;
    const int num_per_thread = 128;
    const int num_blocks = divup<int>(num_coeffs, block_size * num_per_thread);
    half2* scratch = static_cast<half2*>(device.scratchpad());

    if (num_blocks > 1) {
      // We initialize the output and the scrathpad outside the reduction kernel when we can't be sure that there
      // won't be a race conditions between multiple thread blocks.
      hipLaunchKernelGGL(HIP_KERNEL_NAME(ReductionInitFullReduxKernelHalfFloat<Self, Op, Index>),
                         dim3(1), dim3(1), 0, device.stream(), reducer, self, num_coeffs, scratch);
    }

    hipLaunchKernelGGL(HIP_KERNEL_NAME(FullReductionKernelHalfFloat<block_size, num_per_thread, Self, Op, Index>),
                       dim3(num_blocks), dim3(block_size), 0, device.stream(), reducer, self, num_coeffs, output, scratch);

    if (num_blocks > 1) {
      hipLaunchKernelGGL(HIP_KERNEL_NAME(ReductionCleanupKernelHalfFloat<Op>),
                         dim3(1), dim3(1), 0, device.stream(), reducer, output, scratch);
    }
  }
};
#endif


template <typename Self, typename Op, bool Vectorizable>
struct FullReducer<Self, Op, GpuDevice, Vectorizable> {
  // Unfortunately nvidia doesn't support well exotic types such as complex,
  // so reduce the scope of the optimized version of the code to the simple cases
  // of doubles, floats and half floats
#if defined(EIGEN_HAS_HIP_FP16)
  static const bool HasOptimizedImplementation = !Op::IsStateful &&
      (internal::is_same<typename Self::CoeffReturnType, float>::value ||
       internal::is_same<typename Self::CoeffReturnType, double>::value ||
       (internal::is_same<typename Self::CoeffReturnType, Eigen::half>::value && reducer_traits<Op, GpuDevice>::PacketAccess));
#else
  static const bool HasOptimizedImplementation = !Op::IsStateful &&
                                                (internal::is_same<typename Self::CoeffReturnType, float>::value ||
                                                 internal::is_same<typename Self::CoeffReturnType, double>::value);
#endif

  template <typename OutputType>
  static void run(const Self& self, Op& reducer, const GpuDevice& device, OutputType* output) {
    assert(HasOptimizedImplementation && "Should only be called on doubles, floats or half floats");
    const Index num_coeffs = array_prod(self.m_impl.dimensions());
    // Don't crash when we're called with an input tensor of size 0.
    if (num_coeffs == 0) {
      return;
    }

    FullReductionLauncher<Self, Op, OutputType, reducer_traits<Op, GpuDevice>::PacketAccess>::run(self, reducer, device, output, num_coeffs);
  }
};


template <int NumPerThread, typename Self,
          typename Reducer, typename Index>
__global__ void InnerReductionKernel(Reducer reducer, const Self input, Index num_coeffs_to_reduce, Index num_preserved_coeffs,
                                         typename Self::CoeffReturnType* output) {
#if defined(EIGEN_HIP_DEVICE_COMPILE) && (__HIP_DEVICE_COMPILE__ == 1) &&\
    defined(__HIP_ARCH_HAS_WARP_SHUFFLE__)
  typedef typename Self::CoeffReturnType Type;
  eigen_assert(hipBlockDim_y == 1);
  eigen_assert(hipBlockDim_z == 1);
  eigen_assert(hipGridDim_y == 1);
  eigen_assert(hipGridDim_z == 1);

  const int unroll_times = 16;
  eigen_assert(NumPerThread % unroll_times == 0);

  const Index input_col_blocks = divup<Index>(num_coeffs_to_reduce, hipBlockDim_x * NumPerThread);
  const Index num_input_blocks = input_col_blocks * num_preserved_coeffs;

  const Index num_threads = hipBlockDim_x * hipGridDim_x;
  const Index thread_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

  // Initialize the output values if they weren't initialized by the ReductionInitKernel
  if (hipGridDim_x == 1) {
    for (Index i = thread_id; i < num_preserved_coeffs; i += num_threads) {
      output[i] = reducer.initialize();
    }
    __syncthreads();
  }

  for (Index i = hipBlockIdx_x; i < num_input_blocks; i += hipGridDim_x) {
    const Index row = i / input_col_blocks;

    if (row < num_preserved_coeffs) {
      const Index col_block = i % input_col_blocks;
      const Index col_begin = col_block * hipBlockDim_x * NumPerThread + hipThreadIdx_x;

      Type reduced_val = reducer.initialize();

      for (Index j = 0; j < NumPerThread; j += unroll_times) {
        const Index last_col = col_begin + hipBlockDim_x * (j + unroll_times - 1);
        if (last_col >= num_coeffs_to_reduce) {
          for (Index col = col_begin + hipBlockDim_x * j; col < num_coeffs_to_reduce; col += hipBlockDim_x) {
            const Type val = input.m_impl.coeff(row * num_coeffs_to_reduce + col);
            reducer.reduce(val, &reduced_val);
          }
          break;
        } else {
          // Faster version of the loop with no branches after unrolling.
#pragma unroll
          for (int k = 0; k < unroll_times; ++k) {
            const Index col = col_begin + hipBlockDim_x * (j + k);
            reducer.reduce(input.m_impl.coeff(row * num_coeffs_to_reduce + col), &reduced_val);
          }
        }
      }

#pragma unroll
      for (int offset = HIP_WARP_SIZE/2; offset > 0; offset /= 2) {
        // XXX use std::is_floating_point to determine the type of reduced_val
        if (std::is_floating_point<Type>::value) {
          reducer.reduce(__shfl_down(static_cast<float>(reduced_val), offset), &reduced_val);
        } else {
          reducer.reduce(__shfl_down(static_cast<int>(reduced_val), offset), &reduced_val);
        }
      }

      if ((hipThreadIdx_x & (HIP_WARP_SIZE - 1)) == 0) {
        atomicReduce(&(output[row]), reduced_val, reducer);
      }
    }
  }
#else
  assert(0 && "Shouldn't be called on unsupported device");
#endif
}

#if defined(EIGEN_HAS_HIP_FP16)

template <int NumPerThread, typename Self,
          typename Reducer, typename Index>
__global__ void InnerReductionKernelHalfFloat(Reducer reducer, const Self input, Index num_coeffs_to_reduce, Index num_preserved_coeffs,
                                              half* output) {
  eigen_assert(hipBlockDim_y == 1);
  eigen_assert(hipBlockDim_z == 1);
  eigen_assert(hipGridDim_y == 1);
  eigen_assert(hipGridDim_z == 1);

  const int unroll_times = 16;
  eigen_assert(NumPerThread % unroll_times == 0);
  eigen_assert(unroll_times % 2 == 0);

  const Index input_col_blocks = divup<Index>(num_coeffs_to_reduce, hipBlockDim_x * NumPerThread * 2);
  const Index num_input_blocks = divup<Index>(input_col_blocks * num_preserved_coeffs, 2);

  const Index num_threads = hipBlockDim_x * hipGridDim_x;
  const Index thread_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

  // Initialize the output values if they weren't initialized by the ReductionInitKernel
  if (hipGridDim_x == 1) {
    Index i = 2*thread_id;
    for (; i + 1 < num_preserved_coeffs; i += 2*num_threads) {
      half* loc = output + i;
      *((half2*)loc) = reducer.template initializePacket<half2>();
    }
    if (i < num_preserved_coeffs) {
      output[i] = reducer.initialize();
    }
    __syncthreads();
  }

  for (Index i = hipBlockIdx_x; i < num_input_blocks; i += hipGridDim_x) {
    const Index row = 2 * (i / input_col_blocks);

    if (row + 1 < num_preserved_coeffs) {
      const Index col_block = i % input_col_blocks;
      const Index col_begin = 2 * (col_block * hipBlockDim_x * NumPerThread + hipThreadIdx_x);

      half2 reduced_val1 = reducer.template initializePacket<half2>();
      half2 reduced_val2 = reducer.template initializePacket<half2>();

      for (Index j = 0; j < NumPerThread; j += unroll_times) {
        const Index last_col = col_begin + hipBlockDim_x * (j + unroll_times - 1) * 2;
        if (last_col >= num_coeffs_to_reduce) {
          Index col = col_begin + hipBlockDim_x * j;
          for (; col + 1 < num_coeffs_to_reduce; col += hipBlockDim_x) {
            const half2 val1 = input.m_impl.template packet<Unaligned>(row * num_coeffs_to_reduce + col);
            reducer.reducePacket(val1, &reduced_val1);
            const half2 val2 = input.m_impl.template packet<Unaligned>((row+1) * num_coeffs_to_reduce + col);
            reducer.reducePacket(val2, &reduced_val2);
          }
          if (col < num_coeffs_to_reduce) {
            // Peel;
            const half last1 = input.m_impl.coeff(row * num_coeffs_to_reduce + col);
            const half2 val1 = __halves2half2(last1, reducer.initialize());
            reducer.reducePacket(val1, &reduced_val1);
            const half last2 = input.m_impl.coeff((row+1) * num_coeffs_to_reduce + col);
            const half2 val2 = __halves2half2(last2, reducer.initialize());
            reducer.reducePacket(val2, &reduced_val2);
          }
          break;
        } else {
          // Faster version of the loop with no branches after unrolling.
#pragma unroll
          for (int k = 0; k < unroll_times; ++k) {
            const Index col = col_begin + hipBlockDim_x * (j + k) * 2;
            reducer.reducePacket(input.m_impl.template packet<Unaligned>(row * num_coeffs_to_reduce + col), &reduced_val1);
            reducer.reducePacket(input.m_impl.template packet<Unaligned>((row + 1)* num_coeffs_to_reduce + col), &reduced_val2);
          }
        }
      }

#pragma unroll
      for (int offset = HIP_WARP_SIZE/2; offset > 0; offset /= 2) {
	// FIXME : remove this workaround once we have native half/half2 support for __shfl_down
	union { int i; half2 h; } wka_in, wka_out;

	wka_in.h = reduced_val1;
	wka_out.i = __shfl_down(wka_in.i, offset, HIP_WARP_SIZE);
        reducer.reducePacket(wka_out.h, &reduced_val1);
	
	wka_in.h = reduced_val2;
	wka_out.i = __shfl_down(wka_in.i, offset, HIP_WARP_SIZE);
        reducer.reducePacket(wka_out.h, &reduced_val2);
      }

      half val1 =  __low2half(reduced_val1);
      reducer.reduce(__high2half(reduced_val1), &val1);
      half val2 =  __low2half(reduced_val2);
      reducer.reduce(__high2half(reduced_val2), &val2);
      half2 val = __halves2half2(val1, val2);

      if ((hipThreadIdx_x & (HIP_WARP_SIZE - 1)) == 0) {
        half* loc = output + row;
        atomicReduce((half2*)loc, val, reducer);
      }
    }
  }
}

#endif

template <typename Self, typename Op, typename OutputType, bool PacketAccess, typename Enabled = void>
struct InnerReductionLauncher {
  static bool run(const Self&, Op&, const GpuDevice&, OutputType*, typename Self::Index, typename Self::Index) {
    assert(false && "Should only be called to reduce doubles, floats and half floats on a gpu device");
    return true;
  }
};

// Specialization for float and double
template <typename Self, typename Op, typename OutputType, bool PacketAccess>
struct InnerReductionLauncher<
  Self, Op, OutputType, PacketAccess,
  typename internal::enable_if<
    internal::is_same<float, OutputType>::value ||
    internal::is_same<double, OutputType>::value,
  void>::type> {
  static bool run(const Self& self, Op& reducer, const GpuDevice& device, OutputType* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
    typedef typename Self::Index Index;

    const Index num_coeffs = num_coeffs_to_reduce * num_preserved_vals;
    const int block_size = 256;
    const int num_per_thread = 128;
    const int dyn_blocks = divup<int>(num_coeffs, block_size * num_per_thread);
    const int max_blocks = device.getNumHipMultiProcessors() *
                           device.maxHipThreadsPerMultiProcessor() / block_size;
    const int num_blocks = numext::mini<int>(max_blocks, dyn_blocks);

    if (num_blocks > 1) {
      // We initialize the outputs outside the reduction kernel when we can't be sure that there
      // won't be a race conditions between multiple thread blocks.
      const int dyn_blocks = divup<int>(num_preserved_vals, 1024);
      const int max_blocks = device.getNumHipMultiProcessors() *
                           device.maxHipThreadsPerMultiProcessor() / 1024;
      const int num_blocks = numext::mini<int>(max_blocks, dyn_blocks);
      hipLaunchKernelGGL(HIP_KERNEL_NAME(ReductionInitKernel<OutputType, Index>),
                         dim3(num_blocks), dim3(1024), 0, device.stream(),
                         reducer.initialize(), num_preserved_vals, output);
    }

    hipLaunchKernelGGL(HIP_KERNEL_NAME(InnerReductionKernel<num_per_thread, Self, Op, Index>),
                       dim3(num_blocks), dim3(block_size), 0, device.stream(), reducer, self,
                       num_coeffs_to_reduce, num_preserved_vals, output);

    return false;
  }
};

#if defined(EIGEN_HAS_HIP_FP16)
template <typename Self, typename Op>
struct InnerReductionLauncher<Self, Op, Eigen::half, false> {
  static bool run(const Self&, Op&, const GpuDevice&, half*, typename Self::Index, typename Self::Index) {
    assert(false && "Should not be called since there is no packet accessor");
    return true;
  }
};

template <typename Self, typename Op>
struct InnerReductionLauncher<Self, Op, Eigen::half, true> {
  static bool run(const Self& self, Op& reducer, const GpuDevice& device, half* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
    typedef typename Self::Index Index;

    if (num_preserved_vals % 2 != 0) {
      // Not supported yet, revert to the slower code path
      return true;
    }

    const Index num_coeffs = num_coeffs_to_reduce * num_preserved_vals;
    const int block_size = /*256*/128;
    const int num_per_thread = /*128*/64;
    const int dyn_blocks = divup<int>(num_coeffs, block_size * num_per_thread);
    const int max_blocks = device.getNumHipMultiProcessors() *
                           device.maxHipThreadsPerMultiProcessor() / block_size;
    const int num_blocks = numext::mini<int>(max_blocks, dyn_blocks);

    if (num_blocks > 1) {
      // We initialize the outputs outside the reduction kernel when we can't be sure that there
      // won't be a race conditions between multiple thread blocks.
      const int dyn_blocks = divup<int>(num_preserved_vals, 1024);
      const int max_blocks = device.getNumHipMultiProcessors() *
                           device.maxHipThreadsPerMultiProcessor() / 1024;
      const int num_blocks = numext::mini<int>(max_blocks, dyn_blocks);
      hipLaunchKernelGGL(HIP_KERNEL_NAME(ReductionInitKernelHalfFloat<Self, Op, Index>),
                         dim3(1), dim3(1), 0, device.stream(), reducer, self, num_preserved_vals, output);
    }

    hipLaunchKernelGGL(HIP_KERNEL_NAME(InnerReductionKernelHalfFloat<num_per_thread, Self, Op, Index>),
                       dim3(num_blocks), dim3(block_size), 0, device.stream(), reducer, self, num_coeffs_to_reduce, num_preserved_vals, output);

    return false;
  }
};
#endif


template <typename Self, typename Op>
struct InnerReducer<Self, Op, GpuDevice> {
  // Unfortunately nvidia doesn't support well exotic types such as complex,
  // so reduce the scope of the optimized version of the code to the simple case
  // of floats and half floats.
#if defined(EIGEN_HAS_HIP_FP16)
  static const bool HasOptimizedImplementation = !Op::IsStateful &&
      (internal::is_same<typename Self::CoeffReturnType, float>::value ||
       internal::is_same<typename Self::CoeffReturnType, double>::value ||
       (internal::is_same<typename Self::CoeffReturnType, Eigen::half>::value && reducer_traits<Op, GpuDevice>::PacketAccess));
#else
  static const bool HasOptimizedImplementation = !Op::IsStateful &&
                                                 (internal::is_same<typename Self::CoeffReturnType, float>::value ||
                                                  internal::is_same<typename Self::CoeffReturnType, double>::value);
#endif

  template <typename OutputType>
  static bool run(const Self& self, Op& reducer, const GpuDevice& device, OutputType* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
    assert(HasOptimizedImplementation && "Should only be called on doubles, floats or half floats");
    const Index num_coeffs = array_prod(self.m_impl.dimensions());
    // Don't crash when we're called with an input tensor of size 0.
    if (num_coeffs == 0) {
      return true;
    }
    // It's faster to use the usual code.
    if (num_coeffs_to_reduce <= 128) {
      return true;
    }

    return InnerReductionLauncher<Self, Op, OutputType, reducer_traits<Op, GpuDevice>::PacketAccess>::run(self, reducer, device, output, num_coeffs_to_reduce, num_preserved_vals);
  }
};

template <int NumPerThread, typename Self,
          typename Reducer, typename Index>
__global__ void OuterReductionKernel(Reducer reducer, const Self input, Index num_coeffs_to_reduce, Index num_preserved_coeffs,
                                     typename Self::CoeffReturnType* output) {
  const Index num_threads = hipBlockDim_x * hipGridDim_x;
  const Index thread_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  // Initialize the output values if they weren't initialized by the ReductionInitKernel
  if (hipGridDim_x == 1) {
    for (Index i = thread_id; i < num_preserved_coeffs; i += num_threads) {
      output[i] = reducer.initialize();
    }
    __syncthreads();
  }

  // Do the reduction.
  const Index max_iter = num_preserved_coeffs * divup<Index>(num_coeffs_to_reduce, NumPerThread);
  for (Index i = thread_id; i < max_iter; i += num_threads) {
    const Index input_col = i % num_preserved_coeffs;
    const Index input_row = (i / num_preserved_coeffs) * NumPerThread;
    typename Self::CoeffReturnType reduced_val = reducer.initialize();
    const Index max_row = numext::mini(input_row + NumPerThread, num_coeffs_to_reduce);
    for (Index j = input_row; j < max_row; j++) {
      typename Self::CoeffReturnType val = input.m_impl.coeff(j * num_preserved_coeffs + input_col);
      reducer.reduce(val, &reduced_val);
    }
    atomicReduce(&(output[input_col]), reduced_val, reducer);
  }
}


template <typename Self, typename Op>
struct OuterReducer<Self, Op, GpuDevice> {
  // Unfortunately nvidia doesn't support well exotic types such as complex,
  // so reduce the scope of the optimized version of the code to the simple case
  // of floats.
  static const bool HasOptimizedImplementation = !Op::IsStateful &&
                                                 (internal::is_same<typename Self::CoeffReturnType, float>::value ||
                                                  internal::is_same<typename Self::CoeffReturnType, double>::value);
  template <typename Device, typename OutputType>
  static bool run(const Self&, Op&, const Device&, OutputType*, typename Self::Index, typename Self::Index) {
    assert(false && "Should only be called to reduce doubles or floats on a gpu device");
    return true;
  }

  static bool run(const Self& self, Op& reducer, const GpuDevice& device, float* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
    typedef typename Self::Index Index;

    // It's faster to use the usual code.
    if (num_coeffs_to_reduce <= 32) {
      return true;
    }

    const Index num_coeffs = num_coeffs_to_reduce * num_preserved_vals;
    const int block_size = 256;
    const int num_per_thread = 16;
    const int dyn_blocks = divup<int>(num_coeffs, block_size * num_per_thread);
    const int max_blocks = device.getNumHipMultiProcessors() *
                           device.maxHipThreadsPerMultiProcessor() / block_size;
    const int num_blocks = numext::mini<int>(max_blocks, dyn_blocks);

    if (num_blocks > 1) {
      // We initialize the outputs in the reduction kernel itself when we don't have to worry
      // about race conditions between multiple thread blocks.
      const int dyn_blocks = divup<int>(num_preserved_vals, 1024);
      const int max_blocks = device.getNumHipMultiProcessors() *
                             device.maxHipThreadsPerMultiProcessor() / 1024;
      const int num_blocks = numext::mini<int>(max_blocks, dyn_blocks);
      hipLaunchKernelGGL(HIP_KERNEL_NAME(ReductionInitKernel<float, Index>),
                         dim3(num_blocks), dim3(1024), 0, device.stream(),
                         reducer.initialize(), num_preserved_vals, output);
    }

    hipLaunchKernelGGL(HIP_KERNEL_NAME(OuterReductionKernel<num_per_thread, Self, Op, Index>),
                       dim3(num_blocks), dim3(block_size), 0, device.stream(), reducer, self, num_coeffs_to_reduce, num_preserved_vals, output);

    return false;
  }
};

#endif


} // end namespace internal
} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_HIP_H
