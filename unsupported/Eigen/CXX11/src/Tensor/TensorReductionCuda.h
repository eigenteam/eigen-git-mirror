// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_CUDA_H
#define EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_CUDA_H

namespace Eigen {
namespace internal {


#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
// Full reducers for GPU, don't vectorize for now

// Reducer function that enables multiple cuda thread to safely accumulate at the same
// output address. It basically reads the current value of the output variable, and
// attempts to update it with the new value. If in the meantime another cuda thread
// updated the content of the output address it will try again.
template <typename T, typename R>
__device__ EIGEN_ALWAYS_INLINE void atomicReduce(T* output, T accum, R& reducer) {
#if __CUDA_ARCH__ >= 300
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

template <typename T>
__device__ inline void atomicReduce(T* output, T accum, SumReducer<T>&) {
#if __CUDA_ARCH__ >= 300
  atomicAdd(output, accum);
#else
  assert(0 && "Shouldn't be called on unsupported device");
#endif
}


template <typename CoeffType, typename Index>
__global__ void ReductionInitKernel(const CoeffType val, Index num_preserved_coeffs, CoeffType* output) {
  const Index thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const Index num_threads = blockDim.x * gridDim.x;
  for (Index i = thread_id; i < num_preserved_coeffs; i += num_threads) {
    output[i] = val;
  }
}

template <int BlockSize, int NumPerThread, typename Self,
          typename Reducer, typename Index>
__global__ void FullReductionKernel(Reducer reducer, const Self input, Index num_coeffs,
                                    typename Self::CoeffReturnType* output) {
  const Index first_index = blockIdx.x * BlockSize * NumPerThread + threadIdx.x;

  // Initialize the output value if it wasn't initialized by the ReductionInitKernel
  if (gridDim.x == 1 && first_index == 0) {
    *output = reducer.initialize();
  }

  typename Self::CoeffReturnType accum = reducer.initialize();
  Index max_iter = numext::mini<Index>(num_coeffs - first_index, NumPerThread*BlockSize);
  for (Index i = 0; i < max_iter; i+=BlockSize) {
    const Index index = first_index + i;
    eigen_assert(index < num_coeffs);
    typename Self::CoeffReturnType val = input.m_impl.coeff(index);
    reducer.reduce(val, &accum);
  }

#pragma unroll
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    reducer.reduce(__shfl_down(accum, offset), &accum);
  }

  if ((threadIdx.x & (warpSize - 1)) == 0) {
    atomicReduce(output, accum, reducer);
  }
}


template <typename Self, typename Op, bool Vectorizable>
struct FullReducer<Self, Op, GpuDevice, Vectorizable> {
  // Unfortunately nvidia doesn't support well exotic types such as complex,
  // so reduce the scope of the optimized version of the code to the simple case
  // of floats.
  static const bool HasOptimizedImplementation = !Op::IsStateful &&
                                                 internal::is_same<typename Self::CoeffReturnType, float>::value;

  template <typename OutputType>
  static EIGEN_DEVICE_FUNC void run(const Self&, Op&, const GpuDevice&, OutputType*) {
    assert(false && "Should only be called on floats");
  }

  static EIGEN_DEVICE_FUNC void run(const Self& self, Op& reducer, const GpuDevice& device, float* output) {
    typedef typename Self::Index Index;

    const Index num_coeffs = array_prod(self.m_impl.dimensions());
    const int block_size = 256;
    const int num_per_thread = 128;
    const int num_blocks = std::ceil(static_cast<float>(num_coeffs) / (block_size * num_per_thread));

    if (num_blocks > 1) {
      // We initialize the outputs outside the reduction kernel when we can't be sure that there
      // won't be a race conditions between multiple thread blocks.
      LAUNCH_CUDA_KERNEL((ReductionInitKernel<float, Index>),
                         1, 32, 0, device, reducer.initialize(), 1, output);
    }

    LAUNCH_CUDA_KERNEL((FullReductionKernel<block_size, num_per_thread, Self, Op, Index>),
                       num_blocks, block_size, 0, device, reducer, self, num_coeffs, output);
  }
};


template <int NumPerThread, typename Self,
          typename Reducer, typename Index>
__global__ void InnerReductionKernel(Reducer reducer, const Self input, Index num_coeffs_to_reduce, Index num_preserved_coeffs,
                                         typename Self::CoeffReturnType* output) {
  eigen_assert(blockDim.y == 1);
  eigen_assert(blockDim.z == 1);
  eigen_assert(gridDim.y == 1);
  eigen_assert(gridDim.z == 1);

  const int unroll_times = 16;
  eigen_assert(NumPerThread % unroll_times == 0);

  const Index input_col_blocks = divup<Index>(num_coeffs_to_reduce, blockDim.x * NumPerThread);
  const Index num_input_blocks = input_col_blocks * num_preserved_coeffs;

  const Index num_threads = blockDim.x * gridDim.x;
  const Index thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  // Initialize the output values if they weren't initialized by the ReductionInitKernel
  if (gridDim.x == 1) {
    for (Index i = thread_id; i < num_preserved_coeffs; i += num_threads) {
      output[i] = reducer.initialize();
    }
  }

  for (Index i = blockIdx.x; i < num_input_blocks; i += gridDim.x) {
    const Index row = i / input_col_blocks;

    if (row < num_preserved_coeffs) {
      const Index col_block = i % input_col_blocks;
      const Index col_begin = col_block * blockDim.x * NumPerThread + threadIdx.x;

      float reduced_val = reducer.initialize();

      for (Index j = 0; j < NumPerThread; j += unroll_times) {
        const Index last_col = col_begin + blockDim.x * (j + unroll_times - 1);
        if (last_col >= num_coeffs_to_reduce) {
          for (Index col = col_begin + blockDim.x * j; col < num_coeffs_to_reduce; col +=blockDim.x) {
            const float val = input.m_impl.coeff(row * num_coeffs_to_reduce + col);
            reducer.reduce(val, &reduced_val);
          }
          break;
        } else {
          // Faster version of the loop with no branches after unrolling.
#pragma unroll
          for (int k = 0; k < unroll_times; ++k) {
            const Index col = col_begin + blockDim.x * (j + k);
            reducer.reduce(input.m_impl.coeff(row * num_coeffs_to_reduce + col), &reduced_val);
          }
        }
      }

#pragma unroll
      for (int offset = warpSize/2; offset > 0; offset /= 2) {
        reducer.reduce(__shfl_down(reduced_val, offset), &reduced_val);
      }

      if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicReduce(&(output[row]), reduced_val, reducer);
      }
    }

    __syncthreads();
  }
}

template <typename Self, typename Op>
struct InnerReducer<Self, Op, GpuDevice> {
  // Unfortunately nvidia doesn't support well exotic types such as complex,
  // so reduce the scope of the optimized version of the code to the simple case
  // of floats.
  static const bool HasOptimizedImplementation = !Op::IsStateful &&
                                                 internal::is_same<typename Self::CoeffReturnType, float>::value;

  template <typename Device, typename OutputType>
  static EIGEN_DEVICE_FUNC bool run(const Self&, Op&, const Device&, OutputType*, typename Self::Index, typename Self::Index) {
    assert(false && "Should only be called to reduce floats on a gpu device");
    return true;
  }

  static EIGEN_DEVICE_FUNC bool run(const Self& self, Op& reducer, const GpuDevice& device, float* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
    typedef typename Self::Index Index;

    // It's faster to use the usual code.
    if (num_coeffs_to_reduce <= 32) {
      return true;
    }

    const Index num_coeffs = num_coeffs_to_reduce * num_preserved_vals;
    const int block_size = 256;
    const int num_per_thread = 128;
    const int dyn_blocks = divup<int>(num_coeffs, block_size * num_per_thread);
    const int max_blocks = device.getNumCudaMultiProcessors() *
                           device.maxCudaThreadsPerMultiProcessor() / block_size;
    const int num_blocks = numext::mini<int>(max_blocks, dyn_blocks);

    if (num_blocks > 1) {
      // We initialize the outputs outside the reduction kernel when we can't be sure that there
      // won't be a race conditions between multiple thread blocks.
      const int dyn_blocks = divup<int>(num_preserved_vals, 1024);
      const int max_blocks = device.getNumCudaMultiProcessors() *
                           device.maxCudaThreadsPerMultiProcessor() / 1024;
      const int num_blocks = numext::mini<int>(max_blocks, dyn_blocks);
      LAUNCH_CUDA_KERNEL((ReductionInitKernel<float, Index>),
                         num_blocks, 1024, 0, device, reducer.initialize(),
                         num_preserved_vals, output);
    }

    LAUNCH_CUDA_KERNEL((InnerReductionKernel<num_per_thread, Self, Op, Index>),
                       num_blocks, block_size, 0, device, reducer, self, num_coeffs_to_reduce, num_preserved_vals, output);

    return false;
  }
};


template <int NumPerThread, typename Self,
          typename Reducer, typename Index>
__global__ void OuterReductionKernel(Reducer reducer, const Self input, Index num_coeffs_to_reduce, Index num_preserved_coeffs,
                                     typename Self::CoeffReturnType* output) {
  const Index num_threads = blockDim.x * gridDim.x;
  const Index thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  // Initialize the output values if they weren't initialized by the ReductionInitKernel
  if (gridDim.x == 1) {
    for (Index i = thread_id; i < num_preserved_coeffs; i += num_threads) {
      output[i] = reducer.initialize();
    }
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
                                                 internal::is_same<typename Self::CoeffReturnType, float>::value;

  template <typename Device, typename OutputType>
  static EIGEN_DEVICE_FUNC bool run(const Self&, Op&, const Device&, OutputType*, typename Self::Index, typename Self::Index) {
    assert(false && "Should only be called to reduce floats on a gpu device");
    return true;
  }

  static EIGEN_DEVICE_FUNC bool run(const Self& self, Op& reducer, const GpuDevice& device, float* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
    typedef typename Self::Index Index;

    // It's faster to use the usual code.
    if (num_coeffs_to_reduce <= 32) {
      return true;
    }

     const Index num_coeffs = num_coeffs_to_reduce * num_preserved_vals;
    const int block_size = 256;
    const int num_per_thread = 16;
    const int dyn_blocks = divup<int>(num_coeffs, block_size * num_per_thread);
    const int max_blocks = device.getNumCudaMultiProcessors() *
                           device.maxCudaThreadsPerMultiProcessor() / block_size;
    const int num_blocks = numext::mini<int>(max_blocks, dyn_blocks);

    if (num_blocks > 1) {
      // We initialize the outputs in the reduction kernel itself when we don't have to worry
      // about race conditions between multiple thread blocks.
      const int dyn_blocks = divup<int>(num_preserved_vals, 1024);
      const int max_blocks = device.getNumCudaMultiProcessors() *
                             device.maxCudaThreadsPerMultiProcessor() / 1024;
      const int num_blocks = numext::mini<int>(max_blocks, dyn_blocks);
      LAUNCH_CUDA_KERNEL((ReductionInitKernel<float, Index>),
                         num_blocks, 1024, 0, device, reducer.initialize(),
                         num_preserved_vals, output);
    }

    LAUNCH_CUDA_KERNEL((OuterReductionKernel<num_per_thread, Self, Op, Index>),
                       num_blocks, block_size, 0, device, reducer, self, num_coeffs_to_reduce, num_preserved_vals, output);

    return false;
  }
};

#endif


} // end namespace internal
} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_CUDA_H
