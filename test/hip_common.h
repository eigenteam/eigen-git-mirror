
#ifndef EIGEN_TEST_HIP_COMMON_H
#define EIGEN_TEST_HIP_COMMON_H

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include <iostream>

#ifndef __HIPCC__
dim3 threadIdx, blockDim, blockIdx;
#endif

template<typename Kernel, typename Input, typename Output>
void run_on_cpu(const Kernel& ker, int n, const Input& in, Output& out)
{
  for(int i=0; i<n; i++)
    ker(i, in.data(), out.data());
}


template<typename Kernel, typename Input, typename Output>
__global__ __attribute__((used))
void run_on_hip_meta_kernel(const Kernel ker, int n, const Input* in, Output* out)
{
  int i = hipThreadIdx_x + hipBlockIdx_x*hipBlockDim_x;
  if(i<n) {
    ker(i, in, out);
  }
}


template<typename Kernel, typename Input, typename Output>
void run_on_hip(const Kernel& ker, int n, const Input& in, Output& out)
{
  typename Input::Scalar*  d_in;
  typename Output::Scalar* d_out;
  std::ptrdiff_t in_bytes  = in.size()  * sizeof(typename Input::Scalar);
  std::ptrdiff_t out_bytes = out.size() * sizeof(typename Output::Scalar);
  
  hipMalloc((void**)(&d_in),  in_bytes);
  hipMalloc((void**)(&d_out), out_bytes);
  
  hipMemcpy(d_in,  in.data(),  in_bytes,  hipMemcpyHostToDevice);
  hipMemcpy(d_out, out.data(), out_bytes, hipMemcpyHostToDevice);
  
  // Simple and non-optimal 1D mapping assuming n is not too large
  // That's only for unit testing!
  dim3 Blocks(128);
  dim3 Grids( (n+int(Blocks.x)-1)/int(Blocks.x) );

  hipDeviceSynchronize();
  hipLaunchKernelGGL(HIP_KERNEL_NAME(run_on_hip_meta_kernel<Kernel,
                                                         typename std::decay<decltype(*d_in)>::type,
                                                         typename std::decay<decltype(*d_out)>::type>), 
                  dim3(Grids), dim3(Blocks), 0, 0, ker, n, d_in, d_out);
  hipDeviceSynchronize();
  
  // check inputs have not been modified
  hipMemcpy(const_cast<typename Input::Scalar*>(in.data()),  d_in,  in_bytes,  hipMemcpyDeviceToHost);
  hipMemcpy(out.data(), d_out, out_bytes, hipMemcpyDeviceToHost);
  
  hipFree(d_in);
  hipFree(d_out);
}


template<typename Kernel, typename Input, typename Output>
void run_and_compare_to_hip(const Kernel& ker, int n, const Input& in, Output& out)
{
  Input  in_ref,  in_hip;
  Output out_ref, out_hip;
  #ifndef __HIP_DEVICE_COMPILE__
  in_ref = in_hip = in;
  out_ref = out_hip = out;
  #endif
  run_on_cpu (ker, n, in_ref,  out_ref);
  run_on_hip(ker, n, in_hip, out_hip);
  #ifndef __HIP_DEVICE_COMPILE__
  VERIFY_IS_APPROX(in_ref, in_hip);
  VERIFY_IS_APPROX(out_ref, out_hip);
  #endif
}


void ei_test_init_hip()
{
  int device = 0;
  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, device);
  std::cout << "HIP device info:\n";
  std::cout << "  name:                        " << deviceProp.name << "\n";
  std::cout << "  capability:                  " << deviceProp.major << "." << deviceProp.minor << "\n";
  std::cout << "  multiProcessorCount:         " << deviceProp.multiProcessorCount << "\n";
  std::cout << "  maxThreadsPerMultiProcessor: " << deviceProp.maxThreadsPerMultiProcessor << "\n";
  std::cout << "  warpSize:                    " << deviceProp.warpSize << "\n";
  std::cout << "  regsPerBlock:                " << deviceProp.regsPerBlock << "\n";
  std::cout << "  concurrentKernels:           " << deviceProp.concurrentKernels << "\n";
  std::cout << "  clockRate:                   " << deviceProp.clockRate << "\n";
  std::cout << "  canMapHostMemory:            " << deviceProp.canMapHostMemory << "\n";
  std::cout << "  computeMode:                 " << deviceProp.computeMode << "\n";
}

#endif // EIGEN_TEST_HIP_COMMON_H
