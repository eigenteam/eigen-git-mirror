# generate split test header file only if it does not yet exist
# in order to prevent a rebuild everytime cmake is configured
if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/split_test_helper.h)  
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/split_test_helper.h "")
  foreach(i RANGE 1 999)
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/split_test_helper.h
      "#ifdef EIGEN_TEST_PART_${i}\n"
      "#define CALL_SUBTEST_${i}(FUNC) CALL_SUBTEST(FUNC)\n"
      "#else\n"
      "#define CALL_SUBTEST_${i}(FUNC)\n"
      "#endif\n\n"
    )
  endforeach()
endif()

set_property(GLOBAL PROPERTY EIGEN_CURRENT_SUBPROJECT "Unsupported")
add_custom_target(BuildUnsupported)

include_directories(../../test ../../unsupported ../../Eigen 
                    ${CMAKE_CURRENT_BINARY_DIR}/../../test)

find_package (Threads)
                    
find_package(GoogleHash)
if(GOOGLEHASH_FOUND)
  add_definitions("-DEIGEN_GOOGLEHASH_SUPPORT")
  include_directories(${GOOGLEHASH_INCLUDES})
  ei_add_property(EIGEN_TESTED_BACKENDS  "GoogleHash, ")
else(GOOGLEHASH_FOUND)
  ei_add_property(EIGEN_MISSING_BACKENDS  "GoogleHash, ")
endif(GOOGLEHASH_FOUND)

find_package(Adolc)
if(ADOLC_FOUND)
  include_directories(${ADOLC_INCLUDES})
  ei_add_property(EIGEN_TESTED_BACKENDS "Adolc, ")
  ei_add_test(forward_adolc "" ${ADOLC_LIBRARIES})
else(ADOLC_FOUND)
  ei_add_property(EIGEN_MISSING_BACKENDS "Adolc, ")
endif(ADOLC_FOUND)

# this test seems to never have been successful on x87, so is considered to contain a FP-related bug.
# see thread: "non-linear optimization test summary"
ei_add_test(NonLinearOptimization)

ei_add_test(NumericalDiff)
ei_add_test(autodiff_scalar)
ei_add_test(autodiff)

if (NOT CMAKE_CXX_COMPILER MATCHES "clang\\+\\+$")
ei_add_test(BVH)
endif()

ei_add_test(matrix_exponential)
ei_add_test(matrix_function)
ei_add_test(matrix_power)
ei_add_test(matrix_square_root)
ei_add_test(alignedvector3)

ei_add_test(FFT)

find_package(MPFR 2.3.0)
find_package(GMP)
if(MPFR_FOUND AND EIGEN_COMPILER_SUPPORT_CXX11)
  include_directories(${MPFR_INCLUDES} ./mpreal)
  ei_add_property(EIGEN_TESTED_BACKENDS "MPFR C++, ")
  set(EIGEN_MPFR_TEST_LIBRARIES ${MPFR_LIBRARIES} ${GMP_LIBRARIES})
 ei_add_test(mpreal_support "-std=c++11" "${EIGEN_MPFR_TEST_LIBRARIES}" )
else()
  ei_add_property(EIGEN_MISSING_BACKENDS "MPFR C++, ")
endif()

ei_add_test(sparse_extra   "" "")

find_package(FFTW)
if(FFTW_FOUND)
  ei_add_property(EIGEN_TESTED_BACKENDS "fftw, ")
  include_directories( ${FFTW_INCLUDES} )
  if(FFTWL_LIB)
    ei_add_test(FFTW  "-DEIGEN_FFTW_DEFAULT -DEIGEN_HAS_FFTWL" "${FFTW_LIBRARIES}" )
  else()
    ei_add_test(FFTW  "-DEIGEN_FFTW_DEFAULT" "${FFTW_LIBRARIES}" )
  endif()
else()
  ei_add_property(EIGEN_MISSING_BACKENDS "fftw, ")
endif()

option(EIGEN_TEST_NO_OPENGL "Disable OpenGL support in unit tests" OFF)
if(NOT EIGEN_TEST_NO_OPENGL)
  find_package(OpenGL)
  find_package(GLUT)
  find_package(GLEW)
  if(OPENGL_FOUND AND GLUT_FOUND AND GLEW_FOUND)
    include_directories(${OPENGL_INCLUDE_DIR} ${GLUT_INCLUDE_DIR} ${GLEW_INCLUDE_DIRS})
    ei_add_property(EIGEN_TESTED_BACKENDS "OpenGL, ")
    set(EIGEN_GL_LIB ${GLUT_LIBRARIES} ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES})
    ei_add_test(openglsupport  "" "${EIGEN_GL_LIB}" )
  else()
    ei_add_property(EIGEN_MISSING_BACKENDS "OpenGL, ")
  endif()
else()
    ei_add_property(EIGEN_MISSING_BACKENDS "OpenGL, ")
endif()

ei_add_test(polynomialsolver)
ei_add_test(polynomialutils)
ei_add_test(splines)
ei_add_test(gmres)
ei_add_test(minres)
ei_add_test(levenberg_marquardt)
ei_add_test(kronecker_product)

# TODO: The following test names are prefixed with the cxx11 string, since historically
# the tests depended on c++11. This isn't the case anymore so we ought to rename them.
ei_add_test(cxx11_float16)
ei_add_test(cxx11_tensor_dimension)
ei_add_test(cxx11_tensor_map)
ei_add_test(cxx11_tensor_assign)
ei_add_test(cxx11_tensor_comparisons)
ei_add_test(cxx11_tensor_forced_eval)
ei_add_test(cxx11_tensor_math)
ei_add_test(cxx11_tensor_const)
ei_add_test(cxx11_tensor_intdiv)
ei_add_test(cxx11_tensor_casts)
ei_add_test(cxx11_tensor_empty)
ei_add_test(cxx11_tensor_sugar)
ei_add_test(cxx11_tensor_roundings)
ei_add_test(cxx11_tensor_layout_swap)
ei_add_test(cxx11_tensor_io)
if("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
  # This test requires __uint128_t which is only available on 64bit systems 
  ei_add_test(cxx11_tensor_uint128)
endif() 

if(EIGEN_TEST_CXX11)
  # It should be safe to always run these tests as there is some fallback code for
  # older compiler that don't support cxx11.
  set(CMAKE_CXX_STANDARD 11)

  ei_add_test(cxx11_eventcount "-pthread" "${CMAKE_THREAD_LIBS_INIT}")
  ei_add_test(cxx11_runqueue "-pthread" "${CMAKE_THREAD_LIBS_INIT}")
  ei_add_test(cxx11_non_blocking_thread_pool "-pthread" "${CMAKE_THREAD_LIBS_INIT}")

  ei_add_test(cxx11_meta)
  ei_add_test(cxx11_tensor_simple)
#  ei_add_test(cxx11_tensor_symmetry)
  ei_add_test(cxx11_tensor_index_list)
  ei_add_test(cxx11_tensor_mixed_indices)
  ei_add_test(cxx11_tensor_contraction)
  ei_add_test(cxx11_tensor_convolution)
  ei_add_test(cxx11_tensor_expr)
  ei_add_test(cxx11_tensor_fixed_size)
  ei_add_test(cxx11_tensor_of_const_values)
  ei_add_test(cxx11_tensor_of_complex)
  ei_add_test(cxx11_tensor_of_strings)
  ei_add_test(cxx11_tensor_lvalue)
  ei_add_test(cxx11_tensor_broadcasting)
  ei_add_test(cxx11_tensor_chipping)
  ei_add_test(cxx11_tensor_concatenation)
  ei_add_test(cxx11_tensor_inflation)
  ei_add_test(cxx11_tensor_morphing)
  ei_add_test(cxx11_tensor_padding)
  ei_add_test(cxx11_tensor_patch)
  ei_add_test(cxx11_tensor_image_patch)
  ei_add_test(cxx11_tensor_volume_patch)
  ei_add_test(cxx11_tensor_reduction)
  ei_add_test(cxx11_tensor_argmax)
  ei_add_test(cxx11_tensor_shuffling)
  ei_add_test(cxx11_tensor_striding)
  ei_add_test(cxx11_tensor_notification "-pthread" "${CMAKE_THREAD_LIBS_INIT}")
  ei_add_test(cxx11_tensor_thread_pool "-pthread" "${CMAKE_THREAD_LIBS_INIT}")
  ei_add_test(cxx11_tensor_ref)
  ei_add_test(cxx11_tensor_random)
  ei_add_test(cxx11_tensor_generator)
  ei_add_test(cxx11_tensor_custom_op)
  ei_add_test(cxx11_tensor_custom_index)
  ei_add_test(cxx11_tensor_fft)
  ei_add_test(cxx11_tensor_ifft)

endif()

# These tests needs nvcc
find_package(CUDA 7.0)
if(CUDA_FOUND AND EIGEN_TEST_CUDA)
  # Make sure to compile without the -pedantic, -Wundef, -Wnon-virtual-dtor
  # and -fno-check-new flags since they trigger thousands of compilation warnings
  # in the CUDA runtime
  string(REPLACE "-pedantic" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-Wundef" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-Wnon-virtual-dtor" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-fno-check-new" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

  message(STATUS "Flags used to compile cuda code: " ${CMAKE_CXX_FLAGS})

  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CUDA_NVCC_FLAGS "-ccbin /usr/bin/clang" CACHE STRING "nvcc flags" FORCE)
  endif()
  if(EIGEN_TEST_CUDA_CLANG)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 --cuda-gpu-arch=sm_${EIGEN_CUDA_COMPUTE_ARCH}")
  endif()

  set(CUDA_NVCC_FLAGS "-std=c++11 --relaxed-constexpr -arch compute_${EIGEN_CUDA_COMPUTE_ARCH} -Xcudafe \"--display_error_number\"")
  cuda_include_directories("${CMAKE_CURRENT_BINARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}/include")
  set(EIGEN_ADD_TEST_FILENAME_EXTENSION "cu")

  ei_add_test(cxx11_tensor_device)
  ei_add_test(cxx11_tensor_cuda)
  ei_add_test(cxx11_tensor_contract_cuda)
  ei_add_test(cxx11_tensor_reduction_cuda)
  ei_add_test(cxx11_tensor_argmax_cuda)
  ei_add_test(cxx11_tensor_cast_float16_cuda)

  # The random number generation code requires arch 3.5 or greater.
  if (${EIGEN_CUDA_COMPUTE_ARCH} GREATER 34)
    ei_add_test(cxx11_tensor_random_cuda)
  endif()

  ei_add_test(cxx11_tensor_of_float16_cuda)

  unset(EIGEN_ADD_TEST_FILENAME_EXTENSION)
endif()
