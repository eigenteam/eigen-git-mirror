find_package(Adolc)

include_directories(../../test)

if(ADOLC_FOUND)
  include_directories(${ADOLC_INCLUDES})
  ei_add_property(EIGEN_TESTED_BACKENDS "Adolc")
  ei_add_test(forward_adolc " " ${ADOLC_LIBRARIES})
else(ADOLC_FOUND)
  ei_add_property(EIGEN_MISSING_BACKENDS "Adolc")
endif(ADOLC_FOUND)

ei_add_test(NonLinearOptimization)
ei_add_test(NumericalDiff)
ei_add_test(autodiff)
ei_add_test(BVH)
ei_add_test(matrixExponential)
ei_add_test(alignedvector3)
ei_add_test(FFT)

if(CMAKE_COMPILER_IS_GNUCXX)
  ei_add_test(nesting_profiling "" "-lrt")
else()
  ei_add_test(nesting_profiling)
endif()

find_package(FFTW)
if(FFTW_FOUND)
  ei_add_test(FFTW  "-DEIGEN_FFTW_DEFAULT " "-lfftw3 -lfftw3f -lfftw3l" )
endif(FFTW_FOUND)

ei_add_test(Complex)
