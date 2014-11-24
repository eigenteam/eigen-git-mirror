
project(EigenBlas CXX)

include("../cmake/language_support.cmake")

workaround_9220(Fortran EIGEN_Fortran_COMPILER_WORKS)

if(EIGEN_Fortran_COMPILER_WORKS)
  enable_language(Fortran OPTIONAL)
  if(NOT CMAKE_Fortran_COMPILER)
    set(EIGEN_Fortran_COMPILER_WORKS OFF)
  endif()
endif()

add_custom_target(blas)

set(EigenBlas_SRCS single.cpp double.cpp complex_single.cpp complex_double.cpp xerbla.cpp)

if(EIGEN_USE_F2C_BLAS)
  set(EigenBlas_SRCS ${EigenBlas_SRCS}
    f2c/complexdots.c
    f2c/srotm.c f2c/srotmg.c f2c/drotm.c f2c/drotmg.c
    f2c/lsame.c  f2c/dspmv.c f2c/ssbmv.c
    f2c/chbmv.c  f2c/sspmv.c
    f2c/zhbmv.c  f2c/chpmv.c f2c/dsbmv.c
    f2c/zhpmv.c
    f2c/dtbmv.c f2c/stbmv.c f2c/ctbmv.c f2c/ztbmv.c
    f2c/d_cnjg.c f2c/r_cnjg.c
    )
else()
  if (EIGEN_Fortran_COMPILER_WORKS)
    set(EigenBlas_SRCS ${EigenBlas_SRCS}
      fortran/complexdots.f
      fortran/srotm.f fortran/srotmg.f fortran/drotm.f fortran/drotmg.f
      fortran/lsame.f  fortran/dspmv.f fortran/ssbmv.f
      fortran/chbmv.f  fortran/sspmv.f
      fortran/zhbmv.f  fortran/chpmv.f fortran/dsbmv.f
      fortran/zhpmv.f
      fortran/dtbmv.f fortran/stbmv.f fortran/ctbmv.f fortran/ztbmv.f
      )
  else()
    message(WARNING " No Fortran compiler has been detected, the blas build will be incomplete. Define EIGEN_USE_F2C_BLAS to build BLAS without Fortran")
  endif()
endif()

add_library(eigen_blas_static ${EigenBlas_SRCS})
add_library(eigen_blas SHARED ${EigenBlas_SRCS})

if(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO)
  target_link_libraries(eigen_blas_static ${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO})
  target_link_libraries(eigen_blas        ${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO})
endif()

add_dependencies(blas eigen_blas eigen_blas_static)

install(TARGETS eigen_blas eigen_blas_static
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

if(EIGEN_Fortran_COMPILER_WORKS)

if(EIGEN_LEAVE_TEST_IN_ALL_TARGET)
  add_subdirectory(testing) # can't do EXCLUDE_FROM_ALL here, breaks CTest
else()
  add_subdirectory(testing EXCLUDE_FROM_ALL)
endif()

endif()

