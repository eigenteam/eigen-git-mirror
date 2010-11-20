
project(EigenBlas CXX)

if((NOT DEFINED EIGEN_Fortran_COMPILER_WORKS) OR EIGEN_Fortran_COMPILER_WORKS)

  enable_language(Fortran OPTIONAL)

  if(CMAKE_Fortran_COMPILER_WORKS)
    set(EIGEN_Fortran_COMPILER_WORKS TRUE CACHE INTERNAL "workaround cmake's enable_language issue")
  else()
    set(EIGEN_Fortran_COMPILER_WORKS FALSE CACHE INTERNAL "workaround cmake's enable_language issue")
  endif()
  
endif()

add_custom_target(blas)

set(EigenBlas_SRCS single.cpp double.cpp complex_single.cpp complex_double.cpp xerbla.cpp)

add_library(eigen_blas_static ${EigenBlas_SRCS})
add_library(eigen_blas SHARED ${EigenBlas_SRCS})

if(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO)
  target_link_libraries(eigen_blas_static ${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO})
  target_link_libraries(eigen_blas        ${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO})
endif()

add_dependencies(blas eigen_blas eigen_blas_static)

install(TARGETS eigen_blas
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)


if(CMAKE_Fortran_COMPILER_WORKS)
  add_subdirectory(testing)
endif(CMAKE_Fortran_COMPILER_WORKS)
