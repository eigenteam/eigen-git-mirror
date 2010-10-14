project(EigenBlas CXX Fortran)

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

add_subdirectory(testing)
