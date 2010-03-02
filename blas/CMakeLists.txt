project(EigenBlas)

add_custom_target(blas)

set(EigenBlas_SRCS single.cpp double.cpp complex_single.cpp complex_double.cpp xerbla.cpp)

add_library(eigen_blas ${EigenBlas_SRCS})
# add_library(eigen_blas SHARED ${EigenBlas_SRCS})
add_dependencies(blas eigen_blas)

install(TARGETS eigen_blas
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

