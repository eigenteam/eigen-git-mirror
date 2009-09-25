
set(EigenBlas_SRCS single.cpp double.cpp complex_single.cpp complex_double.cpp)

add_library(eigen_blas SHARED ${EigenBlas_SRCS})

install(TARGETS eigen_blas
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

