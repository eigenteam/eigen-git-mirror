
add_subdirectory(Eigen)

if(EIGEN_BUILD_TESTS)
  add_subdirectory(test)
endif(EIGEN_BUILD_TESTS)

