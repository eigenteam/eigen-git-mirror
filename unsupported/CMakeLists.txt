
add_subdirectory(Eigen)

add_subdirectory(doc)

if(EIGEN_BUILD_TESTS)
  add_subdirectory(test)
endif(EIGEN_BUILD_TESTS)

