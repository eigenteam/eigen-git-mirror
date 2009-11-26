get_property(EIGEN_TESTS_LIST GLOBAL PROPERTY EIGEN_TESTS_LIST)
configure_file(buildtests.in ${CMAKE_BINARY_DIR}/buildtests)

configure_file(check.in ${CMAKE_BINARY_DIR}/check)
configure_file(debug.in ${CMAKE_BINARY_DIR}/debug)
configure_file(release.in ${CMAKE_BINARY_DIR}/release)
