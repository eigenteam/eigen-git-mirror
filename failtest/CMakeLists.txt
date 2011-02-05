message(STATUS "Running the failtests")

ei_add_failtest("failtest_sanity_check")
ei_add_failtest("block_nonconst_ctor_on_const_xpr_0")
ei_add_failtest("block_nonconst_ctor_on_const_xpr_1")
ei_add_failtest("block_nonconst_ctor_on_const_xpr_2")
ei_add_failtest("transpose_nonconst_ctor_on_const_xpr")
ei_add_failtest("diagonal_nonconst_ctor_on_const_xpr")

if (EIGEN_FAILTEST_FAILURE_COUNT)
  message(FATAL_ERROR "${EIGEN_FAILTEST_FAILURE_COUNT} out of ${EIGEN_FAILTEST_COUNT} failtests FAILED. "
                      "Failtests succeed when they generate build errors. "
                      "To debug these failures, manually compile these programs in ${CMAKE_CURRENT_SOURCE_DIR}.")
else()
  message(STATUS "Failtest SUCCESS: all ${EIGEN_FAILTEST_COUNT} failtests passed.")
  message(STATUS "")
endif()
