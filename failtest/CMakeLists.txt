message(STATUS "Running the failtests")

ei_add_failtest("failtest_sanity_check")

ei_add_failtest("block_nonconst_ctor_on_const_xpr_0")
ei_add_failtest("block_nonconst_ctor_on_const_xpr_1")
ei_add_failtest("block_nonconst_ctor_on_const_xpr_2")
ei_add_failtest("transpose_nonconst_ctor_on_const_xpr")
ei_add_failtest("diagonal_nonconst_ctor_on_const_xpr")

ei_add_failtest("const_qualified_block_method_retval_0")
ei_add_failtest("const_qualified_block_method_retval_1")
ei_add_failtest("const_qualified_transpose_method_retval")
ei_add_failtest("const_qualified_diagonal_method_retval")

ei_add_failtest("map_nonconst_ctor_on_const_ptr_0")
ei_add_failtest("map_nonconst_ctor_on_const_ptr_1")
ei_add_failtest("map_nonconst_ctor_on_const_ptr_2")
ei_add_failtest("map_nonconst_ctor_on_const_ptr_3")


if (EIGEN_FAILTEST_FAILURE_COUNT)
  message(FATAL_ERROR "${EIGEN_FAILTEST_FAILURE_COUNT} out of ${EIGEN_FAILTEST_COUNT} failtests FAILED. "
                      "Failtests succeed when they generate build errors. "
                      "To debug these failures, manually compile these programs in ${CMAKE_CURRENT_SOURCE_DIR}.")
else()
  message(STATUS "Failtest SUCCESS: all ${EIGEN_FAILTEST_COUNT} failtests passed.")
  message(STATUS "")
endif()
