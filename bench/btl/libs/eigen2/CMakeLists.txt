
find_package(Eigen2)
if (EIGEN2_FOUND)

  include_directories(${EIGEN2_INCLUDE_DIR})
  btl_add_bench(btl_eigen2 main.cpp)

  IF(NOT BTL_NOVEC)
    btl_add_bench(btl_eigen2_novec main.cpp)
    if(BUILD_btl_eigen2_novec)
      set_target_properties(btl_eigen2_novec PROPERTIES COMPILE_FLAGS "-DEIGEN_DONT_VECTORIZE")
    endif(BUILD_btl_eigen2_novec)
  ENDIF(NOT BTL_NOVEC)

  btl_add_bench(btl_tiny_eigen2 btl_tiny_eigen2.cpp OFF)

  IF(NOT BTL_NOVEC)
    btl_add_bench(btl_tiny_eigen2_novec btl_tiny_eigen2.cpp OFF)
    if(BUILD_btl_tiny_eigen2_novec)
      set_target_properties(btl_tiny_eigen2_novec PROPERTIES COMPILE_FLAGS "-DEIGEN_DONT_VECTORIZE")
    endif(BUILD_btl_tiny_eigen2_novec)
  ENDIF(NOT BTL_NOVEC)

endif (EIGEN2_FOUND)
