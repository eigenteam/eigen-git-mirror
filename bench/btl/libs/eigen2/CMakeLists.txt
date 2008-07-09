
include_directories(${EIGEN2_INCLUDE_DIR})
btl_add_bench(btl_eigen2 main.cpp)

IF(NOT BTL_NOVEC)
  btl_add_bench(btl_eigen2_novec main.cpp)
  set_target_properties(btl_eigen2_novec PROPERTIES COMPILE_FLAGS "-DEIGEN_DONT_VECTORIZE")
ENDIF(NOT BTL_NOVEC)
