find_package(Eigen2)
if (EIGEN2_FOUND)

  include_directories(${EIGEN2_INCLUDE_DIR} ${PROJECT_SOURCE_DIR}/libs/f77)
  btl_add_bench(btl_hand_vec main.cpp)

  btl_add_bench(btl_hand_peeling main.cpp)
  if (BUILD_btl_hand_peeling)
    set_target_properties(btl_hand_peeling PROPERTIES COMPILE_FLAGS "-DPEELING")
  endif (BUILD_btl_hand_peeling)

endif (EIGEN2_FOUND)