# find_package(MKL)
find_package(Eigen2)
if (EIGEN2_FOUND)

  include_directories(${EIGEN2_INCLUDE_DIR})
  btl_add_bench(btl_eigen2_linear main_linear.cpp)
  btl_add_bench(btl_eigen2_vecmat main_vecmat.cpp)
  btl_add_bench(btl_eigen2_matmat main_matmat.cpp)
  btl_add_bench(btl_eigen2_adv main_adv.cpp      )

  btl_add_target_property(btl_eigen2_linear COMPILE_FLAGS "-DBTL_PREFIX=eigen2")
  btl_add_target_property(btl_eigen2_vecmat COMPILE_FLAGS "-DBTL_PREFIX=eigen2")
  btl_add_target_property(btl_eigen2_matmat COMPILE_FLAGS "-DBTL_PREFIX=eigen2")
  btl_add_target_property(btl_eigen2_adv    COMPILE_FLAGS "-DBTL_PREFIX=eigen2")

  option(BTL_BENCH_NOGCCVEC "also bench Eigen explicit vec without GCC's auto vec" OFF)
  if(CMAKE_COMPILER_IS_GNUCXX AND BTL_BENCH_NOGCCVEC)
    btl_add_bench(btl_eigen2_nogccvec_linear main_linear.cpp)
    btl_add_bench(btl_eigen2_nogccvec_vecmat main_vecmat.cpp)
    btl_add_bench(btl_eigen2_nogccvec_matmat main_matmat.cpp)
    btl_add_bench(btl_eigen2_nogccvec_adv    main_adv.cpp   )

    btl_add_target_property(btl_eigen2_nogccvec_linear COMPILE_FLAGS "-fno-tree-vectorize -DBTL_PREFIX=eigen2_nogccvec")
    btl_add_target_property(btl_eigen2_nogccvec_vecmat COMPILE_FLAGS "-fno-tree-vectorize -DBTL_PREFIX=eigen2_nogccvec")
    btl_add_target_property(btl_eigen2_nogccvec_matmat COMPILE_FLAGS "-fno-tree-vectorize -DBTL_PREFIX=eigen2_nogccvec")
    btl_add_target_property(btl_eigen2_nogccvec_adv    COMPILE_FLAGS "-fno-tree-vectorize -DBTL_PREFIX=eigen2_nogccvec")
  endif()


  if(NOT BTL_NOVEC)
    btl_add_bench(btl_eigen2_novec_linear main_linear.cpp)
    btl_add_bench(btl_eigen2_novec_vecmat main_vecmat.cpp)
    btl_add_bench(btl_eigen2_novec_matmat main_matmat.cpp)
    btl_add_bench(btl_eigen2_novec_adv main_adv.cpp      )
    btl_add_target_property(btl_eigen2_novec_linear COMPILE_FLAGS "-DEIGEN_DONT_VECTORIZE -DBTL_PREFIX=eigen2_novec")
    btl_add_target_property(btl_eigen2_novec_vecmat COMPILE_FLAGS "-DEIGEN_DONT_VECTORIZE -DBTL_PREFIX=eigen2_novec")
    btl_add_target_property(btl_eigen2_novec_matmat COMPILE_FLAGS "-DEIGEN_DONT_VECTORIZE -DBTL_PREFIX=eigen2_novec")
    btl_add_target_property(btl_eigen2_novec_adv    COMPILE_FLAGS "-DEIGEN_DONT_VECTORIZE -DBTL_PREFIX=eigen2_novec")

#     if(BUILD_btl_eigen2_adv)
#       target_link_libraries(btl_eigen2_adv ${MKL_LIBRARIES})
#     endif(BUILD_btl_eigen2_adv)

  endif(NOT BTL_NOVEC)

  btl_add_bench(btl_tiny_eigen2 btl_tiny_eigen2.cpp OFF)

  if(NOT BTL_NOVEC)
    btl_add_bench(btl_tiny_eigen2_novec btl_tiny_eigen2.cpp OFF)
    btl_add_target_property(btl_tiny_eigen2_novec    COMPILE_FLAGS "-DBTL_PREFIX=eigen2_tiny")

    if(BUILD_btl_tiny_eigen2_novec)
      btl_add_target_property(btl_tiny_eigen2_novec    COMPILE_FLAGS "-DEIGEN_DONT_VECTORIZE -DBTL_PREFIX=eigen2_tiny_novec")
    endif(BUILD_btl_tiny_eigen2_novec)
  endif(NOT BTL_NOVEC)

endif (EIGEN2_FOUND)
