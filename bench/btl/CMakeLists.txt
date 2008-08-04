PROJECT(BTL)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(MacroOptionalAddSubdirectory)

OPTION(BTL_NOVEC "Disable SSE/Altivec optimizations when possible" OFF)

SET(CMAKE_INCLUDE_CURRENT_DIR ON)

string(REGEX MATCH icpc IS_ICPC ${CMAKE_CXX_COMPILER})
IF(CMAKE_COMPILER_IS_GNUCXX OR IS_ICPC)
  SET(CMAKE_CXX_FLAGS "-g0 -O3 -DNDEBUG")
  SET(CMAKE_Fortran_FLAGS "-g0 -O3 -DNDEBUG")
  IF(NOT BTL_NOVEC)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2")
    SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -msse2")
  ELSE(NOT BTL_NOVEC)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_DONT_VECTORIZE")
  ENDIF(NOT BTL_NOVEC)
ENDIF(CMAKE_COMPILER_IS_GNUCXX OR IS_ICPC)

if(IS_ICPC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fast")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fast")
endif(IS_ICPC)

include_directories(
  ${PROJECT_SOURCE_DIR}/actions
  ${PROJECT_SOURCE_DIR}/generic_bench
  ${PROJECT_SOURCE_DIR}/generic_bench/utils
  ${PROJECT_SOURCE_DIR}/libs/STL)

# find_package(MKL)
# if (MKL_FOUND)
#   add_definitions(-DHAVE_MKL)
#   set(DEFAULT_LIBRARIES ${MKL_LIBRARIES})
# endif (MKL_FOUND)

MACRO(BTL_ADD_BENCH targetname)

  foreach(_current_var ${ARGN})
    set(_last_var ${_current_var})
  endforeach(_current_var)

  set(_sources ${ARGN})
  list(LENGTH _sources _argn_length)

  list(REMOVE_ITEM _sources ON OFF TRUE FALSE)

  list(LENGTH _sources _src_length)

  if (${_argn_length} EQUAL ${_src_length})
    set(_last_var ON)
  endif (${_argn_length} EQUAL ${_src_length})

  OPTION(BUILD_${targetname} "Build benchmark ${targetname}" ${_last_var})

  IF(BUILD_${targetname})
    ADD_EXECUTABLE(${targetname} ${_sources})
    ADD_TEST(${targetname} "${targetname}")
    target_link_libraries(${targetname} ${DEFAULT_LIBRARIES})
  ENDIF(BUILD_${targetname})

ENDMACRO(BTL_ADD_BENCH)

ENABLE_TESTING()

add_subdirectory(libs/eigen2)
add_subdirectory(libs/hand_vec)
add_subdirectory(libs/gmm)
add_subdirectory(libs/mtl4)
add_subdirectory(libs/ublas)
add_subdirectory(libs/blitz)
add_subdirectory(libs/tvmet)
add_subdirectory(libs/C_BLAS)
add_subdirectory(libs/f77)
add_subdirectory(libs/C)
add_subdirectory(libs/STL)
add_subdirectory(libs/STL_algo)

add_subdirectory(data)


