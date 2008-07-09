PROJECT(BTL)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(MacroOptionalAddSubdirectory)

OPTION(BTL_NOVEC "Disable SSE/Altivec optimizations when possible" OFF)

SET(CMAKE_INCLUDE_CURRENT_DIR ON)

IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(CMAKE_CXX_FLAGS "-g0 -O3 -DNDEBUG")
  SET(CMAKE_Fortran_FLAGS "-g0 -O3 -DNDEBUG")
  IF(NOT BTL_NOVEC)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2")
    SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -msse2")
  ELSE(NOT BTL_NOVEC)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_DONT_VECTORIZE")
  ENDIF(NOT BTL_NOVEC)
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

include_directories(
  ${PROJECT_SOURCE_DIR}/actions
  ${PROJECT_SOURCE_DIR}/generic_bench
  ${PROJECT_SOURCE_DIR}/generic_bench/utils
  ${PROJECT_SOURCE_DIR}/libs/STL)


MACRO(BTL_ADD_BENCH targetname)
  ADD_EXECUTABLE(${ARGV})
  ADD_TEST(${targetname} "${targetname}")
ENDMACRO(BTL_ADD_BENCH)

ENABLE_TESTING()

# Eigen2
find_package(Eigen2)
if (EIGEN2_FOUND)
  macro_optional_add_subdirectory(libs/eigen2 ON)
  macro_optional_add_subdirectory(libs/tiny_eigen2 OFF)
endif (EIGEN2_FOUND)

# GMM++
find_package(GMM)
if (GMM_FOUND)
  macro_optional_add_subdirectory(libs/gmm ON)
endif (GMM_FOUND)

# Boost
find_package(Boost)
if (Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  macro_optional_add_subdirectory(libs/ublas ON)
endif (Boost_FOUND)

# blitz
find_package(Blitz)
if (BLITZ_FOUND)
  macro_optional_add_subdirectory(libs/blitz ON)
  macro_optional_add_subdirectory(libs/tiny_blitz OFF)
endif (BLITZ_FOUND)

# tvmet
find_package(Tvmet)
if (TVMET_FOUND)
  macro_optional_add_subdirectory(libs/tvmet OFF)
endif (TVMET_FOUND)

# cblas
find_package(CBLAS)
if (CBLAS_FOUND)
  macro_optional_add_subdirectory(libs/C_BLAS ON)
endif (CBLAS_FOUND)


macro_optional_add_subdirectory(libs/f77 ON)
macro_optional_add_subdirectory(libs/C ON)
macro_optional_add_subdirectory(libs/STL ON)
macro_optional_add_subdirectory(libs/STL_algo ON)

add_subdirectory(data)

