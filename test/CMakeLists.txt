
find_package(GSL)
if(GSL_FOUND)
  add_definitions("-DHAS_GSL")
else(GSL_FOUND)
  set(GSL_LIBRARIES " ")
endif(GSL_FOUND)

set(SPARSE_LIBS "")

find_package(Taucs)
if(TAUCS_FOUND)
  add_definitions("-DEIGEN_TAUCS_SUPPORT")
  include_directories(${TAUCS_INCLUDES})
  set(SPARSE_LIBS ${SPARSE_LIBS} ${TAUCS_LIBRARIES})
else(TAUCS_FOUND)
  message("TAUCS not found, this optional backend won't be tested")
endif(TAUCS_FOUND)

find_package(Cholmod)
if(CHOLMOD_FOUND)
  add_definitions("-DEIGEN_CHOLMOD_SUPPORT")
  include_directories(${CHOLMOD_INCLUDES})
  set(SPARSE_LIBS ${SPARSE_LIBS} ${CHOLMOD_LIBRARIES})
else(CHOLMOD_FOUND)
  message("CHOLMOD not found, this optional backend won't be tested")
endif(CHOLMOD_FOUND)

if(NOT MSVC)
  enable_language(Fortran OPTIONAL)
endif(NOT MSVC)

find_package(Umfpack)
if(UMFPACK_FOUND)
  add_definitions("-DEIGEN_UMFPACK_SUPPORT")
  include_directories(${UMFPACK_INCLUDES})
  set(SPARSE_LIBS ${SPARSE_LIBS} ${UMFPACK_LIBRARIES})
else(UMFPACK_FOUND)
  message("UMFPACK not found, this optional backend won't be tested")
endif(UMFPACK_FOUND)

find_package(SuperLU)
if(SUPERLU_FOUND)
  add_definitions("-DEIGEN_SUPERLU_SUPPORT")
  include_directories(${SUPERLU_INCLUDES})
  set(SPARSE_LIBS ${SPARSE_LIBS} ${SUPERLU_LIBRARIES})
else(SUPERLU_FOUND)
  message("SUPERLU not found, this optional backend won't be tested")
endif(SUPERLU_FOUND)

find_package(GoogleHash)
if(GOOGLEHASH_FOUND)
  add_definitions("-DEIGEN_GOOGLEHASH_SUPPORT")
  include_directories(${GOOGLEHASH_INCLUDES})
else(GOOGLEHASH_FOUND)
  message("Google's hash library not found, those map implementations won't be tested")
endif(GOOGLEHASH_FOUND)

if(CMAKE_COMPILER_IS_GNUCXX)
  if(CMAKE_SYSTEM_NAME MATCHES Linux)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g2")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O2 -g2")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-inline-functions")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g2")
  endif(CMAKE_SYSTEM_NAME MATCHES Linux)
  set(EI_OFLAG "-O2")
else(CMAKE_COMPILER_IS_GNUCXX)
  set(EI_OFLAG "")
endif(CMAKE_COMPILER_IS_GNUCXX)

option(EIGEN_NO_ASSERTION_CHECKING "Disable checking of assertions" OFF)

# similar to set_target_properties but append the property instead of overwriting it
macro(ei_add_target_property target prop value)

  get_target_property(previous ${target} ${prop})
  set_target_properties(${target} PROPERTIES ${prop} "${previous} ${value}")

endmacro(ei_add_target_property)

# Macro to add a test
#
# the unique parameter testname must correspond to a file
# <testname>.cpp which follows this pattern:
#
# #include "main.h"
# void test_<testname>() { ... }
#
# this macro add an executable test_<testname> as well as a ctest test
# named <testname>
#
# On platforms with bash simply run:
#   "ctest -V" or "ctest -V -R <testname>"
# On other platform use ctest as usual
#
macro(ei_add_test testname)

  set(targetname test_${testname})

  set(filename ${testname}.cpp)
  add_executable(${targetname} ${filename})

  if(NOT EIGEN_NO_ASSERTION_CHECKING)

    if(MSVC)
      set_target_properties(${targetname} PROPERTIES COMPILE_FLAGS "/EHsc")
    else(MSVC)
      set_target_properties(${targetname} PROPERTIES COMPILE_FLAGS "-fexceptions")
    endif(MSVC)

    option(EIGEN_DEBUG_ASSERTS "Enable debuging of assertions" OFF)
    if(EIGEN_DEBUG_ASSERTS)
      set_target_properties(${targetname} PROPERTIES COMPILE_DEFINITIONS "-DEIGEN_DEBUG_ASSERTS=1")
    endif(EIGEN_DEBUG_ASSERTS)

  else(NOT EIGEN_NO_ASSERTION_CHECKING)

    set_target_properties(${targetname} PROPERTIES COMPILE_DEFINITIONS "-DEIGEN_NO_ASSERTION_CHECKING=1")

  endif(NOT EIGEN_NO_ASSERTION_CHECKING)

  if(${ARGC} GREATER 1)
    ei_add_target_property(${targetname} COMPILE_FLAGS "${ARGV1}")
  endif(${ARGC} GREATER 1)

  ei_add_target_property(${targetname} COMPILE_FLAGS "-DEIGEN_TEST_FUNC=${testname}")

  if(TEST_LIB)
    target_link_libraries(${targetname} Eigen2)
  endif(TEST_LIB)

  target_link_libraries(${targetname} ${EXTERNAL_LIBS})
  if(${ARGC} GREATER 2)
    target_link_libraries(${targetname} ${ARGV2})
  endif(${ARGC} GREATER 2)

  if(WIN32)
    add_test(${testname} "${targetname}")
  else(WIN32)
    add_test(${testname} "${CMAKE_CURRENT_SOURCE_DIR}/runtest.sh" "${testname}")
  endif(WIN32)

endmacro(ei_add_test)


enable_testing()

if(TEST_LIB)
  add_definitions("-DEIGEN_EXTERN_INSTANTIATIONS=1")
endif(TEST_LIB)

ei_add_test(meta)
ei_add_test(sizeof)
ei_add_test(dynalloc)
ei_add_test(nomalloc)
ei_add_test(packetmath)
ei_add_test(basicstuff)
ei_add_test(linearstructure)
ei_add_test(cwiseop)
ei_add_test(sum)
ei_add_test(product_small)
ei_add_test(product_large ${EI_OFLAG})
ei_add_test(adjoint)
ei_add_test(submatrices)
ei_add_test(miscmatrices)
ei_add_test(commainitializer)
ei_add_test(smallvectors)
ei_add_test(map)
ei_add_test(array)
ei_add_test(triangular)
ei_add_test(cholesky " " "${GSL_LIBRARIES}")
ei_add_test(lu ${EI_OFLAG})
ei_add_test(determinant)
ei_add_test(inverse)
ei_add_test(qr)
ei_add_test(eigensolver " " "${GSL_LIBRARIES}")
ei_add_test(svd)
ei_add_test(geometry)
ei_add_test(hyperplane)
ei_add_test(parametrizedline)
ei_add_test(alignedbox)
ei_add_test(regression)
ei_add_test(sparse_basic " " "${SPARSE_LIBS}")
ei_add_test(sparse_solvers " " "${SPARSE_LIBS}")

