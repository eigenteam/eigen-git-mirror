IF(BUILD_TESTS)

find_package(GSL)
if(GSL_FOUND)
  add_definitions("-DHAS_GSL")
endif(GSL_FOUND)

IF(CMAKE_COMPILER_IS_GNUCXX)
  IF(CMAKE_SYSTEM_NAME MATCHES Linux)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g2")
    SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O2 -g2")
    SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-inline-functions")
    SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g2")
  ENDIF(CMAKE_SYSTEM_NAME MATCHES Linux)
  SET(EI_OFLAG "-O2")
ELSE(CMAKE_COMPILER_IS_GNUCXX)
  SET(EI_OFLAG "")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

OPTION(EIGEN_NO_ASSERTION_CHECKING "Disable checking of assertions" OFF)

# similar to SET_TARGET_PROPERTIES but append the property instead of overwriting it
MACRO(EI_ADD_TARGET_PROPERTY target prop value)

  GET_TARGET_PROPERTY(previous ${target} ${prop})
  SET_TARGET_PROPERTIES(${target} PROPERTIES ${prop} "${previous} ${value}")

ENDMACRO(EI_ADD_TARGET_PROPERTY)

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
MACRO(EI_ADD_TEST testname)

  SET(targetname test_${testname})

  SET(filename ${testname}.cpp)
  ADD_EXECUTABLE(${targetname} ${filename})

  IF(NOT EIGEN_NO_ASSERTION_CHECKING)

    SET_TARGET_PROPERTIES(${targetname} PROPERTIES COMPILE_FLAGS "-fexceptions")
    OPTION(EIGEN_DEBUG_ASSERTS "Enable debuging of assertions" OFF)
    IF(EIGEN_DEBUG_ASSERTS)
      SET_TARGET_PROPERTIES(${targetname} PROPERTIES COMPILE_DEFINITIONS "-DEIGEN_DEBUG_ASSERTS=1")
    ENDIF(EIGEN_DEBUG_ASSERTS)

  ELSE(NOT EIGEN_NO_ASSERTION_CHECKING)

    SET_TARGET_PROPERTIES(${targetname} PROPERTIES COMPILE_DEFINITIONS "-DEIGEN_NO_ASSERTION_CHECKING=1")

  ENDIF(NOT EIGEN_NO_ASSERTION_CHECKING)

  IF(${ARGC} GREATER 1)
    EI_ADD_TARGET_PROPERTY(${targetname} COMPILE_FLAGS "${ARGV1}")
  ENDIF(${ARGC} GREATER 1)

  EI_ADD_TARGET_PROPERTY(${targetname} COMPILE_FLAGS "-DEIGEN_TEST_FUNC=${testname}")

  IF(TEST_LIB)
    target_link_libraries(${targetname} Eigen2)
  ENDIF(TEST_LIB)

  if(GSL_FOUND)
    target_link_libraries(${targetname} ${GSL_LIBRARIES})
  endif(GSL_FOUND)

  IF(WIN32)
    ADD_TEST(${testname} "${targetname}")
  ELSE(WIN32)
    ADD_TEST(${testname} "${CMAKE_CURRENT_SOURCE_DIR}/runtest.sh" "${testname}")
  ENDIF(WIN32)

ENDMACRO(EI_ADD_TEST)


ENABLE_TESTING()

IF(TEST_LIB)
  ADD_DEFINITIONS("-DEIGEN_EXTERN_INSTANTIATIONS=1")
ENDIF(TEST_LIB)

EI_ADD_TEST(sizeof)
EI_ADD_TEST(nomalloc)
EI_ADD_TEST(packetmath)
EI_ADD_TEST(basicstuff)
EI_ADD_TEST(linearstructure)
EI_ADD_TEST(cwiseop)
EI_ADD_TEST(sum)
EI_ADD_TEST(product_small)
EI_ADD_TEST(product_large ${EI_OFLAG})
EI_ADD_TEST(adjoint)
EI_ADD_TEST(submatrices)
EI_ADD_TEST(miscmatrices)
EI_ADD_TEST(commainitializer)
EI_ADD_TEST(smallvectors)
EI_ADD_TEST(map)
EI_ADD_TEST(array)
EI_ADD_TEST(triangular)
EI_ADD_TEST(cholesky)
EI_ADD_TEST(lu ${EI_OFLAG})
EI_ADD_TEST(determinant)
EI_ADD_TEST(inverse)
EI_ADD_TEST(qr)
EI_ADD_TEST(eigensolver)
EI_ADD_TEST(geometry)
EI_ADD_TEST(regression)
EI_ADD_TEST(svd)
EI_ADD_TEST(ioformat)
EI_ADD_TEST(sparse)

ENDIF(BUILD_TESTS)
