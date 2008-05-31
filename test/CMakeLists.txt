IF(BUILD_TESTS)

IF(CMAKE_COMPILER_IS_GNUCXX)
  IF(CMAKE_SYSTEM_NAME MATCHES Linux)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1 -g2")
  ENDIF(CMAKE_SYSTEM_NAME MATCHES Linux)
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

OPTION(EIGEN_NO_ASSERTION_CHECKING "Disable checking of assertions" OFF)

# similar to SET_TARGET_PROPERTIES but append the property instead of overwritting it
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

  ADD_EXECUTABLE(${targetname} ${testname}.cpp)

  IF(NOT EIGEN_NO_ASSERTION_CHECKING)

    SET_TARGET_PROPERTIES(${targetname} PROPERTIES COMPILE_FLAGS "-fexceptions")
    OPTION(EIGEN_DEBUG_ASSERTS "Enable debuging of assertions" OFF)
    IF(EIGEN_DEBUG_ASSERTS)
      SET_TARGET_PROPERTIES(${targetname} PROPERTIES COMPILE_DEFINITIONS "-DEIGEN_DEBUG_ASSERTS=1")
    ENDIF(EIGEN_DEBUG_ASSERTS)

  ELSE(NOT EIGEN_NO_ASSERTION_CHECKING)

    SET_TARGET_PROPERTIES(${targetname} PROPERTIES COMPILE_DEFINITIONS "-DEIGEN_NO_ASSERTION_CHECKING=1")

  ENDIF(NOT EIGEN_NO_ASSERTION_CHECKING)

  EI_ADD_TARGET_PROPERTY(${targetname} COMPILE_FLAGS "-DEIGEN_TEST_FUNC=${testname}")

  target_link_libraries(${targetname} Eigen2)

  IF(WIN32)
    ADD_TEST(${testname} "${targetname}")
  ELSE(WIN32)
    ADD_TEST(${testname} "${CMAKE_CURRENT_SOURCE_DIR}/runtest.sh" "${testname}")
  ENDIF(WIN32)

ENDMACRO(EI_ADD_TEST)


ENABLE_TESTING()

ADD_DEFINITIONS("-DEIGEN_EXTERN_INSTANCIATION=1")

EI_ADD_TEST(basicstuff)
EI_ADD_TEST(linearstructure)
EI_ADD_TEST(cwiseop)
EI_ADD_TEST(product)
EI_ADD_TEST(adjoint)
EI_ADD_TEST(submatrices)
EI_ADD_TEST(miscmatrices)
EI_ADD_TEST(smallvectors)
EI_ADD_TEST(map)
EI_ADD_TEST(triangular)
EI_ADD_TEST(cholesky)
# EI_ADD_TEST(determinant)
EI_ADD_TEST(qr)
EI_ADD_TEST(eigensolver)

ENDIF(BUILD_TESTS)
