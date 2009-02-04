

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
    string(STRIP "${ARGV2}" ARGV2_stripped)
    string(LENGTH "${ARGV2_stripped}" ARGV2_stripped_length)
    if(${ARGV2_stripped_length} GREATER 0)
      target_link_libraries(${targetname} ${ARGV2})
    endif(${ARGV2_stripped_length} GREATER 0)
  endif(${ARGC} GREATER 2)

  if(WIN32)
    add_test(${testname} "${targetname}")
  else(WIN32)
    add_test(${testname} "${CMAKE_CURRENT_SOURCE_DIR}/runtest.sh" "${testname}")
  endif(WIN32)

endmacro(ei_add_test)
