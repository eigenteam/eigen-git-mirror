option(EIGEN_NO_ASSERTION_CHECKING "Disable checking of assertions using exceptions" OFF)
option(EIGEN_DEBUG_ASSERTS "Enable advanced debuging of assertions" OFF)

macro(ei_add_property prop value)
  get_property(previous GLOBAL PROPERTY ${prop})
  set_property(GLOBAL PROPERTY ${prop} "${previous} ${value}")
endmacro(ei_add_property)

#internal. See documentation of ei_add_test for details.
macro(ei_add_test_internal testname testname_with_suffix)
  set(targetname ${testname_with_suffix})

  set(filename ${testname}.cpp)
  add_executable(${targetname} ${filename})
  add_dependencies(buildtests ${targetname})

  if(EIGEN_NO_ASSERTION_CHECKING)
    ei_add_target_property(${targetname} COMPILE_FLAGS "-DEIGEN_NO_ASSERTION_CHECKING=1")
  else(EIGEN_NO_ASSERTION_CHECKING)
    if(EIGEN_DEBUG_ASSERTS)
      ei_add_target_property(${targetname} COMPILE_FLAGS "-DEIGEN_DEBUG_ASSERTS=1")
    endif(EIGEN_DEBUG_ASSERTS)
  endif(EIGEN_NO_ASSERTION_CHECKING)

  ei_add_target_property(${targetname} COMPILE_FLAGS "-DEIGEN_TEST_FUNC=${testname}")

  # let the user pass flags.
  if(${ARGC} GREATER 2)
    ei_add_target_property(${targetname} COMPILE_FLAGS "${ARGV2}")
  endif(${ARGC} GREATER 2)

  if(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO)
    target_link_libraries(${targetname} ${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO})
  endif()
  if(EXTERNAL_LIBS)
    target_link_libraries(${targetname} ${EXTERNAL_LIBS})
  endif()
  if(${ARGC} GREATER 3)
    string(STRIP "${ARGV3}" ARGV3_stripped)
    string(LENGTH "${ARGV3_stripped}" ARGV3_stripped_length)
    if(${ARGV3_stripped_length} GREATER 0)
      target_link_libraries(${targetname} ${ARGV3})
    endif(${ARGV3_stripped_length} GREATER 0)
  endif(${ARGC} GREATER 3)

  if(WIN32)
    if(CYGWIN)
      add_test(${testname_with_suffix} "${Eigen_SOURCE_DIR}/test/runtest.sh" "${testname_with_suffix}")
    else(CYGWIN)
      add_test(${testname_with_suffix} "${targetname}")
    endif(CYGWIN)
  else(WIN32)
    add_test(${testname_with_suffix} "${Eigen_SOURCE_DIR}/test/runtest.sh" "${testname_with_suffix}")
  endif(WIN32)

endmacro(ei_add_test_internal)


# Macro to add a test
#
# the unique mandatory parameter testname must correspond to a file
# <testname>.cpp which follows this pattern:
#
# #include "main.h"
# void test_<testname>() { ... }
#
# Depending on the contents of that file, this macro can have 2 behaviors,
# see below.
#
# The optional 2nd parameter is libraries to link to.
#
# A. Default behavior
#
# this macro adds an executable <testname> as well as a ctest test
# named <testname> too.
#
# On platforms with bash simply run:
#   "ctest -V" or "ctest -V -R <testname>"
# On other platform use ctest as usual
#
# B. Multi-part behavior
#
# If the source file matches the regexp
#    CALL_SUBTEST_[0-9]+|EIGEN_TEST_PART_[0-9]+
# then it is interpreted as a multi-part test. The behavior then depends on the
# CMake option EIGEN_SPLIT_LARGE_TESTS, which is ON by default.
#
# If EIGEN_SPLIT_LARGE_TESTS is OFF, the behavior is the same as in A (the multi-part
# aspect is ignored).
#
# If EIGEN_SPLIT_LARGE_TESTS is ON, the test is split into multiple executables
#   test_<testname>_<N>
# where N runs from 1 to the greatest occurence found in the source file. Each of these
# executables is built passing -DEIGEN_TEST_PART_N. This allows to split large tests
# into smaller executables.
#
# Moreover, targets <testname> are still generated, they
# have the effect of building all the parts of the test.
#
# Again, ctest -R allows to run all matching tests.
macro(ei_add_test testname)
  get_property(EIGEN_TESTS_LIST GLOBAL PROPERTY EIGEN_TESTS_LIST)
  set(EIGEN_TESTS_LIST "${EIGEN_TESTS_LIST}${testname}\n")
  set_property(GLOBAL PROPERTY EIGEN_TESTS_LIST "${EIGEN_TESTS_LIST}")

  file(READ "${testname}.cpp" test_source)
  set(parts 0)
  string(REGEX MATCHALL "CALL_SUBTEST_[0-9]+|EIGEN_TEST_PART_[0-9]+"
         occurences "${test_source}")
  string(REGEX REPLACE "CALL_SUBTEST_|EIGEN_TEST_PART_" "" suffixes "${occurences}")
  list(REMOVE_DUPLICATES suffixes)
  if(EIGEN_SPLIT_LARGE_TESTS AND suffixes)
    add_custom_target(${testname})
    foreach(suffix ${suffixes})
      ei_add_test_internal(${testname} ${testname}_${suffix}
        "${ARGV1} -DEIGEN_TEST_PART_${suffix}=1" "${ARGV2}")
      add_dependencies(${testname} ${testname}_${suffix})
    endforeach(suffix)
  else(EIGEN_SPLIT_LARGE_TESTS AND suffixes)
    set(symbols_to_enable_all_parts "")
    foreach(suffix ${suffixes})
      set(symbols_to_enable_all_parts
        "${symbols_to_enable_all_parts} -DEIGEN_TEST_PART_${suffix}=1")
    endforeach(suffix)
    ei_add_test_internal(${testname} ${testname} "${ARGV1} ${symbols_to_enable_all_parts}" "${ARGV2}")
  endif(EIGEN_SPLIT_LARGE_TESTS AND suffixes)
endmacro(ei_add_test)

# print a summary of the different options
macro(ei_testing_print_summary)

  message("************************************************************")
  message("***    Eigen's unit tests configuration summary          ***")
  message("************************************************************")
  message("")
  message("Build type:        ${CMAKE_BUILD_TYPE}")
  get_property(EIGEN_TESTING_SUMMARY GLOBAL PROPERTY EIGEN_TESTING_SUMMARY)
  get_property(EIGEN_TESTED_BACKENDS GLOBAL PROPERTY EIGEN_TESTED_BACKENDS)
  get_property(EIGEN_MISSING_BACKENDS GLOBAL PROPERTY EIGEN_MISSING_BACKENDS)
  message("Enabled backends:  ${EIGEN_TESTED_BACKENDS}")
  message("Disabled backends: ${EIGEN_MISSING_BACKENDS}")

  if(EIGEN_DEFAULT_TO_ROW_MAJOR)
    message("Default order:     Row-major")
  else()
    message("Default order:     Column-major")
  endif()

  if(EIGEN_TEST_NO_EXPLICIT_ALIGNMENT)
    message("Explicit alignment (hence vectorization) disabled")
  elseif(EIGEN_TEST_NO_EXPLICIT_VECTORIZATION)
    message("Explicit vectorization disabled (alignment kept enabled)")
  else()

    if(EIGEN_TEST_SSE2)
      message("SSE2:              ON")
    else()
      message("SSE2:              Using architecture defaults")
    endif()

    if(EIGEN_TEST_SSE3)
      message("SSE3:              ON")
    else()
      message("SSE3:              Using architecture defaults")
    endif()

    if(EIGEN_TEST_SSSE3)
      message("SSSE3:             ON")
    else()
      message("SSSE3:             Using architecture defaults")
    endif()

    if(EIGEN_TEST_SSE4_1)
      message("SSE4.1:            ON")
    else()
      message("SSE4.1:            Using architecture defaults")
    endif()

    if(EIGEN_TEST_SSE4_2)
      message("SSE4.2:            ON")
    else()
      message("SSE4.2:            Using architecture defaults")
    endif()

    if(EIGEN_TEST_ALTIVEC)
      message("Altivec:           ON")
    else()
      message("Altivec:           Using architecture defaults")
    endif()

    if(EIGEN_TEST_NEON)
      message("ARM NEON:          ON")
    else()
      message("ARM NEON:          Using architecture defaults")
    endif()

  endif() # vectorization / alignment options

  message("\n${EIGEN_TESTING_SUMMARY}")

  message("************************************************************")

endmacro(ei_testing_print_summary)

macro(ei_init_testing)
  define_property(GLOBAL PROPERTY EIGEN_TESTED_BACKENDS BRIEF_DOCS " " FULL_DOCS " ")
  define_property(GLOBAL PROPERTY EIGEN_MISSING_BACKENDS BRIEF_DOCS " " FULL_DOCS " ")
  define_property(GLOBAL PROPERTY EIGEN_TESTING_SUMMARY BRIEF_DOCS " " FULL_DOCS " ")
  define_property(GLOBAL PROPERTY EIGEN_TESTS_LIST BRIEF_DOCS " " FULL_DOCS " ")

  set_property(GLOBAL PROPERTY EIGEN_TESTED_BACKENDS "")
  set_property(GLOBAL PROPERTY EIGEN_MISSING_BACKENDS "")
  set_property(GLOBAL PROPERTY EIGEN_TESTING_SUMMARY "")
  set_property(GLOBAL PROPERTY EIGEN_TESTS_LIST "")
endmacro(ei_init_testing)

if(CMAKE_COMPILER_IS_GNUCXX)
  option(EIGEN_COVERAGE_TESTING "Enable/disable gcov" OFF)
  if(EIGEN_COVERAGE_TESTING)
    set(COVERAGE_FLAGS "-fprofile-arcs -ftest-coverage")
    set(CTEST_CUSTOM_COVERAGE_EXCLUDE "/test/")
  else(EIGEN_COVERAGE_TESTING)
    set(COVERAGE_FLAGS "")
  endif(EIGEN_COVERAGE_TESTING)
  if(EIGEN_TEST_C++0x)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x")
  endif(EIGEN_TEST_C++0x)
  if(CMAKE_SYSTEM_NAME MATCHES Linux)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COVERAGE_FLAGS} -g2")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${COVERAGE_FLAGS} -O2 -g2")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${COVERAGE_FLAGS} -fno-inline-functions")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${COVERAGE_FLAGS} -O0 -g3")
  endif(CMAKE_SYSTEM_NAME MATCHES Linux)
elseif(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS /D_SCL_SECURE_NO_WARNINGS")
endif(CMAKE_COMPILER_IS_GNUCXX)
