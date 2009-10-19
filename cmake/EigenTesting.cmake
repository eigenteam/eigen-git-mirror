option(EIGEN_NO_ASSERTION_CHECKING "Disable checking of assertions" OFF)

# similar to set_target_properties but append the property instead of overwriting it
macro(ei_add_target_property target prop value)

  get_target_property(previous ${target} ${prop})
  # if the property wasn't previously set, ${previous} is now "previous-NOTFOUND" which cmake allows catching with plain if()
  if(NOT previous)
    set(previous "")
  endif(NOT previous)
  set_target_properties(${target} PROPERTIES ${prop} "${previous} ${value}")
endmacro(ei_add_target_property)

macro(ei_add_property prop value)
  get_property(previous GLOBAL PROPERTY ${prop})
  set_property(GLOBAL PROPERTY ${prop} "${previous} ${value}")
endmacro(ei_add_property)

#internal. See documentation of ei_add_test for details.
macro(ei_add_test_internal testname testname_with_suffix)
  set(targetname test_${testname_with_suffix})
  if(NOT MSVC_IDE)
    set(debug_targetname debug_${testname_with_suffix})
  endif(NOT MSVC_IDE)

  set(filename ${testname}.cpp)
  add_executable(${targetname} ${filename})
  add_dependencies(btest ${targetname})
  if(NOT MSVC_IDE)
    add_executable(${debug_targetname} EXCLUDE_FROM_ALL ${filename})
  endif(NOT MSVC_IDE)

  if(NOT EIGEN_NO_ASSERTION_CHECKING)

    if(MSVC)
      set_target_properties(${targetname} PROPERTIES COMPILE_FLAGS "/EHsc")
      if(NOT MSVC_IDE)
        set_target_properties(${debug_targetname} PROPERTIES COMPILE_FLAGS "/EHsc")
      endif(NOT MSVC_IDE)
    else(MSVC)
      set_target_properties(${targetname} PROPERTIES COMPILE_FLAGS "-fexceptions")
      set_target_properties(${debug_targetname} PROPERTIES COMPILE_FLAGS "-fexceptions")
    endif(MSVC)

    option(EIGEN_DEBUG_ASSERTS "Enable debuging of assertions" OFF)
    if(EIGEN_DEBUG_ASSERTS)
      ei_add_target_property(${targetname} COMPILE_FLAGS "-DEIGEN_DEBUG_ASSERTS=1")
      if(NOT MSVC_IDE)
        ei_add_target_property(${debug_targetname} COMPILE_FLAGS "-DEIGEN_DEBUG_ASSERTS=1")
      endif(NOT MSVC_IDE)
    endif(EIGEN_DEBUG_ASSERTS)

  else(NOT EIGEN_NO_ASSERTION_CHECKING)

    ei_add_target_property(${targetname} COMPILE_FLAGS "-DEIGEN_NO_ASSERTION_CHECKING=1")
    if(NOT MSVC_IDE)
      ei_add_target_property(${debug_targetname} COMPILE_FLAGS "-DEIGEN_NO_ASSERTION_CHECKING=1")
    endif(NOT MSVC_IDE)

  endif(NOT EIGEN_NO_ASSERTION_CHECKING)

  # let the user pass flags. Note that if the user passes an optimization flag here, it's important that
  # we counter it by a no-optimization flag!
  if(${ARGC} GREATER 2)
    ei_add_target_property(${targetname} COMPILE_FLAGS "${ARGV2}")
    if(NOT MSVC_IDE)
      ei_add_target_property(${debug_targetname} COMPILE_FLAGS "${ARGV2} ${EI_NO_OPTIMIZATION_FLAG}")
    endif(NOT MSVC_IDE)
  endif(${ARGC} GREATER 2)

  # for the debug target, add full debug options
  if(CMAKE_COMPILER_IS_GNUCXX)
    # O0 is in principle redundant here, but doesn't hurt
    ei_add_target_property(${debug_targetname} COMPILE_FLAGS "-O0 -g3")
  elseif(MSVC)
    if(NOT MSVC_IDE)
      ei_add_target_property(${debug_targetname} COMPILE_FLAGS "/Od /Zi")
    endif(NOT MSVC_IDE)
  endif(CMAKE_COMPILER_IS_GNUCXX)

  ei_add_target_property(${targetname} COMPILE_FLAGS "-DEIGEN_TEST_FUNC=${testname}")
  if(NOT MSVC_IDE)
    ei_add_target_property(${debug_targetname} COMPILE_FLAGS "-DEIGEN_TEST_FUNC=${testname}")
  endif(NOT MSVC_IDE)

  target_link_libraries(${targetname} ${EXTERNAL_LIBS})
  if(${ARGC} GREATER 3)
    string(STRIP "${ARGV3}" ARGV3_stripped)
    string(LENGTH "${ARGV3_stripped}" ARGV3_stripped_length)
    if(${ARGV3_stripped_length} GREATER 0)
      target_link_libraries(${targetname} ${ARGV3})
      if(NOT MSVC_IDE)
        target_link_libraries(${debug_targetname} ${ARGV3})
      endif(NOT MSVC_IDE)
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
# the unique parameter testname must correspond to a file
# <testname>.cpp which follows this pattern:
#
# #include "main.h"
# void test_<testname>() { ... }
#
# Depending on the contents of that file, this macro can have 2 behaviors.
#
# A. Default behavior
#
# this macro add an executable test_<testname> as well as a ctest test
# named <testname>.
#
# it also adds another executable debug_<testname> that compiles in full debug mode
# and is not added to the test target. The idea is that when a test fails you want
# a quick way of rebuilding this specific test in full debug mode.
#
# On platforms with bash simply run:
#   "ctest -V" or "ctest -V -R <testname>"
# On other platform use ctest as usual
#
# B. Multi-part behavior
#
# If the source file matches the regexp
#    CALL_SUBTEST[0-9]+|EIGEN_TEST_PART_[0-9]+
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
# The same holds for the debug executables.
#
# Moreover, targets test_<testname> and debug_<testname> are still generated, they
# have the effect of building all the parts of the test.
#
# Again, ctest -R allows to run all matching tests.
#
macro(ei_add_test testname)
  file(READ "${testname}.cpp" test_source)
  set(parts 0)
  string(REGEX MATCHALL "CALL_SUBTEST[0-9]+|EIGEN_TEST_PART_[0-9]+" occurences "${test_source}")
  foreach(occurence ${occurences})
    string(REGEX MATCH "([0-9]+)" _number_in_occurence "${occurence}")
    set(number ${CMAKE_MATCH_1})
    if(${number} GREATER ${parts})
      set(parts ${number})
    endif(${number} GREATER ${parts})
  endforeach(occurence)
  if(EIGEN_SPLIT_LARGE_TESTS AND (parts GREATER 0))
    add_custom_target(test_${testname})
    if(NOT MSVC_IDE)
      add_custom_target(debug_${testname})
    endif(NOT MSVC_IDE)
    foreach(part RANGE 1 ${parts})
      ei_add_test_internal(${testname} ${testname}_${part} "${ARGV1} -DEIGEN_TEST_PART_${part}" "${ARGV2}")
      add_dependencies(test_${testname} test_${testname}_${part})
      if(NOT MSVC_IDE)
        add_dependencies(debug_${testname} debug_${testname}_${part})
      endif(NOT MSVC_IDE)
    endforeach(part)
  else(EIGEN_SPLIT_LARGE_TESTS AND (parts GREATER 0))
    set(symbols_to_enable_all_parts "")
    foreach(part RANGE 1 ${parts})
      set(symbols_to_enable_all_parts "${symbols_to_enable_all_parts} -DEIGEN_TEST_PART_${part}")
    endforeach(part)
    ei_add_test_internal(${testname} ${testname} "${ARGV1} ${symbols_to_enable_all_parts}" "${ARGV2}")
  endif(EIGEN_SPLIT_LARGE_TESTS AND (parts GREATER 0))
endmacro(ei_add_test)

# print a summary of the different options
macro(ei_testing_print_summary)

  message("************************************************************")
  message("***    Eigen's unit tests configuration summary          ***")
  message("************************************************************")

  get_property(EIGEN_TESTING_SUMMARY GLOBAL PROPERTY EIGEN_TESTING_SUMMARY)
  get_property(EIGEN_TESTED_BACKENDS GLOBAL PROPERTY EIGEN_TESTED_BACKENDS)
  get_property(EIGEN_MISSING_BACKENDS GLOBAL PROPERTY EIGEN_MISSING_BACKENDS)
  message("Enabled backends:      ${EIGEN_TESTED_BACKENDS}")
  message("Disabled backends:     ${EIGEN_MISSING_BACKENDS}")

  if(EIGEN_TEST_SSE2)
    message("SSE2:              ON")
  else(EIGEN_TEST_SSE2)
    message("SSE2:              Using architecture defaults")
  endif(EIGEN_TEST_SSE2)

  if(EIGEN_TEST_SSE3)
    message("SSE3:              ON")
  else(EIGEN_TEST_SSE3)
    message("SSE3:              Using architecture defaults")
  endif(EIGEN_TEST_SSE3)

  if(EIGEN_TEST_SSSE3)
    message("SSSE3:             ON")
  else(EIGEN_TEST_SSSE3)
    message("SSSE3:             Using architecture defaults")
  endif(EIGEN_TEST_SSSE3)

  if(EIGEN_TEST_ALTIVEC)
    message("Altivec:           Using architecture defaults")
  else(EIGEN_TEST_ALTIVEC)
    message("Altivec:           Using architecture defaults")
  endif(EIGEN_TEST_ALTIVEC)

  if(EIGEN_TEST_NO_EXPLICIT_VECTORIZATION)
    message("Explicit vec:      OFF")
  else(EIGEN_TEST_NO_EXPLICIT_VECTORIZATION)
    message("Explicit vec:      Using architecture defaults")
  endif(EIGEN_TEST_NO_EXPLICIT_VECTORIZATION)

  message("\n${EIGEN_TESTING_SUMMARY}")
  #   message("CXX:               ${CMAKE_CXX_COMPILER}")
  # if(CMAKE_COMPILER_IS_GNUCXX)
  #   execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version COMMAND head -n 1 OUTPUT_VARIABLE EIGEN_CXX_VERSION_STRING OUTPUT_STRIP_TRAILING_WHITESPACE)
  #   message("CXX_VERSION:       ${EIGEN_CXX_VERSION_STRING}")
  # endif(CMAKE_COMPILER_IS_GNUCXX)
  #   message("CXX_FLAGS:         ${CMAKE_CXX_FLAGS}")
  #   message("Sparse lib flags:  ${SPARSE_LIBS}")

  message("************************************************************")

endmacro(ei_testing_print_summary)

macro(ei_init_testing)
  define_property(GLOBAL PROPERTY EIGEN_TESTED_BACKENDS BRIEF_DOCS " " FULL_DOCS " ")
  define_property(GLOBAL PROPERTY EIGEN_MISSING_BACKENDS BRIEF_DOCS " " FULL_DOCS " ")
  define_property(GLOBAL PROPERTY EIGEN_TESTING_SUMMARY BRIEF_DOCS " " FULL_DOCS " ")

  set_property(GLOBAL PROPERTY EIGEN_TESTED_BACKENDS "")
  set_property(GLOBAL PROPERTY EIGEN_MISSING_BACKENDS "")
  set_property(GLOBAL PROPERTY EIGEN_TESTING_SUMMARY "")
endmacro(ei_init_testing)

if(CMAKE_COMPILER_IS_GNUCXX)
  option(EIGEN_COVERAGE_TESTING "Enable/disable gcov" OFF)
  if(EIGEN_COVERAGE_TESTING)
    set(COVERAGE_FLAGS "-fprofile-arcs -ftest-coverage")
  else(EIGEN_COVERAGE_TESTING)
    set(COVERAGE_FLAGS "")
  endif(EIGEN_COVERAGE_TESTING)
  if(EIGEN_TEST_RVALUE_REF_SUPPORT OR EIGEN_TEST_C++0x)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
  endif(EIGEN_TEST_RVALUE_REF_SUPPORT OR EIGEN_TEST_C++0x)
  if(CMAKE_SYSTEM_NAME MATCHES Linux)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COVERAGE_FLAGS} -g2")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${COVERAGE_FLAGS} -O2 -g2")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${COVERAGE_FLAGS} -fno-inline-functions")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${COVERAGE_FLAGS} -O0 -g3")
  endif(CMAKE_SYSTEM_NAME MATCHES Linux)
  set(EI_OFLAG "-O2")
  set(EI_NO_OPTIMIZATION_FLAG "-O0")
elseif(MSVC)
  set(CMAKE_CXX_FLAGS_DEBUG "/D_DEBUG /MDd /Zi /Ob0 /Od" CACHE STRING "Flags used by the compiler during debug builds." FORCE)
  set(EI_OFLAG "/O2")
  set(EI_NO_OPTIMIZATION_FLAG "/O0")
else(CMAKE_COMPILER_IS_GNUCXX)
  set(EI_OFLAG "")
  set(EI_NO_OPTIMIZATION_FLAG "")
endif(CMAKE_COMPILER_IS_GNUCXX)
