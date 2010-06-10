option(EIGEN_DEFAULT_TO_ROW_MAJOR "Use row-major as default matrix storage order" OFF)
if(EIGEN_DEFAULT_TO_ROW_MAJOR)
  add_definitions("-DEIGEN_DEFAULT_TO_ROW_MAJOR")
endif()

find_package(GSL)
if(GSL_FOUND AND GSL_VERSION_MINOR LESS 9)
  set(GSL_FOUND "")
endif(GSL_FOUND AND GSL_VERSION_MINOR LESS 9)
if(GSL_FOUND)
  add_definitions("-DHAS_GSL" ${GSL_DEFINITIONS})
  include_directories(${GSL_INCLUDE_DIR})
else(GSL_FOUND)
  set(GSL_LIBRARIES " ")
endif(GSL_FOUND)

set(SPARSE_LIBS "")
set(EIGEN_TESTED_BACKENDS "")
set(EIGEN_MISSING_BACKENDS "")

find_package(Taucs)
if(TAUCS_FOUND)
  add_definitions("-DEIGEN_TAUCS_SUPPORT")
  include_directories(${TAUCS_INCLUDES})
  set(SPARSE_LIBS ${SPARSE_LIBS} ${TAUCS_LIBRARIES})
  set(EIGEN_TESTED_BACKENDS ${EIGEN_TESTED_BACKENDS} Taucs)
else(TAUCS_FOUND)
  set(EIGEN_MISSING_BACKENDS ${EIGEN_MISSING_BACKENDS} Taucs)
endif(TAUCS_FOUND)

find_package(Cholmod)
if(CHOLMOD_FOUND)
  add_definitions("-DEIGEN_CHOLMOD_SUPPORT")
  include_directories(${CHOLMOD_INCLUDES})
  set(SPARSE_LIBS ${SPARSE_LIBS} ${CHOLMOD_LIBRARIES})
  set(EIGEN_TESTED_BACKENDS ${EIGEN_TESTED_BACKENDS} Cholmod)
else(CHOLMOD_FOUND)
  set(EIGEN_MISSING_BACKENDS ${EIGEN_MISSING_BACKENDS} Cholmod)
endif(CHOLMOD_FOUND)

option(EIGEN_TEST_NO_FORTRAN "Disable Fortran" ON)
if(NOT MSVC AND NOT EIGEN_TEST_NO_FORTRAN)
  enable_language(Fortran OPTIONAL)
endif(NOT MSVC AND NOT EIGEN_TEST_NO_FORTRAN)

find_package(Umfpack)
if(UMFPACK_FOUND)
  add_definitions("-DEIGEN_UMFPACK_SUPPORT")
  include_directories(${UMFPACK_INCLUDES})
  set(SPARSE_LIBS ${SPARSE_LIBS} ${UMFPACK_LIBRARIES})
  set(EIGEN_TESTED_BACKENDS ${EIGEN_TESTED_BACKENDS} UmfPack)
else(UMFPACK_FOUND)
  set(EIGEN_MISSING_BACKENDS ${EIGEN_MISSING_BACKENDS} UmfPack)
endif(UMFPACK_FOUND)

find_package(SuperLU)
if(SUPERLU_FOUND)
  add_definitions("-DEIGEN_SUPERLU_SUPPORT")
  include_directories(${SUPERLU_INCLUDES})
  set(SPARSE_LIBS ${SPARSE_LIBS} ${SUPERLU_LIBRARIES})
  set(EIGEN_TESTED_BACKENDS ${EIGEN_TESTED_BACKENDS} SuperLU)
else(SUPERLU_FOUND)
  set(EIGEN_MISSING_BACKENDS ${EIGEN_MISSING_BACKENDS} SuperLU)
endif(SUPERLU_FOUND)

find_package(GoogleHash)
if(GOOGLEHASH_FOUND)
  add_definitions("-DEIGEN_GOOGLEHASH_SUPPORT")
  include_directories(${GOOGLEHASH_INCLUDES})
  set(EIGEN_TESTED_BACKENDS ${EIGEN_TESTED_BACKENDS} GoogleHash)
else(GOOGLEHASH_FOUND)
  set(EIGEN_MISSING_BACKENDS ${EIGEN_MISSING_BACKENDS} GoogleHash)
endif(GOOGLEHASH_FOUND)

option(EIGEN_TEST_NOQT "Disable Qt support in unit tests" OFF)
if(NOT EIGEN_TEST_NOQT)
  find_package(Qt4)
endif(NOT EIGEN_TEST_NOQT)
if(QT4_FOUND)
  include(${QT_USE_FILE})
  set(EIGEN_TESTED_BACKENDS ${EIGEN_TESTED_BACKENDS} "Qt4 support")
else(QT4_FOUND)
  set(EIGEN_MISSING_BACKENDS ${EIGEN_MISSING_BACKENDS} "Qt4 support")
endif(QT4_FOUND)


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
if(EIGEN_NO_ASSERTION_CHECKING)
  add_definitions("-DEIGEN_NO_ASSERTION_CHECKING=1")
endif()


# similar to set_target_properties but append the property instead of overwriting it
macro(ei_add_target_property target prop value)

  get_target_property(previous ${target} ${prop})
  if(previous MATCHES "NOTFOUND")
    set_target_properties(${target} PROPERTIES ${prop} "${value}")
  else()
    set_target_properties(${target} PROPERTIES ${prop} "${previous} ${value}")
  endif()

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
      set_target_properties(${targetname} PROPERTIES COMPILE_DEFINITIONS "EIGEN_DEBUG_ASSERTS=1")
    endif(EIGEN_DEBUG_ASSERTS)

  endif(NOT EIGEN_NO_ASSERTION_CHECKING)

  if(${ARGC} GREATER 1)
    ei_add_target_property(${targetname} COMPILE_FLAGS "${ARGV1}")
  endif(${ARGC} GREATER 1)

  ei_add_target_property(${targetname} COMPILE_FLAGS "-DEIGEN_TEST_FUNC=${testname}")

  if(TEST_LIB)
    target_link_libraries(${targetname} Eigen2)
  endif(TEST_LIB)

  if(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO)
    target_link_libraries(${targetname} ${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO})
  endif()

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

enable_testing()

if(TEST_LIB)
  add_definitions("-DEIGEN_EXTERN_INSTANTIATIONS=1")
endif(TEST_LIB)

ei_add_test(meta)
ei_add_test(sizeof)
ei_add_test(dynalloc)
ei_add_test(nomalloc)
ei_add_test(first_aligned)
ei_add_test(mixingtypes)
ei_add_test(packetmath)
ei_add_test(unalignedassert)
ei_add_test(vectorization_logic)
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
ei_add_test(determinant ${EI_OFLAG})
ei_add_test(inverse)
ei_add_test(qr)
ei_add_test(eigensolver " " "${GSL_LIBRARIES}")
ei_add_test(svd)
ei_add_test(geometry)
ei_add_test(hyperplane)
ei_add_test(parametrizedline)
ei_add_test(alignedbox)
ei_add_test(regression)
ei_add_test(stdvector)
ei_add_test(newstdvector)
if(QT4_FOUND)
  ei_add_test(qtvector " " "${QT_QTCORE_LIBRARY}")
endif(QT4_FOUND)
if(NOT EIGEN_DEFAULT_TO_ROW_MAJOR)
  ei_add_test(sparse_vector)
  ei_add_test(sparse_basic)
  ei_add_test(sparse_solvers " " "${SPARSE_LIBS}")
  ei_add_test(sparse_product)
endif()
ei_add_test(swap)
ei_add_test(visitor)
ei_add_test(bug_132)

ei_add_test(prec_inverse_4x4 ${EI_OFLAG})

# print a summary of the different options
message("************************************************************")
message("***    Eigen's unit tests configuration summary          ***")
message("************************************************************")
if(GSL_FOUND)
  message("Comparison with GSL:   ON")
else(GSL_FOUND)
  message("Comparison with GSL:   OFF")
endif(GSL_FOUND)

  message("Enabled backends:      ${EIGEN_TESTED_BACKENDS}")
  message("Disabled backends:     ${EIGEN_MISSING_BACKENDS}")

if(EIGEN_TEST_SSE2)
  message("SSE2:              ON")
else(EIGEN_TEST_SSE2)
  message("SSE2:              AUTO")
endif(EIGEN_TEST_SSE2)

if(EIGEN_TEST_SSE3)
  message("SSE3:              ON")
else(EIGEN_TEST_SSE3)
  message("SSE3:              AUTO")
endif(EIGEN_TEST_SSE3)

if(EIGEN_TEST_SSSE3)
  message("SSSE3:             ON")
else(EIGEN_TEST_SSSE3)
  message("SSSE3:             AUTO")
endif(EIGEN_TEST_SSSE3)

if(EIGEN_TEST_ALTIVEC)
  message("Altivec:           ON")
else(EIGEN_TEST_ALTIVEC)
  message("Altivec:           AUTO")
endif(EIGEN_TEST_ALTIVEC)

if(EIGEN_TEST_NO_EXPLICIT_VECTORIZATION)
  message("Explicit vec:      OFF")
else(EIGEN_TEST_NO_EXPLICIT_VECTORIZATION)
  message("Explicit vec:      AUTO")
endif(EIGEN_TEST_NO_EXPLICIT_VECTORIZATION)

if(EIGEN_DEFAULT_TO_ROW_MAJOR)
  message("Default order:     Row-major")
else()
  message("Default order:     Column-major")
endif()

string(TOLOWER "${CMAKE_CXX_COMPILER}" cmake_cxx_compiler_tolower)
if(cmake_cxx_compiler_tolower MATCHES "qcc")
  set(CXX_IS_QCC "ON")
endif()

  message("CXX:               ${CMAKE_CXX_COMPILER}")
if(CMAKE_COMPILER_IS_GNUCXX AND NOT CXX_IS_QCC)
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version COMMAND head -n 1 OUTPUT_VARIABLE EIGEN_CXX_VERSION_STRING OUTPUT_STRIP_TRAILING_WHITESPACE)
  message("CXX_VERSION:       ${EIGEN_CXX_VERSION_STRING}")
endif()
  message("CXX_FLAGS:         ${CMAKE_CXX_FLAGS}")
  message("Sparse lib flags:  ${SPARSE_LIBS}")

message("************************************************************")
