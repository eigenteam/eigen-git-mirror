add_custom_target(buildtests_eigen2)
add_custom_target(check_eigen2 COMMAND "ctest")
add_dependencies(check_eigen2 buildtests_eigen2)

add_definitions("-DEIGEN2_SUPPORT_STAGE10_FULL_EIGEN2_API")

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
macro(ei_add_test_eigen2 testname)

  set(targetname test_eigen2_${testname})

  set(filename ${testname}.cpp)
  add_executable(${targetname} ${filename})
  add_dependencies(buildtests_eigen2 ${targetname})

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

endmacro(ei_add_test_eigen2)

enable_testing()

if(TEST_LIB)
  add_definitions("-DEIGEN_EXTERN_INSTANTIATIONS=1")
endif(TEST_LIB)

ei_add_test_eigen2(meta)
ei_add_test_eigen2(sizeof)
ei_add_test_eigen2(dynalloc)
ei_add_test_eigen2(nomalloc)
ei_add_test_eigen2(first_aligned)
ei_add_test_eigen2(mixingtypes)
ei_add_test_eigen2(packetmath)
ei_add_test_eigen2(unalignedassert)
#ei_add_test_eigen2(vectorization_logic)
ei_add_test_eigen2(basicstuff)
ei_add_test_eigen2(linearstructure)
ei_add_test_eigen2(cwiseop)
ei_add_test_eigen2(sum)
ei_add_test_eigen2(product_small)
ei_add_test_eigen2(product_large ${EI_OFLAG})
ei_add_test_eigen2(adjoint)
ei_add_test_eigen2(submatrices)
ei_add_test_eigen2(miscmatrices)
ei_add_test_eigen2(commainitializer)
ei_add_test_eigen2(smallvectors)
ei_add_test_eigen2(map)
ei_add_test_eigen2(array)
ei_add_test_eigen2(triangular)
ei_add_test_eigen2(cholesky " " "${GSL_LIBRARIES}")
ei_add_test_eigen2(lu ${EI_OFLAG})
ei_add_test_eigen2(determinant ${EI_OFLAG})
ei_add_test_eigen2(inverse)
ei_add_test_eigen2(qr)
ei_add_test_eigen2(eigensolver " " "${GSL_LIBRARIES}")
ei_add_test_eigen2(svd)
ei_add_test_eigen2(geometry)
ei_add_test_eigen2(hyperplane)
ei_add_test_eigen2(parametrizedline)
ei_add_test_eigen2(alignedbox)
ei_add_test_eigen2(regression)
ei_add_test_eigen2(stdvector)
ei_add_test_eigen2(newstdvector)
if(QT4_FOUND)
  ei_add_test_eigen2(qtvector " " "${QT_QTCORE_LIBRARY}")
endif(QT4_FOUND)
if(NOT EIGEN_DEFAULT_TO_ROW_MAJOR)
  ei_add_test_eigen2(sparse_vector)
  ei_add_test_eigen2(sparse_basic)
  ei_add_test_eigen2(sparse_solvers " " "${SPARSE_LIBS}")
  ei_add_test_eigen2(sparse_product)
endif()
ei_add_test_eigen2(swap)
ei_add_test_eigen2(visitor)
ei_add_test_eigen2(bug_132)

ei_add_test_eigen2(prec_inverse_4x4 ${EI_OFLAG})
