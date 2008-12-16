project(Eigen)
set(EIGEN_VERSION_NUMBER "2.0-beta3")

#if the svnversion program is absent, this will leave the SVN_REVISION string empty,
#but won't stop CMake.
execute_process(COMMAND svnversion -n ${CMAKE_SOURCE_DIR}
                OUTPUT_VARIABLE EIGEN_SVN_REVISION)

if(EIGEN_SVN_REVISION)
  set(EIGEN_VERSION "${EIGEN_VERSION_NUMBER} (SVN revision ${EIGEN_SVN_REVISION})")
else(EIGEN_SVN_REVISION)
  set(EIGEN_VERSION "${EIGEN_VERSION_NUMBER}")
endif(EIGEN_SVN_REVISION)

cmake_minimum_required(VERSION 2.4)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

option(EIGEN_BUILD_TESTS "Build tests" OFF)
option(EIGEN_BUILD_DEMOS "Build demos" OFF)
if(NOT WIN32)
  option(EIGEN_BUILD_LIB "Build the binary shared library" OFF)
endif(NOT WIN32)
option(EIGEN_BUILD_BTL "Build benchmark suite" OFF)

if(EIGEN_BUILD_LIB)
  option(EIGEN_TEST_LIB "Build the unit tests using the library (disable -pedantic)" OFF)
endif(EIGEN_BUILD_LIB)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(CMAKE_COMPILER_IS_GNUCXX)
  if(CMAKE_SYSTEM_NAME MATCHES Linux)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wnon-virtual-dtor -Wno-long-long -ansi -Wundef -Wcast-align -Wchar-subscripts -Wall -W -Wpointer-arith -Wwrite-strings -Wformat-security -fno-exceptions -fno-check-new -fno-common -fstrict-aliasing")
    if(NOT EIGEN_TEST_LIB)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")
    endif(NOT EIGEN_TEST_LIB)
    if(EIGEN_TEST_SSE2)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2")
      message("Enabling SSE2 in tests/examples")
    endif(EIGEN_TEST_SSE2)
    if(EIGEN_TEST_SSE3)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3")
      message("Enabling SSE3 in tests/examples")
    endif(EIGEN_TEST_SSE3)
    if(EIGEN_TEST_SSSE3)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mssse3")
      message("Enabling SSSE3 in tests/examples")
    endif(EIGEN_TEST_SSSE3)
    if(EIGEN_TEST_ALTIVEC)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -maltivec -mabi=altivec")
      message("Enabling AltiVec in tests/examples")
    endif(EIGEN_TEST_ALTIVEC)
  endif(CMAKE_SYSTEM_NAME MATCHES Linux)
endif(CMAKE_COMPILER_IS_GNUCXX)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(Eigen)

if(EIGEN_BUILD_TESTS)
  add_subdirectory(test)
endif(EIGEN_BUILD_TESTS)

add_subdirectory(doc)

if(EIGEN_BUILD_DEMOS)
  add_subdirectory(demos)
endif(EIGEN_BUILD_DEMOS)

if(EIGEN_BUILD_BTL)
  add_subdirectory(bench/btl)
endif(EIGEN_BUILD_BTL)
