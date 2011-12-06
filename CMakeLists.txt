project(Eigen)
cmake_minimum_required(VERSION 2.6.2)

set(INCLUDE_INSTALL_DIR
    "${CMAKE_INSTALL_PREFIX}/include/eigen2"
    CACHE PATH
    "The directory where we install the header files"
    FORCE)

set(EIGEN_VERSION_NUMBER "2.0.17")
set(EIGEN_VERSION "${EIGEN_VERSION_NUMBER}")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

option(EIGEN_BUILD_TESTS "Build tests" OFF)
option(EIGEN_BUILD_DEMOS "Build demos" OFF)
if(NOT WIN32)
  option(EIGEN_BUILD_LIB "Build the binary shared library" OFF)
endif(NOT WIN32)
option(EIGEN_BUILD_BTL "Build benchmark suite" OFF)
if(NOT WIN32)
  option(EIGEN_BUILD_PKGCONFIG "Build pkg-config .pc file for Eigen" ON)
endif(NOT WIN32)

if(EIGEN_BUILD_LIB)
  option(EIGEN_TEST_LIB "Build the unit tests using the library (disable -pedantic)" OFF)
endif(EIGEN_BUILD_LIB)


#############################################################################
# find how to link to the standard libraries                                #
#############################################################################

find_package(StandardMathLibrary)

set(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO "")

if(NOT STANDARD_MATH_LIBRARY_FOUND)

  message(FATAL_ERROR
    "Can't link to the standard math library. Please report to the Eigen developers, telling them about your platform.")

else()

  if(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO)
    set(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO "${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO} ${STANDARD_MATH_LIBRARY}")
  else()
    set(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO "${STANDARD_MATH_LIBRARY}")
  endif()

endif()

if(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO)
  message(STATUS "Standard libraries to link to explicitly: ${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO}")
else()
  message(STATUS "Standard libraries to link to explicitly: none")
endif()



set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(CMAKE_COMPILER_IS_GNUCXX)
  if(CMAKE_SYSTEM_NAME MATCHES Linux)
    include(CheckCXXCompilerFlag)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wnon-virtual-dtor -Wno-long-long -ansi -Wundef -Wcast-align -Wchar-subscripts -Wall -W -Wpointer-arith -Wwrite-strings -Wformat-security -fno-exceptions -fno-check-new -fno-common -fstrict-aliasing")
    check_cxx_compiler_flag("-Wextra" has_wextra)
    if(has_wextra)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
    endif()
    if(NOT EIGEN_TEST_LIB)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")
    endif(NOT EIGEN_TEST_LIB)

    option(EIGEN_TEST_SSE2 "Enable/Disable SSE2 in tests/examples" OFF)
    if(EIGEN_TEST_SSE2)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2")
      message("Enabling SSE2 in tests/examples")
    endif(EIGEN_TEST_SSE2)

    option(EIGEN_TEST_SSE3 "Enable/Disable SSE3 in tests/examples" OFF)
    if(EIGEN_TEST_SSE3)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3")
      message("Enabling SSE3 in tests/examples")
    endif(EIGEN_TEST_SSE3)

    option(EIGEN_TEST_SSSE3 "Enable/Disable SSSE3 in tests/examples" OFF)
    if(EIGEN_TEST_SSSE3)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mssse3")
      message("Enabling SSSE3 in tests/examples")
    endif(EIGEN_TEST_SSSE3)

    option(EIGEN_TEST_ALTIVEC "Enable/Disable altivec in tests/examples" OFF)
    if(EIGEN_TEST_ALTIVEC)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -maltivec -mabi=altivec")
      message("Enabling AltiVec in tests/examples")
    endif(EIGEN_TEST_ALTIVEC)

  endif(CMAKE_SYSTEM_NAME MATCHES Linux)
endif(CMAKE_COMPILER_IS_GNUCXX)

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")

  option(EIGEN_TEST_SSE2 "Enable/Disable SSE2 in tests/examples" OFF)
  if(EIGEN_TEST_SSE2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:SSE2")
    message("Enabling SSE2 in tests/examples")
  endif(EIGEN_TEST_SSE2)
endif(MSVC)

option(EIGEN_TEST_NO_EXPLICIT_VECTORIZATION "Disable explicit vectorization in tests/examples" OFF)
if(EIGEN_TEST_NO_EXPLICIT_VECTORIZATION)
  add_definitions(-DEIGEN_DONT_VECTORIZE=1)
  message("Disabling vectorization in tests/examples")
endif(EIGEN_TEST_NO_EXPLICIT_VECTORIZATION)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

if(EIGEN_BUILD_PKGCONFIG)
    configure_file(eigen2.pc.in eigen2.pc) # uses INCLUDE_INSTALL_DIR
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/eigen2.pc
        DESTINATION share/pkgconfig
        )
endif(EIGEN_BUILD_PKGCONFIG)

add_subdirectory(Eigen)
add_subdirectory(unsupported)

if(EIGEN_BUILD_TESTS)
  include(CTest)
  add_subdirectory(test)
endif(EIGEN_BUILD_TESTS)

add_subdirectory(doc)

if(EIGEN_BUILD_DEMOS)
  add_subdirectory(demos)
endif(EIGEN_BUILD_DEMOS)

if(EIGEN_BUILD_BTL)
  add_subdirectory(bench/btl)
endif(EIGEN_BUILD_BTL)
