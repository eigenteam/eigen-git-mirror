project(Eigen)

cmake_minimum_required(VERSION 2.6.2)

# automatically parse the version number
file(READ "${CMAKE_SOURCE_DIR}/Eigen/src/Core/util/Macros.h" _eigen2_version_header LIMIT 5000 OFFSET 1000)
string(REGEX MATCH "define *EIGEN_WORLD_VERSION ([0-9]*)" _eigen2_world_version_match "${_eigen2_version_header}")
set(EIGEN2_WORLD_VERSION "${CMAKE_MATCH_1}")
string(REGEX MATCH "define *EIGEN_MAJOR_VERSION ([0-9]*)" _eigen2_major_version_match "${_eigen2_version_header}")
set(EIGEN2_MAJOR_VERSION "${CMAKE_MATCH_1}")
string(REGEX MATCH "define *EIGEN_MINOR_VERSION ([0-9]*)" _eigen2_minor_version_match "${_eigen2_version_header}")
set(EIGEN2_MINOR_VERSION "${CMAKE_MATCH_1}")
set(EIGEN_VERSION_NUMBER ${EIGEN2_WORLD_VERSION}.${EIGEN2_MAJOR_VERSION}.${EIGEN2_MINOR_VERSION})

# if the mercurial program is absent, this will leave the EIGEN_HG_CHANGESET string empty,
# but won't stop CMake.
execute_process(COMMAND hg tip -R ${CMAKE_SOURCE_DIR} OUTPUT_VARIABLE EIGEN_HGTIP_OUTPUT)
execute_process(COMMAND hg branch -R ${CMAKE_SOURCE_DIR} OUTPUT_VARIABLE EIGEN_BRANCH_OUTPUT)

# if this is the default (aka development) branch, extract the mercurial changeset number from the hg tip output...
if(EIGEN_BRANCH_OUTPUT MATCHES "default")
string(REGEX MATCH "^changeset: *[0-9]*:([0-9;a-f]+).*" EIGEN_HG_CHANGESET_MATCH "${EIGEN_HGTIP_OUTPUT}")
set(EIGEN_HG_CHANGESET "${CMAKE_MATCH_1}")
endif(EIGEN_BRANCH_OUTPUT MATCHES "default")
#...and show it next to the version number
if(EIGEN_HG_CHANGESET)
  set(EIGEN_VERSION "${EIGEN_VERSION_NUMBER} (mercurial changeset ${EIGEN_HG_CHANGESET})")
else(EIGEN_HG_CHANGESET)
  set(EIGEN_VERSION "${EIGEN_VERSION_NUMBER}")
endif(EIGEN_HG_CHANGESET)

include(CheckCXXCompilerFlag)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

option(EIGEN_BUILD_BTL "Build benchmark suite" OFF)
if(NOT WIN32)
  option(EIGEN_BUILD_PKGCONFIG "Build pkg-config .pc file for Eigen" ON)
endif(NOT WIN32)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(CMAKE_COMPILER_IS_GNUCXX)
  if(CMAKE_SYSTEM_NAME MATCHES Linux)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wnon-virtual-dtor -Wno-long-long -ansi -Wundef -Wcast-align -Wchar-subscripts -Wall -W -Wpointer-arith -Wwrite-strings -Wformat-security -fno-exceptions -fno-check-new -fno-common -fstrict-aliasing")

    check_cxx_compiler_flag("-Wextra" COMPILER_SUPPORT_WEXTRA)
    if(COMPILER_SUPPORT_WEXTRA)
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
    if(NOT CMAKE_CL_64)
      # arch is not supported on 64 bit systems, SSE is enabled automatically.
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:SSE2")
    endif(NOT CMAKE_CL_64)
    message("Enabling SSE2 in tests/examples")
  endif(EIGEN_TEST_SSE2)
endif(MSVC)

option(EIGEN_TEST_NO_EXPLICIT_VECTORIZATION "Disable explicit vectorization in tests/examples" OFF)
if(EIGEN_TEST_NO_EXPLICIT_VECTORIZATION)
  add_definitions(-DEIGEN_DONT_VECTORIZE=1)
  message("Disabling vectorization in tests/examples")
endif(EIGEN_TEST_NO_EXPLICIT_VECTORIZATION)

option(EIGEN_TEST_C++0x "Enables all C++0x features." OFF)

option(EIGEN_TEST_RVALUE_REF_SUPPORT "Enable rvalue references for unit tests." OFF)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

set(INCLUDE_INSTALL_DIR
    "${CMAKE_INSTALL_PREFIX}/include/eigen2"
    CACHE PATH
    "The directory where we install the header files"
)

if(EIGEN_BUILD_PKGCONFIG)
    configure_file(eigen2.pc.in eigen2.pc)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/eigen2.pc
        DESTINATION lib/pkgconfig
        )
endif(EIGEN_BUILD_PKGCONFIG)

add_subdirectory(Eigen)

add_subdirectory(doc EXCLUDE_FROM_ALL)

include(CTest)
enable_testing() # must be called from the root CMakeLists, see man page
add_subdirectory(test EXCLUDE_FROM_ALL)

add_subdirectory(unsupported)

add_subdirectory(demos EXCLUDE_FROM_ALL)

add_subdirectory(blas EXCLUDE_FROM_ALL)

# TODO: consider also replacing EIGEN_BUILD_BTL by a custom target "make btl"?
if(EIGEN_BUILD_BTL)
  add_subdirectory(bench/btl EXCLUDE_FROM_ALL)
endif(EIGEN_BUILD_BTL)

ei_testing_print_summary()
message("You can now do the following:")
message("-------------+-----------------------------------------------------------------")
message("Command      |   Description")
message("-------------+-----------------------------------------------------------------")
message("make install | Install Eigen to ${CMAKE_INSTALL_PREFIX}")
message("             | Do:  cmake . -DCMAKE_INSTALL_PREFIX=yourprefix   to change that")
message("make test    | Build and run the unit tests (using CTest)")
message("             | Note: this takes lots of time & memory! Easy on the -j option!")
message("make btest   | Only build tests, don't run them")
message("make doc     | Generate the API documentation, requires Doxygen & LaTeX")
message("make blas    | Build BLAS library")
message("-------------+-----------------------------------------------------------------")
