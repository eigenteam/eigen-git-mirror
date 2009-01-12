
####################################################################
#
# Usage: 
#  - create a new folder, let's call it cdash
#  - in that folder, do:
#    ctest -S path/to/eigen2/test/testsuite.cmake[,option1=value1[,option2=value2]]
#
# Options:
#  - EIGEN_CXX: compiler, eg.: g++-4.2
#      default: default c++ compiler
#  - EIGEN_SITE: eg, INRIA-Bdx_pc-gael, or the name of the contributor, etc.
#      default: hostname
#  - EIGEN_BUILD_STRING: a string which identify the system/compiler. It should be formed like that:
#        <OS_name>-<OS_version>-<arch>-<compiler-version>
#      with:
#        <OS_name> = opensuse, debian, osx, windows, cygwin, freebsd, solaris, etc.
#        <OS_version> = 11.1, XP, vista, leopard, etc.
#        <arch> = i386, x86_64, ia64, powerpc, etc.
#        <compiler-version> = gcc-4.3.2, icc-11.0, MSVC-2008, etc.
#  - EIGEN_EXPLICIT_VECTORIZATION: novec, SSE2, Altivec
#      default: SSE2 for x86_64 systems, novec otherwise
#  - EIGEN_CMAKE_DIR: path to cmake executable
#  - EIGEN_MODE: dashboard model, can be Experimental, Nightly, or Continuous
#      default: Nightly
#  - EIGEN_WORK_DIRECTORY: directory used to download the source files and make the builds
#      default: folder which contains this script
#  - CTEST_SOURCE_DIRECTORY: path to eigen's src (use a new and empty folder, not the one you are working on)
#      default: <EIGEN_WORK_DIRECTORY>/src
#  - CTEST_BINARY_DIRECTORY: build directory
#      default: <EIGEN_WORK_DIRECTORY>/nightly-<EIGEN_CXX>
#
# Here is an example running several compilers on a linux system:
# #!/bin/bash
# EIGEN_ARCH=`uname -m`
# EIGEN_SITE=`hostname`
# EIGEN_OS_VERSION=opensuse-11.1
# COMMON=/home/gael/Coding/eigen2/test/testsuite.cmake,EIGEN_WORK_DIRECTORY=/home/gael/Coding/eigen2/cdash,EIGEN_SITE=$EIGEN_SITE,EIGEN_MODE=$1,EIGEN_BUILD_STRING=$EIGEN_OS_VERSION-$EIGEN_ARCH
# ctest -S $COMMON-gcc-3.4.6,EIGEN_CXX=g++-3.4
# ctest -S $COMMON-gcc-4.0.1,EIGEN_CXX=g++-4.0.1
# ctest -S $COMMON-gcc-4.3.2,EIGEN_CXX=g++-4.3
# ctest -S $COMMON-icc-11.0,EIGEN_CXX=icpc
#
####################################################################

# process the arguments

set(ARGLIST ${CTEST_SCRIPT_ARG})
while(${ARGLIST} MATCHES  ".+.*")

  # pick first
  string(REGEX MATCH "([^,]*)(,.*)?" DUMMY ${ARGLIST})
  SET(TOP ${CMAKE_MATCH_1})
  
  # remove first
  string(REGEX MATCHALL "[^,]*,(.*)" DUMMY ${ARGLIST})
  SET(ARGLIST ${CMAKE_MATCH_1})
  
  # decompose as a pair key=value
  string(REGEX MATCH "([^=]*)(=.*)?" DUMMY ${TOP})
  SET(KEY ${CMAKE_MATCH_1})
  
  string(REGEX MATCH "[^=]*=(.*)" DUMMY ${TOP})
  SET(VALUE ${CMAKE_MATCH_1})
  
  # set the variable to the specified value
  if(VALUE)
    SET(${KEY} ${VALUE})
  else(VALUE)
    SET(${KEY} ON)
  endif(VALUE)
  
endwhile(${ARGLIST} MATCHES ".+.*")

####################################################################
# Automatically set some user variables if they have not been defined manually
####################################################################
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

if(NOT EIGEN_SITE)
  site_name(EIGEN_SITE)
endif(NOT EIGEN_SITE)

if(NOT EIGEN_CMAKE_DIR)
  if(WIN32)
    SET(EIGEN_CMAKE_DIR "C:/Program Files/CMake/bin/")
  else(WIN32)
    SET(EIGEN_CMAKE_DIR "")
  endif(WIN32)
endif(NOT EIGEN_CMAKE_DIR)

if(NOT EIGEN_BUILD_STRING)

  # let's try to find all information we need to make the build string ourself

  # OS
#   if(CYGWIN)
#     SET(EIGEN_OS_VERSION cygwin)
#   elseif(WIN32)
#     SET(EIGEN_OS_VERSION windows)
#   elseif(UNIX)
#     SET(EIGEN_OS_VERSION unix)
#   elseif(APPLE)
#     SET(EIGEN_OS_VERSION osx)
#   else(CYGWIN)
    build_name(EIGEN_OS_VERSION)
#   endif(CYGWIN)

  # arch
  set(EIGEN_ARCH ${CMAKE_SYSTEM_PROCESSOR})
  if(WIN32)
    set(EIGEN_ARCH $ENV{PROCESSOR_ARCHITECTURE})
  else(WIN32)
    execute_process(COMMAND uname -m OUTPUT_VARIABLE EIGEN_ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif(WIN32)

  set(EIGEN_BUILD_STRING ${EIGEN_OS_VERSION}${EIGEN_ARCH}-${EIGEN_CXX})

endif(NOT EIGEN_BUILD_STRING)

if(DEFINED EIGEN_EXPLICIT_VECTORIZATION)
  set(EIGEN_BUILD_STRING ${EIGEN_BUILD_STRING}-${EIGEN_EXPLICIT_VECTORIZATION})
endif(DEFINED EIGEN_EXPLICIT_VECTORIZATION)

if(NOT EIGEN_WORK_DIRECTORY)
  set(EIGEN_WORK_DIRECTORY ${CTEST_SCRIPT_DIRECTORY})
endif(NOT EIGEN_WORK_DIRECTORY)

if(NOT CTEST_SOURCE_DIRECTORY)
  SET (CTEST_SOURCE_DIRECTORY "${EIGEN_WORK_DIRECTORY}/src")
endif(NOT CTEST_SOURCE_DIRECTORY)

if(NOT CTEST_BINARY_DIRECTORY)
  SET (CTEST_BINARY_DIRECTORY "${EIGEN_WORK_DIRECTORY}/nightly_${EIGEN_CXX}")
endif(NOT CTEST_BINARY_DIRECTORY)

if(NOT EIGEN_MODE)
  set(EIGEN_MODE Nightly)
endif(NOT EIGEN_MODE)

## mandatory variables (the default should be ok in most cases):

SET (CTEST_CVS_COMMAND "svn")
SET (CTEST_CVS_CHECKOUT "${CTEST_CVS_COMMAND} co svn://anonsvn.kde.org/home/kde/trunk/kdesupport/eigen2 \"${CTEST_SOURCE_DIRECTORY}\"")

# which ctest command to use for running the dashboard
SET (CTEST_COMMAND "${EIGEN_CMAKE_DIR}ctest -D ${EIGEN_MODE}")
# what cmake command to use for configuring this dashboard
SET (CTEST_CMAKE_COMMAND "${EIGEN_CMAKE_DIR}cmake -DEIGEN_BUILD_TESTS=on ")

####################################################################
# The values in this section are optional you can either
# have them or leave them commented out
####################################################################

# this make sure we get consistent outputs
SET($ENV{LC_MESSAGES} "en_EN")

# should ctest wipe the binary tree before running
SET(CTEST_START_WITH_EMPTY_BINARY_DIRECTORY TRUE)
SET(CTEST_BACKUP_AND_RESTORE TRUE)

# this is the initial cache to use for the binary tree, be careful to escape
# any quotes inside of this string if you use it
if(WIN32)
  SET (CTEST_INITIAL_CACHE "
    MAKECOMMAND:STRING=nmake -i
    CMAKE_MAKE_PROGRAM:FILEPATH=nmake
    CMAKE_GENERATOR:INTERNAL=NMake Makefiles
    BUILDNAME:STRING=${EIGEN_BUILD_STRING}
    SITE:STRING=${EIGEN_SITE}
  ")
else(WIN32)
  SET (CTEST_INITIAL_CACHE "
    BUILDNAME:STRING=${EIGEN_BUILD_STRING}
    SITE:STRING=${EIGEN_SITE}
  ")
endif(WIN32)

# MAKECOMMAND:STRING=nmake -i
# CMAKE_MAKE_PROGRAM:FILEPATH=make
# CMAKE_GENERATOR:INTERNAL=Makefiles
SET (CTEST_INITIAL_CACHE "
  BUILDNAME:STRING=opensuse-11_1-x86_64-${EIGEN_CXX}
  SITE:STRING=pc-gael
  CVSCOMMAND:FILEPATH=/usr/bin/svn"
)

# set any extra environment variables to use during the execution of the script here:

if(EIGEN_CXX)
  set(CTEST_ENVIRONMENT "CXX=${EIGEN_CXX}")
endif(EIGEN_CXX)

if(DEFINED EIGEN_EXPLICIT_VECTORIZATION)
  if(EIGEN_EXPLICIT_VECTORIZATION MATCHES SSE2)
    set(CTEST_CMAKE_COMMAND "${CTEST_CMAKE_COMMAND} -DEIGEN_TEST_SSE2=ON")
  elseif(EIGEN_EXPLICIT_VECTORIZATION MATCHES SSE3)
    set(CTEST_CMAKE_COMMAND "${CTEST_CMAKE_COMMAND} -DEIGEN_TEST_SSE2=ON -DEIGEN_TEST_SSE3=ON")
  elseif(EIGEN_EXPLICIT_VECTORIZATION MATCHES Altivec)
    set(CTEST_CMAKE_COMMAND "${CTEST_CMAKE_COMMAND} -DEIGEN_TEST_ALTIVEC=ON")
  elseif(EIGEN_EXPLICIT_VECTORIZATION MATCHES novec)
    set(CTEST_CMAKE_COMMAND "${CTEST_CMAKE_COMMAND} -DEIGEN_TEST_NO_EXPLICIT_VECTORIZATION=ON")
  else(EIGEN_EXPLICIT_VECTORIZATION MATCHES SSE2)
    message(FATAL_ERROR "Invalid value for EIGEN_EXPLICIT_VECTORIZATION (${EIGEN_EXPLICIT_VECTORIZATION}), must be: novec, SSE2, SSE3, Altivec")
  endif(EIGEN_EXPLICIT_VECTORIZATION MATCHES SSE2)
endif(DEFINED EIGEN_EXPLICIT_VECTORIZATION)
