PROJECT(Eigen)

OPTION(BUILD_TESTS "Build tests" OFF)
OPTION(BUILD_DOC "Build documentation and examples" OFF)

SET(CMAKE_INCLUDE_CURRENT_DIR ON)

IF(CMAKE_COMPILER_IS_GNUCXX)
  IF(CMAKE_SYSTEM_NAME MATCHES Linux)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wnon-virtual-dtor -Wno-long-long -ansi -Wundef -Wcast-align -Wchar-subscripts -Wall -W -Wpointer-arith -Wwrite-strings -Wformat-security -fno-exceptions -fno-check-new -fno-common -fstrict-aliasing")
  ENDIF(CMAKE_SYSTEM_NAME MATCHES Linux)
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR})

ADD_SUBDIRECTORY(Eigen)
ADD_SUBDIRECTORY(test)
ADD_SUBDIRECTORY(doc)
