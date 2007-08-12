INCLUDE(CheckIncludeFiles)

IF(BUILD_TESTS)

ENABLE_TESTING()

FIND_PACKAGE(Qt4 REQUIRED)

INCLUDE_DIRECTORIES( ${QT_INCLUDE_DIR} )

SET(testsuite_SOURCES
  main.cpp
  selftest.cpp
  testtraits.cpp
#   TestTraits.h TestTraits.cc
#   TestTraitsComplex.h TestTraitsComplex.cc
#   TestConstruction.h TestConstruction.cc
#   TestComplexMatrix.h TestComplexMatrix.cc
#   TestComplexVector.h TestComplexVector.cc
#   TestDimension.h TestDimension.cc
#   TestMathOps.h TestMathOps.cc
#   TestMatrixEval.h TestMatrixEval.cc
#   TestMatrixFunctions.h TestMatrixFunctions.cc
#   TestMatrixOperators.h TestMatrixOperators.cc
#   TestSTL.h TestSTL.cc
#   TestUnFunc.h TestUnFunc.cc
#   TestVectorEval.h TestVectorEval.cc
#   TestVectorFunctions.h TestVectorFunctions.cc
#   TestVectorOperators.h TestVectorOperators.cc
#   TestXpr.h TestXpr.cc
#   TestXprMatrixFunctions.h TestXprMatrixFunctions.cc
#   TestXprMatrixOperators.h TestXprMatrixOperators.cc
#   TestXprVectorFunctions.h TestXprVectorFunctions.cc
#   TestXprVectorOperators.h TestXprVectorOperators.cc
#   TestUnloops.h TestUnloops.cc
#   TestMatrixElementwise.h TestMatrixElementwise.cc
)

QT4_AUTOMOC(${testsuite_SOURCES})

INCLUDE_DIRECTORIES(${CPPUNIT_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/tvmet-1.7.1/include
  ${CMAKE_INSTALL_PREFIX}/include)

ADD_EXECUTABLE(testsuite ${testsuite_SOURCES})

TARGET_LINK_LIBRARIES(testsuite ${QT_QTCORE_LIBRARY} ${QT_QTTEST_LIBRARY})

ADD_TEST(Eigen testsuite)

ENDIF(BUILD_TESTS)
