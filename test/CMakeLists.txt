IF(BUILD_TESTS)

ENABLE_TESTING()
FIND_PACKAGE(Qt4 REQUIRED)
INCLUDE_DIRECTORIES( ${QT_INCLUDE_DIR} )

SET(test_SRCS
  main.cpp
  basicstuff.cpp
  linearstructure.cpp
  product.cpp
  adjoint.cpp
  submatrices.cpp
  miscmatrices.cpp
  smallvectors.cpp
  map.cpp
)
QT4_AUTOMOC(${test_SRCS})

ADD_EXECUTABLE(test ${test_SRCS})
TARGET_LINK_LIBRARIES(test ${QT_QTCORE_LIBRARY} ${QT_QTTEST_LIBRARY})

ADD_TEST(Eigen test)

ENDIF(BUILD_TESTS)
