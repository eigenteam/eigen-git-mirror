SET(Eigen_HEADERS Core CoreDeclarations LU Cholesky QR)

SET(Eigen_SRCS
  src/Core/CoreInstantiations.cpp
  src/Cholesky/CholeskyInstantiations.cpp
  src/QR/QrInstantiations.cpp
)

ADD_LIBRARY(Eigen2 SHARED ${Eigen_SRCS})

IF(CMAKE_COMPILER_IS_GNUCXX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g1 -O2")
    SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -g1 -O2")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

SET(INCLUDE_INSTALL_DIR
    "${CMAKE_INSTALL_PREFIX}/include/eigen2"
    CACHE PATH
    "The directory where we install the header files"
    FORCE)

INSTALL(FILES
  ${Eigen_HEADERS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen
  )

INSTALL(TARGETS Eigen2
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

ADD_SUBDIRECTORY(src)