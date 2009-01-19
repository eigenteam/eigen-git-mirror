FILE(GLOB Eigen_StdVector_SRCS "*.h")

INSTALL(FILES
  ${Eigen_StdVector_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/StdVector
  )
