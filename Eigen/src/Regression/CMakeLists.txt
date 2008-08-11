FILE(GLOB Eigen_Regression_SRCS "*.h")

INSTALL(FILES 
  ${Eigen_Regression_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/Regression
  )
