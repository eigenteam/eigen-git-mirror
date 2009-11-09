FILE(GLOB Eigen_NonLinear_SRCS "*.h")

INSTALL(FILES
  ${Eigen_NonLinear_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/unsupported/Eigen/src/NonLinear COMPONENT Devel
  )
