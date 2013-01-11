FILE(GLOB Eigen_SparseQRSupport_SRCS "*.h")

INSTALL(FILES
  ${Eigen_SparseQRSupport_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/SparseQRSupport/ COMPONENT Devel
  )
