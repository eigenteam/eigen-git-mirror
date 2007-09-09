FILE(GLOB Eigen_internal_SRCS "*.h")

INSTALL(FILES 
  ${Eigen_internal_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/internal
  )
