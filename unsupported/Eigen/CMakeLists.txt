set(Eigen_HEADERS AdolcForward IterativeSolvers)

install(FILES
  ${Eigen_HEADERS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/unsupported/Eigen
  )

add_subdirectory(src)
