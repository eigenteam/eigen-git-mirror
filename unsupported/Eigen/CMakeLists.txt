set(Eigen_HEADERS AdolcForward)

install(FILES
  ${Eigen_HEADERS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/unsupported/Eigen
  )

add_subdirectory(src)
