set(Eigen_HEADERS AdolcForward BVH IterativeSolvers MoreVectorization)

install(FILES
  ${Eigen_HEADERS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/unsupported/Eigen COMPONENT Devel
  )

add_subdirectory(src)
