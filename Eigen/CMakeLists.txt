file(GLOB Eigen_directory_files "*")
foreach(f ${Eigen_directory_files})
  if(NOT(f MATCHES ".txt") AND NOT (f MATCHES "src"))
    list(APPEND Eigen_directory_files_to_install ${f})
  endif()
endforeach(f ${Eigen_directory_files})

install(FILES
  ${Eigen_directory_files_to_install}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen COMPONENT Devel
  )

add_subdirectory(src)
