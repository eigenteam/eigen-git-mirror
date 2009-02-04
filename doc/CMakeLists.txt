
set_directory_properties(PROPERTIES EXCLUDE_FROM_ALL TRUE)

if(CMAKE_COMPILER_IS_GNUCXX)
  if(CMAKE_SYSTEM_NAME MATCHES Linux)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1 -g1")
  endif(CMAKE_SYSTEM_NAME MATCHES Linux)
endif(CMAKE_COMPILER_IS_GNUCXX)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
  ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/eigendoxy_header.html.in
  ${CMAKE_CURRENT_BINARY_DIR}/eigendoxy_header.html
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/eigendoxy_footer.html.in
  ${CMAKE_CURRENT_BINARY_DIR}/eigendoxy_footer.html
)

set(examples_targets "")
set(snippets_targets "")

add_definitions("-DEIGEN_DEFAULT_IO_FORMAT=EIGEN_DOCS_IO_FORMAT")

add_subdirectory(examples)
add_subdirectory(snippets)

add_custom_target(
  doc
  ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/html/
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/eigendoxy_tabs.css
                                   ${CMAKE_CURRENT_BINARY_DIR}/html/
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Eigen_Silly_Professor_64x64.png
                                   ${CMAKE_CURRENT_BINARY_DIR}/html/
  COMMAND doxygen
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_dependencies(doc all_snippets all_examples)
