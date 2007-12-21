IF(BUILD_DOC)

CONFIGURE_FILE(
  ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
  ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
)

SET(examples_targets "")
SET(snippets_targets "")

ADD_SUBDIRECTORY(examples)
ADD_SUBDIRECTORY(snippets)

ADD_CUSTOM_TARGET(
  run_doxygen
  ALL
  COMMAND doxygen
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

ADD_DEPENDENCIES(run_doxygen all_snippets all_examples)

ENDIF(BUILD_DOC)
