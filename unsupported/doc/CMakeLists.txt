
set_directory_properties(PROPERTIES EXCLUDE_FROM_ALL TRUE)


if(NOT MSVC)
  add_subdirectory(examples)
  add_subdirectory(snippets)
endif(NOT MSVC)
