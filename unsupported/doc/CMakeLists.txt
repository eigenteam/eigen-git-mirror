
set_directory_properties(PROPERTIES EXCLUDE_FROM_ALL TRUE)

if(NOT MSVC)
  add_subdirectory(examples)
endif(NOT MSVC)
