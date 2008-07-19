FILE(GLOB snippets_SRCS "*.cpp")

ADD_CUSTOM_TARGET(all_snippets)

FOREACH(snippet_src ${snippets_SRCS})
GET_FILENAME_COMPONENT(snippet ${snippet_src} NAME_WE)
SET(compile_snippet_target compile_${snippet})
SET(compile_snippet_src ${compile_snippet_target}.cpp)
FILE(READ ${snippet_src} snippet_source_code)
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/compile_snippet.cpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/${compile_snippet_src})
ADD_EXECUTABLE(${compile_snippet_target}
               ${CMAKE_CURRENT_BINARY_DIR}/${compile_snippet_src})
GET_TARGET_PROPERTY(compile_snippet_executable
                    ${compile_snippet_target} LOCATION)
ADD_CUSTOM_COMMAND(
  TARGET ${compile_snippet_target}
  POST_BUILD
  COMMAND ${compile_snippet_executable}
  ARGS >${CMAKE_CURRENT_BINARY_DIR}/${snippet}.out
)
ADD_DEPENDENCIES(all_snippets ${compile_snippet_target})
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${compile_snippet_src}
                            PROPERTIES OBJECT_DEPENDS ${snippet_src})
ENDFOREACH(snippet_src)
