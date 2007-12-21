FILE(GLOB examples_SRCS "*.cpp")

ADD_CUSTOM_TARGET(all_examples)

FOREACH(example_src ${examples_SRCS})
GET_FILENAME_COMPONENT(example ${example_src} NAME_WE)
ADD_EXECUTABLE(${example} ${example_src})
GET_TARGET_PROPERTY(example_executable
                    ${example} LOCATION)
ADD_CUSTOM_COMMAND(
  TARGET ${example}
  POST_BUILD
  COMMAND ${example_executable}
  ARGS >${CMAKE_CURRENT_BINARY_DIR}/${example}.out
)
ADD_DEPENDENCIES(all_examples ${example})
ENDFOREACH(example_src)
