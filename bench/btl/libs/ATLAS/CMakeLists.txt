
include_directories(${BLITZ_INCLUDES})
add_executable(btl_blitz main.cpp)
target_link_libraries(btl_blitz ${BLITZ_LIBRARIES})
