
include_directories(${BLITZ_INCLUDES})
btl_add_bench(btl_tiny_blitz main.cpp)
target_link_libraries(btl_tiny_blitz ${BLITZ_LIBRARIES})
