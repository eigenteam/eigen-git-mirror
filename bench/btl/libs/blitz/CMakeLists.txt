
include_directories(${BLITZ_INCLUDES})
btl_add_bench(btl_blitz main.cpp)
target_link_libraries(btl_blitz ${BLITZ_LIBRARIES})
