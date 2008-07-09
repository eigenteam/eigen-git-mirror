
include_directories(${CBLAS_INCLUDES} ${PROJECT_SOURCE_DIR}/libs/f77)
btl_add_bench(btl_cblas main.cpp)
target_link_libraries(btl_cblas ${CBLAS_LIBRARIES})
