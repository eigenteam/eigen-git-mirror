FILE(GLOB tvmet_loop_header_SRCS "*.h")

INSTALL(FILES 
  ${tvmet_loop_header_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/loop
  )
