FILE(GLOB tvmet_meta_header_SRCS "*.h")

INSTALL(FILES 
  ${tvmet_meta_header_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/meta
  )
