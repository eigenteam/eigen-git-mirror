FILE(GLOB tvmet_xpr_header_SRCS "*.h")

INSTALL(FILES 
  ${tvmet_xpr_header_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/xpr
  )
