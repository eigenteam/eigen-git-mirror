# Intel mkl-dnn support.
# Link: https://github.com/intel/mkl-dnn
if (MKLDNN)
    set(MKLDNN_FIND_QUIETLY TRUE)
    set(MKLDNN_INCLUDES ${MKLDNN}/include)
    set(MKLDNN_LIBRARIES ${MKLDNN}/lib)
endif (MKLDNN)
find_path(MKLDNN
        NAMES
        mkldnn.h
        PATHS
        $ENV{MKLDNNDIR}/include
        ${INCLUDE_INSTALL_DIR}
        )
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKLDNN DEFAULT_MSG
        MKLDNN)
mark_as_advanced(MKLDNN)