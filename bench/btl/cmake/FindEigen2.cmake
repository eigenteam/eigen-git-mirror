# - Try to find eigen2 headers
# Once done this will define
#
#  EIGEN2_FOUND - system has eigen2 lib
#  EIGEN2_INCLUDE_DIR - the eigen2 include directory
#
# Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
# Adapted from FindEigen.cmake:
# Copyright (c) 2006, 2007 Montel Laurent, <montel@kde.org>
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.

if (EIGEN2_INCLUDE_DIR)

  # in cache already
  set(EIGEN2_FOUND TRUE)

else (EIGEN2_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Eigen2 DEFAULT_MSG EIGEN2_INCLUDE_DIR)

find_path(EIGEN2_INCLUDE_DIR NAMES Eigen/Core
     PATHS
     ${Eigen_SOURCE_DIR}/
     ${INCLUDE_INSTALL_DIR}
   )

mark_as_advanced(EIGEN2_INCLUDE_DIR)

endif(EIGEN2_INCLUDE_DIR)

