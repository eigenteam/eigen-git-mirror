
FIND_PACKAGE(Qt4 REQUIRED)

set(QT_USE_QTOPENGL TRUE)
include(${QT_USE_FILE})

SET(CMAKE_INCLUDE_CURRENT_DIR ON)

INCLUDE_DIRECTORIES( ${QT_INCLUDE_DIR} )

SET(quaternion_demo_SRCS  gpuhelper.cpp camera.cpp trackball.cpp quaternion_demo.cpp)

QT4_AUTOMOC(${quaternion_demo_SRCS})

ADD_EXECUTABLE(quaternion_demo ${quaternion_demo_SRCS})

TARGET_LINK_LIBRARIES(quaternion_demo ${QT_QTCORE_LIBRARY} ${QT_QTGUI_LIBRARY}  ${QT_QTOPENGL_LIBRARY})
