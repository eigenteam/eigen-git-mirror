// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
//
// Eigen is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// Alternatively, you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of
// the License, or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License or the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License and a copy of the GNU General Public License along with
// Eigen. If not, see <http://www.gnu.org/licenses/>.

#ifndef EIGEN_QUATERNION_DEMO_H
#define EIGEN_QUATERNION_DEMO_H

#include "gpuhelper.h"
#include "camera.h"
#include "trackball.h"
#include <map>
#include <QTimer>
#include <QtGui/QApplication>
#include <QtOpenGL/QGLWidget>

class QuaternionDemo : public QGLWidget
{
  Q_OBJECT

    typedef std::map<float,Frame> TimeLine;
    TimeLine m_timeline;
    Frame lerpFrame(float t);

    Frame mInitFrame;
    bool mAnimate;
    float m_alpha;

    
    enum TrackMode {
      TM_NO_TRACK=0, TM_ROTATE_AROUND, TM_ZOOM,
      TM_QUAKE_ROTATE, TM_QUAKE_WALK, TM_QUAKE_PAN
    };

    Camera mCamera;
    TrackMode mTrackMode;
    Vector2i mMouseCoords;
    Trackball mTrackball;

    QTimer m_timer;

    void setupCamera();

  protected slots:

    virtual void animate(void);
    virtual void drawScene(void);
    virtual void drawPath(void);

    virtual void grabFrame(void);
    virtual void stopAnimation();

  protected:

    virtual void initializeGL();
    virtual void resizeGL(int width, int height);
    virtual void paintGL();
    
    //--------------------------------------------------------------------------------
    virtual void mousePressEvent(QMouseEvent * e);
    virtual void mouseReleaseEvent(QMouseEvent * e);
    virtual void mouseMoveEvent(QMouseEvent * e);
    virtual void keyPressEvent(QKeyEvent * e);
    //--------------------------------------------------------------------------------

  public:
    QuaternionDemo();
    ~QuaternionDemo() { }
};

#endif // EIGEN_QUATERNION_DEMO_H
