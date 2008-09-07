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

#include "quaternion_demo.h"

#include <Eigen/Array>
#include <Eigen/QR>
#include <Eigen/LU>

#include <QEvent>
#include <QMouseEvent>
#include <QInputDialog>

using namespace Eigen;



template<typename T> T lerp(float t, const T& a, const T& b)
{
  return a*(1-t) + b*t;
}

template<> Quaternionf lerp(float t, const Quaternionf& a, const Quaternionf& b)
{ return a.slerp(t,b); }

template<> AngleAxisf lerp(float t, const AngleAxisf& a, const AngleAxisf& b)
{
  return AngleAxisf(lerp(t,a.angle(),b.angle()),
                    lerp(t,a.axis(),b.axis()).normalized());
}

template<typename OrientationType>
inline static Frame lerpFrame(float alpha, const Frame& a, const Frame& b)
{
  return Frame(::lerp(alpha,a.position,b.position),
               Quaternionf(::lerp(alpha,OrientationType(a.orientation),OrientationType(b.orientation))));
}

QuaternionDemo::QuaternionDemo()
{
  mAnimate = false;
  mTrackMode = TM_NO_TRACK;
  mTrackball.setCamera(&mCamera);
}

void QuaternionDemo::grabFrame(void)
{
    // ask user for a time
    bool ok = false;
    double t = 0;
    if (!m_timeline.empty())
      t = (--m_timeline.end())->first + 1.;
    t = QInputDialog::getDouble(this, "Eigen's QuaternionDemo", "time value: ",
      t, 0, 1e3, 1, &ok);
    if (ok)
    {
      Frame aux;
      aux.orientation = mCamera.viewMatrix().linear();
      aux.position = mCamera.viewMatrix().translation();
      m_timeline[t] = aux;
    }
}

void QuaternionDemo::drawScene()
{
  float length = 50;
  gpu.drawVector(Vector3f::Zero(), length*Vector3f::UnitX(), Color(1,0,0,1));
  gpu.drawVector(Vector3f::Zero(), length*Vector3f::UnitY(), Color(0,1,0,1));
  gpu.drawVector(Vector3f::Zero(), length*Vector3f::UnitZ(), Color(0,0,1,1));
}

void QuaternionDemo::drawPath()
{

}

void QuaternionDemo::animate()
{
  m_alpha += double(m_timer.interval()) * 1e-3;

  TimeLine::const_iterator hi = m_timeline.upper_bound(m_alpha);
  TimeLine::const_iterator lo = hi;
  --lo;

  Frame currentFrame;

  if(hi==m_timeline.end())
  {
    // end
    currentFrame = lo->second;
    stopAnimation();
  }
  else if(hi==m_timeline.begin())
  {
    // start
    currentFrame = hi->second;
  }
  else
  {
    float s = (m_alpha - lo->first)/(hi->first - lo->first);
    currentFrame = ::lerpFrame<Eigen::Quaternionf>(s, lo->second, hi->second);
    currentFrame.orientation.coeffs().normalize();
  }

  currentFrame.orientation = currentFrame.orientation.inverse();
  currentFrame.position = - (currentFrame.orientation * currentFrame.position);
  mCamera.setFrame(currentFrame);
  
  updateGL();
}

void QuaternionDemo::keyPressEvent(QKeyEvent * e)
{
    switch(e->key())
    {
        case Qt::Key_Up:
            mCamera.zoom(2);
            break;
        case Qt::Key_Down:
            mCamera.zoom(-2);
            break;
        // add a frame
        case Qt::Key_G:
            grabFrame();
            break;
        // clear the time line
        case Qt::Key_C:
            m_timeline.clear();
            break;
        // move the camera to initial pos
        case Qt::Key_R:
          {
            if (mAnimate)
              stopAnimation();
            m_timeline.clear();
            float duration = 3/*AngleAxisf(mCamera.orientation().inverse()
                              * mInitFrame.orientation).angle()*/;
            Frame aux = mCamera.frame();
            aux.orientation = aux.orientation.inverse();
            aux.position = mCamera.viewMatrix().translation();
            m_timeline[0] = aux;
            m_timeline[duration] = mInitFrame;
          }
        // start/stop the animation
        case Qt::Key_A:
            if (mAnimate)
            {
              stopAnimation();
            }
            else
            {
              m_alpha = 0;
              connect(&m_timer, SIGNAL(timeout()), this, SLOT(animate()));
              m_timer.start(1000/30);
              mAnimate = true;
            }
            break;
        default:
            break;
    }

    updateGL();
}

void QuaternionDemo::stopAnimation()
{
  disconnect(&m_timer, SIGNAL(timeout()), this, SLOT(animate()));
  m_timer.stop();
  mAnimate = false;
  m_alpha = 0;
}

void QuaternionDemo::mousePressEvent(QMouseEvent* e)
{
  mMouseCoords = Vector2i(e->pos().x(), e->pos().y());
  switch(e->button())
  {
    case Qt::LeftButton:
      if(e->modifiers()&Qt::ControlModifier)
      {
        mTrackMode = TM_QUAKE_ROTATE;
      }
      else
      {
        mTrackMode = TM_ROTATE_AROUND;
        mTrackball.reset();
        mTrackball.track(mMouseCoords);
      }
      break;
    case Qt::MidButton:
      if(e->modifiers()&Qt::ControlModifier)
        mTrackMode = TM_QUAKE_WALK;
      else
        mTrackMode = TM_ZOOM;
      break;
    case Qt::RightButton:
        mTrackMode = TM_QUAKE_PAN;
      break;
    default:
      break;
  }
}
void QuaternionDemo::mouseReleaseEvent(QMouseEvent*)
{
    mTrackMode = TM_NO_TRACK;
    updateGL();
}

void QuaternionDemo::mouseMoveEvent(QMouseEvent* e)
{
    // tracking
    if(mTrackMode != TM_NO_TRACK)
    {
        float dx =   float(e->x() - mMouseCoords.x()) / float(mCamera.vpWidth());
        float dy = - float(e->y() - mMouseCoords.y()) / float(mCamera.vpHeight());

        if(e->modifiers() & Qt::ShiftModifier)
        {
          dx *= 10.;
          dy *= 10.;
        }

        switch(mTrackMode)
        {
          case TM_ROTATE_AROUND :
            mTrackball.track(Vector2i(e->pos().x(), e->pos().y()));
            break;
          case TM_ZOOM :
            mCamera.zoom(dy*50);
            break;
          case TM_QUAKE_WALK :
            mCamera.localTranslate(Vector3f(0, 0, dy*100));
            break;
          case TM_QUAKE_PAN :
            mCamera.localTranslate(Vector3f(dx*100, dy*100, 0));
            break;
          case TM_QUAKE_ROTATE :
            mCamera.localRotate(-dx*M_PI, dy*M_PI);
            break;
          default:
            break;
        }

        updateGL();
    }

    mMouseCoords = Vector2i(e->pos().x(), e->pos().y());
}

void QuaternionDemo::paintGL()
{
  glEnable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);
  glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
  glDisable(GL_COLOR_MATERIAL);
  glDisable(GL_BLEND);
  glDisable(GL_ALPHA_TEST);
  glDisable(GL_TEXTURE_1D);
  glDisable(GL_TEXTURE_2D);
  glDisable(GL_TEXTURE_3D);

  // Clear buffers
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  mCamera.activateGL();

  drawScene();
}

void QuaternionDemo::initializeGL()
{
  glClearColor(1., 1., 1., 0.);
  glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, 1);
  glDepthMask(GL_TRUE);
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

  mInitFrame.orientation = mCamera.viewMatrix().linear();
  mInitFrame.position = mCamera.viewMatrix().translation();
}

void QuaternionDemo::resizeGL(int width, int height)
{
    mCamera.setViewport(width,height);
}

int main(int argc, char *argv[])
{
  QApplication app(argc, argv);
  QuaternionDemo demo;
  demo.show();
  return app.exec();
}

#include "quaternion_demo.moc"
