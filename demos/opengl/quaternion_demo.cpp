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
#include "icosphere.h"

#include <Eigen/Array>
#include <Eigen/QR>
#include <Eigen/LU>

#include <QEvent>
#include <QMouseEvent>
#include <QInputDialog>
#include <QGridLayout>
#include <QButtonGroup>
#include <QRadioButton>
#include <QDockWidget>
#include <QPushButton>

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

RenderingWidget::RenderingWidget()
{
  mAnimate = false;
  mCurrentTrackingMode = TM_NO_TRACK;
  mNavMode = NavTurnAround;
  mLerpMode = LerpQuaternion;
  mRotationMode = RotationStable;
  mTrackball.setCamera(&mCamera);

  // required to capture key press events
  setFocusPolicy(Qt::ClickFocus);
}

void RenderingWidget::grabFrame(void)
{
    // ask user for a time
    bool ok = false;
    double t = 0;
    if (!m_timeline.empty())
      t = (--m_timeline.end())->first + 1.;
    t = QInputDialog::getDouble(this, "Eigen's RenderingWidget", "time value: ",
      t, 0, 1e3, 1, &ok);
    if (ok)
    {
      Frame aux;
      aux.orientation = mCamera.viewMatrix().linear();
      aux.position = mCamera.viewMatrix().translation();
      m_timeline[t] = aux;
    }
}

void RenderingWidget::drawScene()
{
  float length = 50;
  gpu.drawVector(Vector3f::Zero(), length*Vector3f::UnitX(), Color(1,0,0,1));
  gpu.drawVector(Vector3f::Zero(), length*Vector3f::UnitY(), Color(0,1,0,1));
  gpu.drawVector(Vector3f::Zero(), length*Vector3f::UnitZ(), Color(0,0,1,1));

  // draw the fractal object
  float sqrt3 = ei_sqrt(3.);
  glLightfv(GL_LIGHT0, GL_AMBIENT, Vector4f(0.5,0.5,0.5,1).data());
  glLightfv(GL_LIGHT0, GL_DIFFUSE, Vector4f(0.5,1,0.5,1).data());
  glLightfv(GL_LIGHT0, GL_SPECULAR, Vector4f(1,1,1,1).data());
  glLightfv(GL_LIGHT0, GL_POSITION, Vector4f(-sqrt3,-sqrt3,sqrt3,0).data());

  glLightfv(GL_LIGHT1, GL_AMBIENT, Vector4f(0,0,0,1).data());
  glLightfv(GL_LIGHT1, GL_DIFFUSE, Vector4f(1,0.5,0.5,1).data());
  glLightfv(GL_LIGHT1, GL_SPECULAR, Vector4f(1,1,1,1).data());
  glLightfv(GL_LIGHT1, GL_POSITION, Vector4f(-sqrt3,sqrt3,-sqrt3,0).data());

  glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, Vector4f(0.7, 0.7, 0.7, 1).data());
  glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, Vector4f(0.8, 0.75, 0.6, 1).data());
  glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, Vector4f(1, 1, 1, 1).data());
  glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 64);

  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  glEnable(GL_LIGHT1);
  
  glColor3f(0.4, 0.7, 0.4);
  glVertexPointer(3, GL_FLOAT, 0, mVertices[0].data());
  glNormalPointer(GL_FLOAT, 0, mNormals[0].data());
  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_NORMAL_ARRAY);
  glDrawArrays(GL_TRIANGLES, 0, mVertices.size());
  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_NORMAL_ARRAY);
  
  glDisable(GL_LIGHTING);
}

void RenderingWidget::animate()
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

void RenderingWidget::keyPressEvent(QKeyEvent * e)
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
          resetCamera();
          break;
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

void RenderingWidget::stopAnimation()
{
  disconnect(&m_timer, SIGNAL(timeout()), this, SLOT(animate()));
  m_timer.stop();
  mAnimate = false;
  m_alpha = 0;
}

void RenderingWidget::mousePressEvent(QMouseEvent* e)
{
  mMouseCoords = Vector2i(e->pos().x(), e->pos().y());
  bool fly = (mNavMode==NavFly) || (e->modifiers()&Qt::ControlModifier);
  switch(e->button())
  {
    case Qt::LeftButton:
      if(fly)
      {
        mCurrentTrackingMode = TM_LOCAL_ROTATE;
        mTrackball.start(Trackball::Local);
      }
      else
      {
        mCurrentTrackingMode = TM_ROTATE_AROUND;
        mTrackball.start(Trackball::Around);
      }
      mTrackball.track(mMouseCoords);
      break;
    case Qt::MidButton:
      if(fly)
        mCurrentTrackingMode = TM_FLY_Z;
      else
        mCurrentTrackingMode = TM_ZOOM;
      break;
    case Qt::RightButton:
        mCurrentTrackingMode = TM_FLY_PAN;
      break;
    default:
      break;
  }
}
void RenderingWidget::mouseReleaseEvent(QMouseEvent*)
{
    mCurrentTrackingMode = TM_NO_TRACK;
    updateGL();
}

void RenderingWidget::mouseMoveEvent(QMouseEvent* e)
{
    // tracking
    if(mCurrentTrackingMode != TM_NO_TRACK)
    {
        float dx =   float(e->x() - mMouseCoords.x()) / float(mCamera.vpWidth());
        float dy = - float(e->y() - mMouseCoords.y()) / float(mCamera.vpHeight());

        if(e->modifiers() & Qt::ShiftModifier)
        {
          dx *= 10.;
          dy *= 10.;
        }

        switch(mCurrentTrackingMode)
        {
          case TM_ROTATE_AROUND:
          case TM_LOCAL_ROTATE:
            mTrackball.track(Vector2i(e->pos().x(), e->pos().y()));
            break;
          case TM_ZOOM :
            mCamera.zoom(dy*50);
            break;
          case TM_FLY_Z :
            mCamera.localTranslate(Vector3f(0, 0, -dy*100));
            break;
          case TM_FLY_PAN :
            mCamera.localTranslate(Vector3f(dx*100, dy*100, 0));
            break;
          default:
            break;
        }

        updateGL();
    }

    mMouseCoords = Vector2i(e->pos().x(), e->pos().y());
}

void RenderingWidget::paintGL()
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

void RenderingWidget::initializeGL()
{
  glClearColor(1., 1., 1., 0.);
  glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, 1);
  glDepthMask(GL_TRUE);
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

  mCamera.setPosition(Vector3f(-200, -200, -200));
  mCamera.setTarget(Vector3f(0, 0, 0));
  mInitFrame.orientation = mCamera.orientation().inverse();
  mInitFrame.position = mCamera.viewMatrix().translation();

  // create a kind of fractal sphere
  {
    IcoSphere pattern;
    
    int levels = 3;
    float scale = 0.45;
    float radius = 100;
    std::vector<Vector3f> centers;
    std::vector<int> parents;
    std::vector<float> radii;
    centers.push_back(Vector3f::Zero());
    parents.push_back(-1);
    radii.push_back(radius);
    radius *= scale;

    // generate level 1 using icosphere vertices
    {
      float dist = radii[0]*0.9;
      for (int i=0; i<12; ++i)
      {
        centers.push_back(pattern.vertices()[i] * dist);
        radii.push_back(radius);
        parents.push_back(0);
      }
    }

    scale = 0.33;
    static const float angles [10] = {
      0, 0,
      M_PI, 0.*M_PI,
      M_PI, 0.5*M_PI,
      M_PI, 1.*M_PI,
      M_PI, 1.5*M_PI};
    
    // generate other levels
    int start = 1;
    float maxAngle = M_PI/2;
    for (int l=1; l<levels; l++)
    {
      radius *= scale;
      int end = centers.size();
      for (int i=start; i<end; ++i)
      {
        Vector3f c = centers[i];
        Vector3f ax0, ax1;
        if (parents[i]==-1)
          ax0 = Vector3f::UnitZ();
        else
          ax0 = (c - centers[parents[i]]).normalized();
        ax1 = ax0.unitOrthogonal();
        Quaternionf q;
        q.setFromTwoVectors(Vector3f::UnitZ(), ax0);
        Transform3f t = Translation3f(c) * q * Scaling3f(radii[i]+radius);
        for (int j=0; j<5; ++j)
        {
          Vector3f newC = c + ( (AngleAxisf(angles[j*2+1], ax0)
                               * AngleAxisf(angles[j*2+0] * (l==1 ? 0.35 : 0.5), ax1)) * ax0)*(radii[i] + radius*0.8);
          centers.push_back(newC);
          radii.push_back(radius);
          parents.push_back(i);
        }
      }
      start = end;
      maxAngle = M_PI/2;
    }
    parents.clear();
    // instanciate the geometry
    {
      const std::vector<int>& sphereIndices = pattern.indices(2);
      std::cout << "instanciate geometry...  (" << sphereIndices.size() * centers.size() << " vertices)\n";
      mVertices.reserve(sphereIndices.size() * centers.size());
      mNormals.reserve(sphereIndices.size() * centers.size());
      int end = centers.size();
      for (int i=0; i<end; ++i)
      {
        Transform3f t = Translation3f(centers[i]) * Scaling3f(radii[i]);
        // copy vertices
        for (unsigned int j=0; j<sphereIndices.size(); ++j)
        {
          Vector3f v = pattern.vertices()[sphereIndices[j]];
          mVertices.push_back(t * v);
          mNormals.push_back(v);
        }
      }
    }
  }
}

void RenderingWidget::resizeGL(int width, int height)
{
    mCamera.setViewport(width,height);
}

void RenderingWidget::setNavMode(int m)
{
  mNavMode = NavMode(m);
}

void RenderingWidget::setLerpMode(int m)
{
  mLerpMode = LerpMode(m);
}

void RenderingWidget::setRotationMode(int m)
{
  mRotationMode = RotationMode(m);
}

void RenderingWidget::resetCamera()
{
  if (mAnimate)
    stopAnimation();
  m_timeline.clear();
  Frame aux0 = mCamera.frame();
  aux0.orientation = aux0.orientation.inverse();
  aux0.position = mCamera.viewMatrix().translation();
  m_timeline[0] = aux0;

  Vector3f currentTarget = mCamera.target();
  mCamera.setTarget(Vector3f::Zero());

  // compute the rotation duration to move the camera to the target
  Frame aux1 = mCamera.frame();
  aux1.orientation = aux1.orientation.inverse();
  aux1.position = mCamera.viewMatrix().translation();
  float rangle = AngleAxisf(aux0.orientation.inverse() * aux1.orientation).angle();
  if (rangle>M_PI)
    rangle = 2.*M_PI - rangle;
  float duration = rangle * 0.9;
  if (duration<0.1) duration = 0.1;

  // put the camera at that time step:
  aux1 = aux0.lerp(duration/2,mInitFrame);
  // and make it look at teh target again
  aux1.orientation = aux1.orientation.inverse();
  aux1.position = - (aux1.orientation * aux1.position);
  mCamera.setFrame(aux1);
  mCamera.setTarget(Vector3f::Zero());

  // add this camera keyframe
  aux1.orientation = aux1.orientation.inverse();
  aux1.position = mCamera.viewMatrix().translation();
  m_timeline[duration] = aux1;

  m_timeline[2] = mInitFrame;
  m_alpha = 0;
  animate();
  connect(&m_timer, SIGNAL(timeout()), this, SLOT(animate()));
  m_timer.start(1000/30);
  mAnimate = true;
}

QWidget* RenderingWidget::createNavigationControlWidget()
{
  QWidget* panel = new QWidget();
  QVBoxLayout* layout = new QVBoxLayout();

  {
    // navigation mode
    QButtonGroup* group = new QButtonGroup(panel);
    QRadioButton* but;
    but = new QRadioButton("turn around");
    group->addButton(but, NavTurnAround);
    layout->addWidget(but);
    but = new QRadioButton("fly");
    group->addButton(but, NavFly);
    layout->addWidget(but);
    group->button(mNavMode)->setChecked(true);
    connect(group, SIGNAL(buttonClicked(int)), this, SLOT(setNavMode(int)));
  }
  {
    QPushButton* but = new QPushButton("reset");
    layout->addWidget(but);
    connect(but, SIGNAL(clicked()), this, SLOT(resetCamera()));
  }
  {
    // track ball, rotation mode
    QButtonGroup* group = new QButtonGroup(panel);
    QRadioButton* but;
    but = new QRadioButton("stable trackball");
    group->addButton(but, RotationStable);
    layout->addWidget(but);
    but = new QRadioButton("standard rotation");
    group->addButton(but, RotationStandard);
    layout->addWidget(but);
    but->setEnabled(false);
    group->button(mRotationMode)->setChecked(true);
    connect(group, SIGNAL(buttonClicked(int)), this, SLOT(setRotationMode(int)));
  }
  {
    // interpolation mode
    QButtonGroup* group = new QButtonGroup(panel);
    QRadioButton* but;
    but = new QRadioButton("quaternion slerp");
    group->addButton(but, LerpQuaternion);
    layout->addWidget(but);
    but = new QRadioButton("euler angles");
    group->addButton(but, LerpEulerAngles);
    layout->addWidget(but);
    but->setEnabled(false);
    group->button(mNavMode)->setChecked(true);
    connect(group, SIGNAL(buttonClicked(int)), this, SLOT(setLerpMode(int)));
  }
  layout->addItem(new QSpacerItem(0,0,QSizePolicy::Minimum,QSizePolicy::Expanding));
  panel->setLayout(layout);
  return panel;
}

QuaternionDemo::QuaternionDemo()
{
  mRenderingWidget = new RenderingWidget();
  setCentralWidget(mRenderingWidget);

  QDockWidget* panel = new QDockWidget("navigation", this);
  panel->setAllowedAreas((QFlags<Qt::DockWidgetArea>)(Qt::RightDockWidgetArea | Qt::LeftDockWidgetArea));
  addDockWidget(Qt::RightDockWidgetArea, panel);
  panel->setWidget(mRenderingWidget->createNavigationControlWidget());
}

int main(int argc, char *argv[])
{
  QApplication app(argc, argv);
  QuaternionDemo demo;
  demo.resize(600,500);
  demo.show();
  return app.exec();
}

#include "quaternion_demo.moc"
