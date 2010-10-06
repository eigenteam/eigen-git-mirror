// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>
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

#include <main.h>
#include <iostream>
#include <GL/glut.h>
#include <Eigen/OpenGLSupport>
using namespace Eigen;




#define VERIFY_MATRIX(CODE,REF) { \
    glLoadIdentity(); \
    CODE; \
    Matrix4f m; m.setZero(); \
    glGet(GL_MODELVIEW_MATRIX, m); \
    if(!(REF).cast<float>().isApprox(m)) { \
      std::cerr << "Expected:\n" << ((REF).cast<float>()) << "\n" << "got\n" << m << "\n\n"; \
    } \
    VERIFY_IS_APPROX((REF).cast<float>(), m); \
  }

void test_openglsupport()
{
  int argc = 0;
  glutInit(&argc, 0);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowPosition (0,0);
  glutInitWindowSize(10, 10);

  if(glutCreateWindow("Eigen") <= 0)
  {
    std::cerr << "Unable to create GLUT Window.\n";
    exit(1);
  }

  Vector3f v3f;
  Matrix3f rot;
  glBegin(GL_POINTS);
  
  glVertex(v3f);
  glVertex(2*v3f+v3f);
  glVertex(rot*v3f);
  
  glEnd();
  
  // 4x4 matrices
  Matrix4f mf44; mf44.setRandom();
  VERIFY_MATRIX(glLoadMatrix(mf44), mf44);
  VERIFY_MATRIX(glMultMatrix(mf44), mf44);
  Matrix4d md44; md44.setRandom();
  VERIFY_MATRIX(glLoadMatrix(md44), md44);
  VERIFY_MATRIX(glMultMatrix(md44), md44);
  
  // Quaternion
  Quaterniond qd(AngleAxisd(ei_random<double>(), Vector3d::Random()));
  VERIFY_MATRIX(glRotate(qd), Projective3d(qd).matrix());
  
  Quaternionf qf(AngleAxisf(ei_random<double>(), Vector3f::Random()));
  VERIFY_MATRIX(glRotate(qf), Projective3f(qf).matrix());
  
  // 3D Transform
  Transform<float,3,AffineCompact> acf3; acf3.matrix().setRandom();
  VERIFY_MATRIX(glLoadMatrix(acf3), Projective3f(acf3).matrix());
  VERIFY_MATRIX(glMultMatrix(acf3), Projective3f(acf3).matrix());
  
  Transform<float,3,Affine> af3(acf3);
  VERIFY_MATRIX(glLoadMatrix(af3), Projective3f(af3).matrix());
  VERIFY_MATRIX(glMultMatrix(af3), Projective3f(af3).matrix());
  
  Transform<float,3,Projective> pf3; pf3.matrix().setRandom();
  VERIFY_MATRIX(glLoadMatrix(pf3), Projective3f(pf3).matrix());
  VERIFY_MATRIX(glMultMatrix(pf3), Projective3f(pf3).matrix());
  
  Transform<double,3,AffineCompact> acd3; acd3.matrix().setRandom();
  VERIFY_MATRIX(glLoadMatrix(acd3), Projective3d(acd3).matrix());
  VERIFY_MATRIX(glMultMatrix(acd3), Projective3d(acd3).matrix());
  
  Transform<double,3,Affine> ad3(acd3);
  VERIFY_MATRIX(glLoadMatrix(ad3), Projective3d(ad3).matrix());
  VERIFY_MATRIX(glMultMatrix(ad3), Projective3d(ad3).matrix());
  
  Transform<double,3,Projective> pd3; pd3.matrix().setRandom();
  VERIFY_MATRIX(glLoadMatrix(pd3), Projective3d(pd3).matrix());
  VERIFY_MATRIX(glMultMatrix(pd3), Projective3d(pd3).matrix());
  
  // translations (2D and 3D)
  {
    Vector2f vf2; vf2.setRandom(); Vector3f vf23; vf23 << vf2, 0;
    VERIFY_MATRIX(glTranslate(vf2), Projective3f(Translation3f(vf23)).matrix());
    Vector2d vd2; vd2.setRandom(); Vector3d vd23; vd23 << vd2, 0;
    VERIFY_MATRIX(glTranslate(vd2), Projective3d(Translation3d(vd23)).matrix());
    
    Vector3f vf3; vf3.setRandom();
    VERIFY_MATRIX(glTranslate(vf3), Projective3f(Translation3f(vf3)).matrix());
    Vector3d vd3; vd3.setRandom();
    VERIFY_MATRIX(glTranslate(vd3), Projective3d(Translation3d(vd3)).matrix());
    
    Translation<float,3> tf3; tf3.vector().setRandom();
    VERIFY_MATRIX(glTranslate(tf3), Projective3f(tf3).matrix());
    
    Translation<double,3> td3;  td3.vector().setRandom();
    VERIFY_MATRIX(glTranslate(td3), Projective3d(td3).matrix());
  }
  
  // scaling (2D and 3D)
  {
    Vector2f vf2; vf2.setRandom(); Vector3f vf23; vf23 << vf2, 1;
    VERIFY_MATRIX(glScale(vf2), Projective3f(Scaling(vf23)).matrix());
    Vector2d vd2; vd2.setRandom(); Vector3d vd23; vd23 << vd2, 1;
    VERIFY_MATRIX(glScale(vd2), Projective3d(Scaling(vd23)).matrix());
    
    Vector3f vf3; vf3.setRandom();
    VERIFY_MATRIX(glScale(vf3), Projective3f(Scaling(vf3)).matrix());
    Vector3d vd3; vd3.setRandom();
    VERIFY_MATRIX(glScale(vd3), Projective3d(Scaling(vd3)).matrix());
    
    UniformScaling<float> usf(ei_random<float>());
    VERIFY_MATRIX(glScale(usf), Projective3f(usf).matrix());
    
    UniformScaling<double> usd(ei_random<double>());
    VERIFY_MATRIX(glScale(usd), Projective3d(usd).matrix());
  }
  
}
