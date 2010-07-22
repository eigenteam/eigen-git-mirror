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

#include <iostream>
#include <GL/glut.h>
#include <Eigen/OpenGLSupport>
using namespace Eigen;

int main(int argc, char **argv)
{
  glutInit(&argc, argv);
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
  
  Quaterniond qd;
  glRotate(qd);
  
  Matrix4f m44;
  glLoadMatrix(m44);
  glMultMatrix(m44);
  
  return 0;
}
