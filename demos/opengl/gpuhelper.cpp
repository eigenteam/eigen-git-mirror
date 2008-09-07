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

#include "gpuhelper.h"
#include <GL/glu.h>
// PLEASE don't look at this old code... ;)

#include <fstream>
#include <algorithm>

GpuHelper gpu;

//--------------------------------------------------------------------------------
// icosahedron
//--------------------------------------------------------------------------------
#define X .525731112119133606
#define Z .850650808352039932

static GLfloat vdata[12][3] = {
   {-X, 0.0, Z}, {X, 0.0, Z}, {-X, 0.0, -Z}, {X, 0.0, -Z},
   {0.0, Z, X}, {0.0, Z, -X}, {0.0, -Z, X}, {0.0, -Z, -X},
   {Z, X, 0.0}, {-Z, X, 0.0}, {Z, -X, 0.0}, {-Z, -X, 0.0}
};

static GLint tindices[20][3] = {
   {0,4,1}, {0,9,4}, {9,5,4}, {4,5,8}, {4,8,1},
   {8,10,1}, {8,3,10}, {5,3,8}, {5,2,3}, {2,7,3},
   {7,10,3}, {7,6,10}, {7,11,6}, {11,0,6}, {0,1,6},
   {6,1,10}, {9,0,11}, {9,11,2}, {9,2,5}, {7,2,11} };
//--------------------------------------------------------------------------------


GpuHelper::GpuHelper()
{
    mVpWidth = mVpHeight = 0;
    mCurrentMatrixTarget = 0;
    mInitialized = false;
}

GpuHelper::~GpuHelper()
{
}

void GpuHelper::pushProjectionMode2D(ProjectionMode2D pm)
{
    // switch to 2D projection
    pushMatrix(Matrix4f::Identity(),GL_PROJECTION);

    if(pm==PM_Normalized)
    {
        //glOrtho(-1., 1., -1., 1., 0., 1.);
    }
    else if(pm==PM_Viewport)
    {
        GLint vp[4];
        glGetIntegerv(GL_VIEWPORT, vp);
        glOrtho(0., vp[2], 0., vp[3], -1., 1.);
    }

    pushMatrix(Matrix4f::Identity(),GL_MODELVIEW);
}

void GpuHelper::popProjectionMode2D(void)
{
    popMatrix(GL_PROJECTION);
    popMatrix(GL_MODELVIEW);
}

void GpuHelper::drawVector(const Vector3f& position, const Vector3f& vec, const Color& color, float aspect /* = 50.*/)
{
    static GLUquadricObj *cylindre = gluNewQuadric();
    glColor4fv(color.data());
    float length = vec.norm();
    pushMatrix(GL_MODELVIEW);
    glTranslatef(position.x(), position.y(), position.z());
    Vector3f ax = Matrix3f::Identity().col(2).cross(vec);
    ax.normalize();
    Vector3f tmp = vec;
    tmp.normalize();
    float angle = 180.f/M_PI * acos(tmp.z());
    if (angle>1e-3)
        glRotatef(angle, ax.x(), ax.y(), ax.z());
    gluCylinder(cylindre, length/aspect, length/aspect, 0.8*length, 10, 10);
    glTranslatef(0.0,0.0,0.8*length);
    gluCylinder(cylindre, 2.0*length/aspect, 0.0, 0.2*length, 10, 10);

    popMatrix(GL_MODELVIEW);
}

void GpuHelper::drawVectorBox(const Vector3f& position, const Vector3f& vec, const Color& color, float aspect)
{
    static GLUquadricObj *cylindre = gluNewQuadric();
    glColor4fv(color.data());
    float length = vec.norm();
    pushMatrix(GL_MODELVIEW);
    glTranslatef(position.x(), position.y(), position.z());
    Vector3f ax = Matrix3f::Identity().col(2).cross(vec);
    ax.normalize();
    Vector3f tmp = vec;
    tmp.normalize();
    float angle = 180.f/M_PI * acos(tmp.z());
    if (angle>1e-3)
        glRotatef(angle, ax.x(), ax.y(), ax.z());
    gluCylinder(cylindre, length/aspect, length/aspect, 0.8*length, 10, 10);
    glTranslatef(0.0,0.0,0.8*length);
    glScalef(4.0*length/aspect,4.0*length/aspect,4.0*length/aspect);
    drawUnitCube();
    popMatrix(GL_MODELVIEW);
}

void GpuHelper::drawUnitCube(void)
{
    static float vertices[][3] = {
        {-0.5,-0.5,-0.5},
        { 0.5,-0.5,-0.5},
        {-0.5, 0.5,-0.5},
        { 0.5, 0.5,-0.5},
        {-0.5,-0.5, 0.5},
        { 0.5,-0.5, 0.5},
        {-0.5, 0.5, 0.5},
        { 0.5, 0.5, 0.5}};

    glBegin(GL_QUADS);
    glNormal3f(0,0,-1); glVertex3fv(vertices[0]); glVertex3fv(vertices[2]); glVertex3fv(vertices[3]); glVertex3fv(vertices[1]);
    glNormal3f(0,0, 1); glVertex3fv(vertices[4]); glVertex3fv(vertices[5]); glVertex3fv(vertices[7]); glVertex3fv(vertices[6]);
    glNormal3f(0,-1,0); glVertex3fv(vertices[0]); glVertex3fv(vertices[1]); glVertex3fv(vertices[5]); glVertex3fv(vertices[4]);
    glNormal3f(0, 1,0); glVertex3fv(vertices[2]); glVertex3fv(vertices[6]); glVertex3fv(vertices[7]); glVertex3fv(vertices[3]);
    glNormal3f(-1,0,0); glVertex3fv(vertices[0]); glVertex3fv(vertices[4]); glVertex3fv(vertices[6]); glVertex3fv(vertices[2]);
    glNormal3f( 1,0,0); glVertex3fv(vertices[1]); glVertex3fv(vertices[3]); glVertex3fv(vertices[7]); glVertex3fv(vertices[5]);
    glEnd();
}

void _normalize(float* v)
{
    float s = 1.f/ei_sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    for (uint k=0; k<3; ++k)
        v[k] *= s;
}

void _subdivide(float *v1, float *v2, float *v3, long depth)
{
    GLfloat v12[3], v23[3], v31[3];
    GLint i;

    if (depth == 0) {
        //drawtriangle(v1, v2, v3);
        glNormal3fv(v1);
        glVertex3fv(v1);

        glNormal3fv(v3);
        glVertex3fv(v3);

        glNormal3fv(v2);
        glVertex3fv(v2);

        return;
    }
    for (i = 0; i < 3; i++) {
         v12[i] = v1[i]+v2[i];
         v23[i] = v2[i]+v3[i];
         v31[i] = v3[i]+v1[i];
    }
    _normalize(v12);
    _normalize(v23);
    _normalize(v31);
    _subdivide(v1, v12, v31, depth-1);
    _subdivide(v2, v23, v12, depth-1);
    _subdivide(v3, v31, v23, depth-1);
    _subdivide(v12, v23, v31, depth-1);
}

void GpuHelper::drawUnitLightSphere(int level)
{
  static int dlId = 0;
  if (!dlId)
  {
    dlId = glGenLists(1);
    glNewList(dlId, GL_COMPILE);
    glBegin(GL_TRIANGLES);
    for (int i = 0; i < 20; i++)
    {
      _subdivide(&vdata[tindices[i][0]][0], &vdata[tindices[i][1]][0], &vdata[tindices[i][2]][0], 1);
    }
    glEnd();
    glEndList();
  }
  glCallList(dlId);
}


