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

#include "main.h"
#include <Eigen/Geometry>

template<typename Scalar> void geometry(void)
{
  /* this test covers the following files:
     Cross.h Quaternion.h
  */

  typedef Matrix<Scalar,3,3> Matrix3;
  typedef Matrix<Scalar,4,4> Matrix4;
  typedef Matrix<Scalar,3,1> Vector3;
  typedef Matrix<Scalar,4,1> Vector4;
  typedef Quaternion<Scalar> Quaternion;

  Quaternion q1, q2;
  Vector3 v0 = Vector3::random(),
    v1 = Vector3::random(),
    v2 = Vector3::random();

  Scalar a;

  q1.fromAngleAxis(ei_random<Scalar>(-M_PI, M_PI), v0.normalized());
  q2.fromAngleAxis(ei_random<Scalar>(-M_PI, M_PI), v1.normalized());

  // rotation matrix conversion
  VERIFY_IS_APPROX(q1 * v2, q1.toRotationMatrix() * v2);
  VERIFY_IS_APPROX(q1 * q2 * v2,
    q1.toRotationMatrix() * q2.toRotationMatrix() * v2);
  VERIFY_IS_NOT_APPROX(q2 * q1 * v2,
    q1.toRotationMatrix() * q2.toRotationMatrix() * v2);
  q2.fromRotationMatrix(q1.toRotationMatrix());
  VERIFY_IS_APPROX(q1*v1,q2*v1);

  // Euler angle conversion
  VERIFY_IS_APPROX(q2.fromEulerAngles(q1.toEulerAngles()) * v1, q1 * v1);
  v2 = q2.toEulerAngles();
  VERIFY_IS_APPROX(q2.fromEulerAngles(v2).toEulerAngles(), v2);
  VERIFY_IS_NOT_APPROX(q2.fromEulerAngles(v2.cwiseProduct(Vector3(0.2,-0.2,1))).toEulerAngles(), v2);

  // angle-axis conversion
  q1.toAngleAxis(a, v2);
  VERIFY_IS_APPROX(q1 * v1, q2.fromAngleAxis(a,v2) * v1);
  VERIFY_IS_NOT_APPROX(q1 * v1, q2.fromAngleAxis(2*a,v2) * v1);

  // from two vector creation
  VERIFY_IS_APPROX(v2.normalized(),(q2.fromTwoVectors(v1,v2)*v1).normalized());
  VERIFY_IS_APPROX(v2.normalized(),(q2.fromTwoVectors(v1,v2)*v1).normalized());

  // cross product
  VERIFY_IS_MUCH_SMALLER_THAN(v1.cross(v2).dot(v1), Scalar(1));
  Matrix3 m;
  m << v0.normalized(),
      (v0.cross(v1)).normalized(),
      (v0.cross(v1).cross(v0)).normalized();
  VERIFY(m.isOrtho());
}

void test_geometry()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST( geometry<float>() );
    CALL_SUBTEST( geometry<double>() );
  }
}
