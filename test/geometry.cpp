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
#include <Eigen/LU>

template<typename Scalar> void geometry(void)
{
  /* this test covers the following files:
     Cross.h Quaternion.h, Transform.cpp
  */

  typedef Matrix<Scalar,2,2> Matrix2;
  typedef Matrix<Scalar,3,3> Matrix3;
  typedef Matrix<Scalar,4,4> Matrix4;
  typedef Matrix<Scalar,2,1> Vector2;
  typedef Matrix<Scalar,3,1> Vector3;
  typedef Matrix<Scalar,4,1> Vector4;
  typedef Quaternion<Scalar> Quaternion;
  typedef AngleAxis<Scalar> AngleAxis;

  Quaternion q1, q2;
  Vector3 v0 = test_random_matrix<Vector3>(),
    v1 = test_random_matrix<Vector3>(),
    v2 = test_random_matrix<Vector3>();
  Vector2 u0 = test_random_matrix<Vector2>();
  Matrix3 matrot1;

  Scalar a = ei_random<Scalar>(-M_PI, M_PI);

  // cross product
  VERIFY_IS_MUCH_SMALLER_THAN(v1.cross(v2).dot(v1), Scalar(1));
  Matrix3 m;
  m << v0.normalized(),
      (v0.cross(v1)).normalized(),
      (v0.cross(v1).cross(v0)).normalized();
  VERIFY(m.isUnitary());

  // unitOrthogonal
  VERIFY_IS_MUCH_SMALLER_THAN(u0.unitOrthogonal().dot(u0), Scalar(1));
  VERIFY_IS_MUCH_SMALLER_THAN(v0.unitOrthogonal().dot(v0), Scalar(1));
  VERIFY_IS_APPROX(u0.unitOrthogonal().norm(), Scalar(1));
  VERIFY_IS_APPROX(v0.unitOrthogonal().norm(), Scalar(1));


  VERIFY_IS_APPROX(v0, AngleAxis(a, v0.normalized()) * v0);
  VERIFY_IS_APPROX(-v0, AngleAxis(M_PI, v0.unitOrthogonal()) * v0);
  VERIFY_IS_APPROX(cos(a)*v0.norm2(), v0.dot(AngleAxis(a, v0.unitOrthogonal()) * v0));
  m = AngleAxis(a, v0.normalized()).toRotationMatrix().adjoint();
  VERIFY_IS_APPROX(Matrix3::Identity(), m * AngleAxis(a, v0.normalized()));
  VERIFY_IS_APPROX(Matrix3::Identity(), AngleAxis(a, v0.normalized()) * m);

  q1 = AngleAxis(a, v0.normalized());
  q2 = AngleAxis(a, v1.normalized());

  // rotation matrix conversion
  VERIFY_IS_APPROX(q1 * v2, q1.toRotationMatrix() * v2);
  VERIFY_IS_APPROX(q1 * q2 * v2,
    q1.toRotationMatrix() * q2.toRotationMatrix() * v2);
  VERIFY( !(q2 * q1 * v2).isApprox(
    q1.toRotationMatrix() * q2.toRotationMatrix() * v2));
  q2 = q1.toRotationMatrix();
  VERIFY_IS_APPROX(q1*v1,q2*v1);

  matrot1 = AngleAxis(0.1, Vector3::UnitX())
          * AngleAxis(0.2, Vector3::UnitY())
          * AngleAxis(0.3, Vector3::UnitZ());
  VERIFY_IS_APPROX(matrot1 * v1,
       AngleAxis(0.1, Vector3(1,0,0)).toRotationMatrix()
    * (AngleAxis(0.2, Vector3(0,1,0)).toRotationMatrix()
    * (AngleAxis(0.3, Vector3(0,0,1)).toRotationMatrix() * v1)));

  // angle-axis conversion
  AngleAxis aa = q1;
  VERIFY_IS_APPROX(q1 * v1, Quaternion(aa) * v1);
  VERIFY_IS_NOT_APPROX(q1 * v1, Quaternion(AngleAxis(aa.angle()*2,aa.axis())) * v1);

  // from two vector creation
  VERIFY_IS_APPROX(v2.normalized(),(q2.setFromTwoVectors(v1,v2)*v1).normalized());
  VERIFY_IS_APPROX(v2.normalized(),(q2.setFromTwoVectors(v1,v2)*v1).normalized());

  // inverse and conjugate
  VERIFY_IS_APPROX(q1 * (q1.inverse() * v1), v1);
  VERIFY_IS_APPROX(q1 * (q1.conjugate() * v1), v1);

  // AngleAxis
  VERIFY_IS_APPROX(AngleAxis(a,v1.normalized()).toRotationMatrix(),
    Quaternion(AngleAxis(a,v1.normalized())).toRotationMatrix());

  AngleAxis aa1;
  m = q1.toRotationMatrix();
  aa1 = m;
  VERIFY_IS_APPROX(AngleAxis(m).toRotationMatrix(),
    Quaternion(m).toRotationMatrix());


  // Transform
  // TODO complete the tests !
  typedef Transform<Scalar,2> Transform2;
  typedef Transform<Scalar,3> Transform3;

  a = 0;
  while (ei_abs(a)<0.1)
    a = ei_random<Scalar>(-0.4*M_PI, 0.4*M_PI);
  q1 = AngleAxis(a, v0.normalized());
  Transform3 t0, t1, t2;
  t0.setIdentity();
  t0.linear() = q1.toRotationMatrix();
  t1.setIdentity();
  t1.linear() = q1.toRotationMatrix();

  v0 << 50, 2, 1;//= test_random_matrix<Vector3>().cwiseProduct(Vector3(10,2,0.5));
  t0.scale(v0);
  t1.prescale(v0);

  VERIFY_IS_APPROX( (t0 * Vector3(1,0,0)).norm(), v0.x());
  VERIFY(!ei_isApprox((t1 * Vector3(1,0,0)).norm(), v0.x()));

  t0.setIdentity();
  t1.setIdentity();
  v1 << 1, 2, 3;
  t0.linear() = q1.toRotationMatrix();
  t0.pretranslate(v0);
  t0.scale(v1);
  t1.linear() = q1.conjugate().toRotationMatrix();
  t1.prescale(v1.cwise().inverse());
  t1.translate(-v0);

  VERIFY((t0.matrix() * t1.matrix()).isIdentity(test_precision<Scalar>()));

  t1.fromPositionOrientationScale(v0, q1, v1);
  VERIFY_IS_APPROX(t1.matrix(), t0.matrix());

  // 2D transformation
  Transform2 t20, t21;
  Vector2 v20 = test_random_matrix<Vector2>();
  Vector2 v21 = test_random_matrix<Vector2>();
  for (int k=0; k<2; ++k)
    if (ei_abs(v21[k])<1e-3) v21[k] = 1e-3;
  t21.setIdentity();
  t21.linear() = Rotation2D<Scalar>(a).toRotationMatrix();
  VERIFY_IS_APPROX(t20.fromPositionOrientationScale(v20,a,v21).matrix(),
    t21.pretranslate(v20).scale(v21).matrix());

  t21.setIdentity();
  t21.linear() = Rotation2D<Scalar>(-a).toRotationMatrix();
  VERIFY( (t20.fromPositionOrientationScale(v20,a,v21)
        * (t21.prescale(v21.cwise().inverse()).translate(-v20))).isIdentity(test_precision<Scalar>()) );
}

void test_geometry()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST( geometry<float>() );
    CALL_SUBTEST( geometry<double>() );
  }
}
