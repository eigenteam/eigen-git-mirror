// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Tal Hadad <tal_hd@hotmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <unsupported/Eigen/EulerAngles>

using namespace Eigen;

template<typename EulerSystem, typename Scalar>
void verify_euler(const Matrix<Scalar,3,1>& ea)
{
  typedef EulerAngles<Scalar, EulerSystem> EulerAnglesType;
  typedef Matrix<Scalar,3,3> Matrix3;
  typedef Matrix<Scalar,3,1> Vector3;
  typedef AngleAxis<Scalar> AngleAxisx;
  using std::abs;
  
  const int i = EulerSystem::HeadingAxisAbs - 1;
  const int j = EulerSystem::PitchAxisAbs - 1;
  const int k = EulerSystem::RollAxisAbs - 1;
  
  const int iFactor = EulerSystem::IsHeadingOpposite ? -1 : 1;
  const int jFactor = EulerSystem::IsPitchOpposite ? -1 : 1;
  const int kFactor = EulerSystem::IsRollOpposite ? -1 : 1;
  
  const Vector3 I = EulerAnglesType::HeadingAxisVector();
  const Vector3 J = EulerAnglesType::PitchAxisVector();
  const Vector3 K = EulerAnglesType::RollAxisVector();
  
  EulerAnglesType e(ea[0], ea[1], ea[2]);
  
  Matrix3 m(e);
  Vector3 eabis = EulerAnglesType(m).coeffs();
  Vector3 eabis2 = m.eulerAngles(i, j, k);
  eabis2[0] *= iFactor;
  eabis2[1] *= jFactor;
  eabis2[2] *= kFactor;
  
  VERIFY_IS_APPROX(eabis, eabis2);// Verify that our estimation is the same as m.eulerAngles() is
  
  Matrix3 mbis(AngleAxisx(eabis[0], I) * AngleAxisx(eabis[1], J) * AngleAxisx(eabis[2], K));
  VERIFY_IS_APPROX(m,  mbis);
  /* If I==K, and ea[1]==0, then there no unique solution. */ 
  /* The remark apply in the case where I!=K, and |ea[1]| is close to pi/2. */ 
  if( (i!=k || ea[1]!=0) && (i==k || !internal::isApprox(abs(ea[1]),Scalar(EIGEN_PI/2),test_precision<Scalar>())) ) 
    VERIFY((ea-eabis).norm() <= test_precision<Scalar>());
  
  // approx_or_less_than does not work for 0
  VERIFY(0 < eabis[0] || test_isMuchSmallerThan(eabis[0], Scalar(1)));
  VERIFY_IS_APPROX_OR_LESS_THAN(eabis[0], Scalar(EIGEN_PI));
  VERIFY_IS_APPROX_OR_LESS_THAN(-Scalar(EIGEN_PI), eabis[1]);
  VERIFY_IS_APPROX_OR_LESS_THAN(eabis[1], Scalar(EIGEN_PI));
  VERIFY_IS_APPROX_OR_LESS_THAN(-Scalar(EIGEN_PI), eabis[2]);
  VERIFY_IS_APPROX_OR_LESS_THAN(eabis[2], Scalar(EIGEN_PI));
}

template<typename Scalar> void check_all_var(const Matrix<Scalar,3,1>& ea)
{
  verify_euler<EulerSystemXYZ, Scalar>(ea);
  verify_euler<EulerSystemXYX, Scalar>(ea);
  verify_euler<EulerSystemXZY, Scalar>(ea);
  verify_euler<EulerSystemXZX, Scalar>(ea);
  
  verify_euler<EulerSystemYZX, Scalar>(ea);
  verify_euler<EulerSystemYZY, Scalar>(ea);
  verify_euler<EulerSystemYXZ, Scalar>(ea);
  verify_euler<EulerSystemYXY, Scalar>(ea);
  
  verify_euler<EulerSystemZXY, Scalar>(ea);
  verify_euler<EulerSystemZXZ, Scalar>(ea);
  verify_euler<EulerSystemZYX, Scalar>(ea);
  verify_euler<EulerSystemZYZ, Scalar>(ea);
}

template<typename Scalar> void eulerangles()
{
  typedef Matrix<Scalar,3,3> Matrix3;
  typedef Matrix<Scalar,3,1> Vector3;
  typedef Array<Scalar,3,1> Array3;
  typedef Quaternion<Scalar> Quaternionx;
  typedef AngleAxis<Scalar> AngleAxisx;

  Scalar a = internal::random<Scalar>(-Scalar(EIGEN_PI), Scalar(EIGEN_PI));
  Quaternionx q1;
  q1 = AngleAxisx(a, Vector3::Random().normalized());
  Matrix3 m;
  m = q1;
  
  Vector3 ea = m.eulerAngles(0,1,2);
  check_all_var(ea);
  ea = m.eulerAngles(0,1,0);
  check_all_var(ea);
  
  // Check with purely random Quaternion:
  q1.coeffs() = Quaternionx::Coefficients::Random().normalized();
  m = q1;
  ea = m.eulerAngles(0,1,2);
  check_all_var(ea);
  ea = m.eulerAngles(0,1,0);
  check_all_var(ea);
  
  // Check with random angles in range [0:pi]x[-pi:pi]x[-pi:pi].
  ea = (Array3::Random() + Array3(1,0,0))*Scalar(EIGEN_PI)*Array3(0.5,1,1);
  check_all_var(ea);
  
  ea[2] = ea[0] = internal::random<Scalar>(0,Scalar(EIGEN_PI));
  check_all_var(ea);
  
  ea[0] = ea[1] = internal::random<Scalar>(0,Scalar(EIGEN_PI));
  check_all_var(ea);
  
  ea[1] = 0;
  check_all_var(ea);
  
  ea.head(2).setZero();
  check_all_var(ea);
  
  ea.setZero();
  check_all_var(ea);
}

void test_EulerAngles()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( eulerangles<float>() );
    CALL_SUBTEST_2( eulerangles<double>() );
  }
}
