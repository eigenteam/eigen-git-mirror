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

// Verify that x is in the approxed range [a, b]
#define VERIFY_APPROXED_RANGE(a, x, b) \
	do { \
	VERIFY_IS_APPROX_OR_LESS_THAN(a, x); \
	VERIFY_IS_APPROX_OR_LESS_THAN(x, b); \
	} while(0)

template<typename EulerSystem, typename Scalar>
void verify_euler(const Matrix<Scalar,3,1>& ea)
{
  typedef EulerAngles<Scalar, EulerSystem> EulerAnglesType;
  typedef Matrix<Scalar,3,3> Matrix3;
  typedef Matrix<Scalar,3,1> Vector3;
  typedef Quaternion<Scalar> QuaternionType;
  typedef AngleAxis<Scalar> AngleAxisType;
  
  const Scalar ONE = Scalar(1);
  const Scalar HALF_PI = Scalar(EIGEN_PI / 2);
  const Scalar PI = Scalar(EIGEN_PI);
  
  Scalar betaRangeStart, betaRangeEnd;
  if (EulerSystem::IsTaitBryan)
  {
    betaRangeStart = -HALF_PI;
    betaRangeEnd = HALF_PI;
  }
  else
  {
    betaRangeStart = -PI;
    betaRangeEnd = PI;
  }
  
  const Vector3 I = EulerAnglesType::AlphaAxisVector();
  const Vector3 J = EulerAnglesType::BetaAxisVector();
  const Vector3 K = EulerAnglesType::GammaAxisVector();
  
  EulerAnglesType e(ea[0], ea[1], ea[2]);

  Matrix3 m(e);

  Vector3 eabis = static_cast<EulerAnglesType>(m).angles();
  
  // Check that eabis in range
  VERIFY_APPROXED_RANGE(-PI, eabis[0], PI);
  VERIFY_APPROXED_RANGE(betaRangeStart, eabis[1], betaRangeEnd);
  VERIFY_APPROXED_RANGE(-PI, eabis[2], PI);

  Matrix3 mbis(AngleAxisType(eabis[0], I) * AngleAxisType(eabis[1], J) * AngleAxisType(eabis[2], K));
  VERIFY_IS_APPROX(m,  mbis);

  // Test if ea and eabis are the same
  // Need to check both singular and non-singular cases
  // There are two singular cases.
  // 1. When I==K and sin(ea(1)) == 0
  // 2. When I!=K and cos(ea(1)) == 0

  // Tests that are only relevant for no positive range
  /*if (!(positiveRangeAlpha || positiveRangeGamma))
  {
    // If I==K, and ea[1]==0, then there no unique solution.
    // The remark apply in the case where I!=K, and |ea[1]| is close to pi/2.
    if( (i!=k || ea[1]!=0) && (i==k || !internal::isApprox(abs(ea[1]),Scalar(EIGEN_PI/2),test_precision<Scalar>())) ) 
      VERIFY((ea-eabis).norm() <= test_precision<Scalar>());
    
    // approx_or_less_than does not work for 0
    VERIFY(0 < eabis[0] || VERIFY_IS_MUCH_SMALLER_THAN(eabis[0], Scalar(1)));
  }*/
  
  // Quaternions
  QuaternionType q(e);
  eabis = static_cast<EulerAnglesType>(q).angles();
  QuaternionType qbis(AngleAxisType(eabis[0], I) * AngleAxisType(eabis[1], J) * AngleAxisType(eabis[2], K));
  VERIFY_IS_APPROX(std::abs(q.dot(qbis)), ONE);
  //VERIFY_IS_APPROX(eabis, eabis2);// Verify that the euler angles are still the same
}

template<typename Scalar> void check_all_var(const Matrix<Scalar,3,1>& ea)
{
  verify_euler<EulerSystemXYZ>(ea);
  verify_euler<EulerSystemXYX>(ea);
  verify_euler<EulerSystemXZY>(ea);
  verify_euler<EulerSystemXZX>(ea);
  
  verify_euler<EulerSystemYZX>(ea);
  verify_euler<EulerSystemYZY>(ea);
  verify_euler<EulerSystemYXZ>(ea);
  verify_euler<EulerSystemYXY>(ea);
  
  verify_euler<EulerSystemZXY>(ea);
  verify_euler<EulerSystemZXZ>(ea);
  verify_euler<EulerSystemZYX>(ea);
  verify_euler<EulerSystemZYZ>(ea);
  
  // TODO: Test negative axes as well! (only test if the angles get negative when needed)
}

template<typename Scalar> void eulerangles()
{
  typedef Matrix<Scalar,3,3> Matrix3;
  typedef Matrix<Scalar,3,1> Vector3;
  typedef Array<Scalar,3,1> Array3;
  typedef Quaternion<Scalar> Quaternionx;
  typedef AngleAxis<Scalar> AngleAxisType;

  Scalar a = internal::random<Scalar>(-Scalar(EIGEN_PI), Scalar(EIGEN_PI));
  Quaternionx q1;
  q1 = AngleAxisType(a, Vector3::Random().normalized());
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
