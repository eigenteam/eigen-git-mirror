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
void verify_euler_ranged(const Matrix<Scalar,3,1>& ea,
  bool positiveRangeHeading, bool positiveRangePitch, bool positiveRangeRoll)
{
  typedef EulerAngles<Scalar, EulerSystem> EulerAnglesType;
  typedef Matrix<Scalar,3,3> Matrix3;
  typedef Matrix<Scalar,3,1> Vector3;
  typedef Quaternion<Scalar> QuaternionType;
  typedef AngleAxis<Scalar> AngleAxisType;
  using std::abs;
  
  Scalar headingRangeStart, headingRangeEnd;
  Scalar pitchRangeStart, pitchRangeEnd;
  Scalar rollRangeStart, rollRangeEnd;
  
  if (positiveRangeHeading)
  {
    headingRangeStart = Scalar(0);
    headingRangeEnd = Scalar(2 * EIGEN_PI);
  }
  else
  {
    headingRangeStart = -Scalar(EIGEN_PI);
    headingRangeEnd = Scalar(EIGEN_PI);
  }
  
  if (positiveRangePitch)
  {
    pitchRangeStart = Scalar(0);
    pitchRangeEnd = Scalar(2 * EIGEN_PI);
  }
  else
  {
    pitchRangeStart = -Scalar(EIGEN_PI);
    pitchRangeEnd = Scalar(EIGEN_PI);
  }
  
  if (positiveRangeRoll)
  {
    rollRangeStart = Scalar(0);
    rollRangeEnd = Scalar(2 * EIGEN_PI);
  }
  else
  {
    rollRangeStart = -Scalar(EIGEN_PI);
    rollRangeEnd = Scalar(EIGEN_PI);
  }
  
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
  Vector3 eabis = EulerAnglesType(m, positiveRangeHeading, positiveRangePitch, positiveRangeRoll).coeffs();
  
  // Check that eabis in range
  VERIFY(headingRangeStart <= eabis[0] && eabis[0] <= headingRangeEnd);
  VERIFY(pitchRangeStart <= eabis[1] && eabis[1] <= pitchRangeEnd);
  VERIFY(rollRangeStart <= eabis[2] && eabis[2] <= rollRangeEnd);
  
  Vector3 eabis2 = m.eulerAngles(i, j, k);
  
  // Invert the relevant axes
  eabis2[0] *= iFactor;
  eabis2[1] *= jFactor;
  eabis2[2] *= kFactor;
  
  // Saturate the angles to the correct range
  if (positiveRangeHeading && (eabis2[0] < 0))
    eabis2[0] += Scalar(2 * EIGEN_PI);
  if (positiveRangePitch && (eabis2[1] < 0))
    eabis2[1] += Scalar(2 * EIGEN_PI);
  if (positiveRangeRoll && (eabis2[2] < 0))
    eabis2[2] += Scalar(2 * EIGEN_PI);
  
  VERIFY_IS_APPROX(eabis, eabis2);// Verify that our estimation is the same as m.eulerAngles() is
  
  Matrix3 mbis(AngleAxisType(eabis[0], I) * AngleAxisType(eabis[1], J) * AngleAxisType(eabis[2], K));
  VERIFY_IS_APPROX(m,  mbis);
  
  // Tests that are only relevant for no possitive range
  if (!(positiveRangeHeading || positiveRangePitch || positiveRangeRoll))
  {
    /* If I==K, and ea[1]==0, then there no unique solution. */ 
    /* The remark apply in the case where I!=K, and |ea[1]| is close to pi/2. */ 
    if( (i!=k || ea[1]!=0) && (i==k || !internal::isApprox(abs(ea[1]),Scalar(EIGEN_PI/2),test_precision<Scalar>())) ) 
      VERIFY((ea-eabis).norm() <= test_precision<Scalar>());
    
    // approx_or_less_than does not work for 0
    VERIFY(0 < eabis[0] || test_isMuchSmallerThan(eabis[0], Scalar(1)));
  }
  
  // Quaternions
  QuaternionType q(e);
  eabis = EulerAnglesType(q, positiveRangeHeading, positiveRangePitch, positiveRangeRoll).coeffs();
  VERIFY_IS_APPROX(eabis, eabis2);// Verify that the euler angles are still the same
}

template<typename EulerSystem, typename Scalar>
void verify_euler(const Matrix<Scalar,3,1>& ea)
{
  verify_euler_ranged<EulerSystem>(ea, false, false, false);
  verify_euler_ranged<EulerSystem>(ea, false, false, true);
  verify_euler_ranged<EulerSystem>(ea, false, true, false);
  verify_euler_ranged<EulerSystem>(ea, false, true, true);
  verify_euler_ranged<EulerSystem>(ea, true, false, false);
  verify_euler_ranged<EulerSystem>(ea, true, false, true);
  verify_euler_ranged<EulerSystem>(ea, true, true, false);
  verify_euler_ranged<EulerSystem>(ea, true, true, true);
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
