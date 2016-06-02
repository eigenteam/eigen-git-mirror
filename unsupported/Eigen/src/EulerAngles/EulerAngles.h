// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Tal Hadad <tal_hd@hotmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_EULERANGLESCLASS_H// TODO: Fix previous "EIGEN_EULERANGLES_H" definition?
#define EIGEN_EULERANGLESCLASS_H

namespace Eigen
{
  /*template<typename Other,
         int OtherRows=Other::RowsAtCompileTime,
         int OtherCols=Other::ColsAtCompileTime>
  struct ei_eulerangles_assign_impl;*/

  /** \class EulerAngles
    *
    * \brief Represents a rotation in a 3 dimensional space as three Euler angles
    *
    * \param _Scalar the scalar type, i.e., the type of the angles.
    *
    * \sa cl
    */
  template <typename _Scalar, class _System>
  class EulerAngles : public RotationBase<EulerAngles<_Scalar, _System>, 3>
  {
    public:
      /** the scalar type of the angles */
      typedef _Scalar Scalar;
      typedef _System System;
    
      typedef Matrix<Scalar,3,3> Matrix3;
      typedef Matrix<Scalar,3,1> Vector3;
      typedef Quaternion<Scalar> QuaternionType;
      typedef AngleAxis<Scalar> AngleAxisType;
      
      static Vector3 AlphaAxisVector() {
        const Vector3& u = Vector3::Unit(System::AlphaAxisAbs - 1);
        return System::IsAlphaOpposite ? -u : u;
      }
      
      static Vector3 BetaAxisVector() {
        const Vector3& u = Vector3::Unit(System::BetaAxisAbs - 1);
        return System::IsBetaOpposite ? -u : u;
      }
      
      static Vector3 GammaAxisVector() {
        const Vector3& u = Vector3::Unit(System::GammaAxisAbs - 1);
        return System::IsGammaOpposite ? -u : u;
      }

    private:
      Vector3 m_angles;

    public:

      EulerAngles() {}
      inline EulerAngles(Scalar alpha, Scalar beta, Scalar gamma) : m_angles(alpha, beta, gamma) {}
      
      template<typename Derived>
      inline EulerAngles(const MatrixBase<Derived>& m) { *this = m; }
      
      template<typename Derived>
      inline EulerAngles(
        const MatrixBase<Derived>& m,
        bool positiveRangeAlpha,
        bool positiveRangeBeta,
        bool positiveRangeGamma) {
        
        System::CalcEulerAngles(*this, m, positiveRangeAlpha, positiveRangeBeta, positiveRangeGamma);
      }
      
      template<typename Derived>
      inline EulerAngles(const RotationBase<Derived, 3>& rot) { *this = rot; }
      
      template<typename Derived>
      inline EulerAngles(
        const RotationBase<Derived, 3>& rot,
        bool positiveRangeAlpha,
        bool positiveRangeBeta,
        bool positiveRangeGamma) {
        
        System::CalcEulerAngles(*this, rot.toRotationMatrix(), positiveRangeAlpha, positiveRangeBeta, positiveRangeGamma);
      }

      const Vector3& angles() const { return m_angles; }
      Vector3& angles() { return m_angles; }

      Scalar alpha() const { return m_angles[0]; }
      Scalar& alpha() { return m_angles[0]; }

      Scalar beta() const { return m_angles[1]; }
      Scalar& beta() { return m_angles[1]; }

      Scalar gamma() const { return m_angles[2]; }
      Scalar& gamma() { return m_angles[2]; }

      EulerAngles inverse() const
      {
        EulerAngles res;
        res.m_angles = -m_angles;
        return res;
      }

      EulerAngles operator -() const
      {
        return inverse();
      }
      
      /** Constructs and \returns an equivalent 3x3 rotation matrix.
        */
      template<
        bool PositiveRangeAlpha,
        bool PositiveRangeBeta,
        bool PositiveRangeGamma,
        typename Derived>
      static EulerAngles FromRotation(const MatrixBase<Derived>& m)
      {
        EulerAngles e;
        System::CalcEulerAngles<PositiveRangeAlpha, PositiveRangeBeta, PositiveRangeGamma>(e, m);
        return e;
      }
      
      template<
        bool PositiveRangeAlpha,
        bool PositiveRangeBeta,
        bool PositiveRangeGamma,
        typename Derived>
      static EulerAngles& FromRotation(const RotationBase<Derived, 3>& rot)
      {
        return FromRotation<PositiveRangeAlpha, PositiveRangeBeta, PositiveRangeGamma>(rot.toRotationMatrix());
      }
      
      /*EulerAngles& fromQuaternion(const QuaternionType& q)
      {
        // TODO: Implement it in a faster way for quaternions
        // According to http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler/
        //  we can compute only the needed matrix cells and then convert to euler angles. (see ZYX example below)
        // Currently we compute all matrix cells from quaternion.

        // Special case only for ZYX
        //Scalar y2 = q.y() * q.y();
        //m_angles[0] = std::atan2(2*(q.w()*q.z() + q.x()*q.y()), (1 - 2*(y2 + q.z()*q.z())));
        //m_angles[1] = std::asin( 2*(q.w()*q.y() - q.z()*q.x()));
        //m_angles[2] = std::atan2(2*(q.w()*q.x() + q.y()*q.z()), (1 - 2*(q.x()*q.x() + y2)));
      }*/
      
      /** Set \c *this from a rotation matrix(i.e. pure orthogonal matrix with determinent of +1).
        */
      template<typename Derived>
      EulerAngles& operator=(const MatrixBase<Derived>& m) {
        System::CalcEulerAngles(*this, m);
        return *this;
      }

      // TODO: Assign and construct from another EulerAngles (with different system)
      
      /** Set \c *this from a rotation.
        */
      template<typename Derived>
      EulerAngles& operator=(const RotationBase<Derived, 3>& rot) {
        System::CalcEulerAngles(*this, rot.toRotationMatrix());
        return *this;
      }
      
      // TODO: Support isApprox function

      Matrix3 toRotationMatrix() const
      {
        return static_cast<QuaternionType>(*this).toRotationMatrix();
      }

      QuaternionType toQuaternion() const
      {
        return
          AngleAxisType(alpha(), AlphaAxisVector()) *
          AngleAxisType(beta(), BetaAxisVector()) *
          AngleAxisType(gamma(), GammaAxisVector());
      }
    
      operator QuaternionType() const
      {
        return toQuaternion();
      }
  };

#define EIGEN_EULER_ANGLES_SINGLE_TYPEDEF(SYSTEM, SCALAR_TYPE, SCALAR_POSTFIX) \
  typedef EulerAngles<SCALAR_TYPE, SYSTEM> SYSTEM##SCALAR_POSTFIX;

#define EIGEN_EULER_ANGLES_TYPEDEFS(SCALAR_TYPE, SCALAR_POSTFIX) \
  EIGEN_EULER_ANGLES_SINGLE_TYPEDEF(EulerSystemXYZ, SCALAR_TYPE, SCALAR_POSTFIX) \
  EIGEN_EULER_ANGLES_SINGLE_TYPEDEF(EulerSystemXYX, SCALAR_TYPE, SCALAR_POSTFIX) \
  EIGEN_EULER_ANGLES_SINGLE_TYPEDEF(EulerSystemXZY, SCALAR_TYPE, SCALAR_POSTFIX) \
  EIGEN_EULER_ANGLES_SINGLE_TYPEDEF(EulerSystemXZX, SCALAR_TYPE, SCALAR_POSTFIX) \
 \
  EIGEN_EULER_ANGLES_SINGLE_TYPEDEF(EulerSystemYZX, SCALAR_TYPE, SCALAR_POSTFIX) \
  EIGEN_EULER_ANGLES_SINGLE_TYPEDEF(EulerSystemYZY, SCALAR_TYPE, SCALAR_POSTFIX) \
  EIGEN_EULER_ANGLES_SINGLE_TYPEDEF(EulerSystemYXZ, SCALAR_TYPE, SCALAR_POSTFIX) \
  EIGEN_EULER_ANGLES_SINGLE_TYPEDEF(EulerSystemYXY, SCALAR_TYPE, SCALAR_POSTFIX) \
 \
  EIGEN_EULER_ANGLES_SINGLE_TYPEDEF(EulerSystemZXY, SCALAR_TYPE, SCALAR_POSTFIX) \
  EIGEN_EULER_ANGLES_SINGLE_TYPEDEF(EulerSystemZXZ, SCALAR_TYPE, SCALAR_POSTFIX) \
  EIGEN_EULER_ANGLES_SINGLE_TYPEDEF(EulerSystemZYX, SCALAR_TYPE, SCALAR_POSTFIX) \
  EIGEN_EULER_ANGLES_SINGLE_TYPEDEF(EulerSystemZYZ, SCALAR_TYPE, SCALAR_POSTFIX)

EIGEN_EULER_ANGLES_TYPEDEFS(float, f)
EIGEN_EULER_ANGLES_TYPEDEFS(double, d)

  namespace internal
  {
    template<typename _Scalar, class _System>
    struct traits<EulerAngles<_Scalar, _System> >
    {
      typedef _Scalar Scalar;
    };
  }
  
}

#endif // EIGEN_EULERANGLESCLASS_H
