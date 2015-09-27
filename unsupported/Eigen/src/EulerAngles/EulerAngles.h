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
      /** the scalar type of the coefficients */
      typedef _Scalar Scalar;
      typedef _System System;
    
      typedef Matrix<Scalar,3,3> Matrix3;
      typedef Matrix<Scalar,3,1> Vector3;
      typedef Quaternion<Scalar> QuaternionType;
      typedef AngleAxis<Scalar> AngleAxisType;

    protected:

      Vector3 m_angles;

    public:

      EulerAngles() {}
      inline EulerAngles(Scalar a0, Scalar a1, Scalar a2) : m_angles(a0, a1, a2) {}
      inline EulerAngles(Vector3 angles) : m_angles(angles) {}
      inline EulerAngles(const QuaternionType& q) { *this = q; }
      inline EulerAngles(const AngleAxisType& aa) { *this = aa; }
      template<typename Derived>
      inline EulerAngles(const MatrixBase<Derived>& m) { *this = m; }

      // TODO: Support assignment from euler to euler

      Scalar angle(int i) const { return m_angles.coeff(i); }
      Scalar& angle(int i) { return m_angles.coeffRef(i); }

      const Vector3& coeffs() const { return m_angles; }
      Vector3& coeffs() { return m_angles; }

      // TODO: Add set/get functions

      Scalar h() const { return m_angles[0]; }
      Scalar& h() { return m_angles[0]; }

      Scalar p() const { return m_angles[1]; }
      Scalar& p() { return m_angles[1]; }

      Scalar r() const { return m_angles[2]; }
      Scalar& r() { return m_angles[2]; }

      EulerAngles invert() const
      {
        //m_angles = -m_angles;// I want to do this but there could be an aliasing issue!
        m_angles *= -1;
        
        return *this;
      }

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
      template<typename Derived>
      // TODO: Add booleans which let the user control desired output angles range( (-PI, PI) or [0, 2*PI) )
      EulerAngles& fromRotationMatrix(const MatrixBase<Derived>& m)
      {
        System::eulerAngles(*this, m);
        return *this;
      }
      
      /** Set \c *this from a rotation matrix(i.e. pure orthogonal matrix with determinent of +1).
        */
      template<typename Derived>
      EulerAngles& operator=(const MatrixBase<Derived>& mat){
        return fromRotationMatrix(mat);
      }

      // TODO: Assign and construct from another EulerAngle (with different system)
      
      /** Set \c *this from a quaternion.
        * The axis is normalized.
        */
      EulerAngles& operator=(const QuaternionType& q){
        // TODO: Implement it in a better way
        // According to http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler/
        //  we can compute only the needed matrix cells and then convert to euler angles.
        // Currently we compute all matrix cells from quaternion.

        fromRotationMatrix(q.toRotationMatrix());

        // Special case only for ZYX
        /*Scalar y2 = q.y() * q.y();
        m_angles[0] = std::atan2(2*(q.w()*q.z() + q.x()*q.y()), (1 - 2*(y2 + q.z()*q.z())));
        m_angles[1] = std::asin( 2*(q.w()*q.y() - q.z()*q.x()));
        m_angles[2] = std::atan2(2*(q.w()*q.x() + q.y()*q.z()), (1 - 2*(q.x()*q.x() + y2)));
        */
        
        return *this;
      }
      
      /** Set \c *this from AngleAxis \a ea.
       */
      EulerAngles& operator=(const AngleAxisType& ea)
      {
        // TODO: Implement it in a better way
        return *this = ea.toRotationMatrix();
      }

      // TODO: Fix this function, and make it generic
      Matrix3 toRotationMatrix(void) const
      {
        return static_cast<QuaternionType>(*this).toRotationMatrix();
      }
    
      operator QuaternionType() const
      {
        return
          AngleAxisType((System::IsHeadingOpposite ? -1 : 1) * h(), Vector3::Unit(System::HeadingAxisAbs - 1)) *
          AngleAxisType((System::IsPitchOpposite ? -1 : 1) * p(), Vector3::Unit(System::PitchAxisAbs - 1)) *
          AngleAxisType((System::IsRollOpposite ? -1 : 1) * r(), Vector3::Unit(System::RollAxisAbs - 1));
      }
  };

  typedef EulerAngles<double, EulerSystemXYZ> EulerAnglesXYZd;
  typedef EulerAngles<double, EulerSystemXYX> EulerAnglesXYXd;
  typedef EulerAngles<double, EulerSystemXZY> EulerAnglesXZYd;
  typedef EulerAngles<double, EulerSystemXZX> EulerAnglesXZXd;

  typedef EulerAngles<double, EulerSystemYZX> EulerAnglesYZXd;
  typedef EulerAngles<double, EulerSystemYZY> EulerAnglesYZYd;
  typedef EulerAngles<double, EulerSystemYXZ> EulerAnglesYXZd;
  typedef EulerAngles<double, EulerSystemYXY> EulerAnglesYXYd;

  typedef EulerAngles<double, EulerSystemZXY> EulerAnglesZXYd;
  typedef EulerAngles<double, EulerSystemZXZ> EulerAnglesZXZd;
  typedef EulerAngles<double, EulerSystemZYX> EulerAnglesZYXd;
  typedef EulerAngles<double, EulerSystemZYZ> EulerAnglesZYZd;

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
