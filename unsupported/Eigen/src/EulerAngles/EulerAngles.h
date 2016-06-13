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
    * \brief Represents a rotation in a 3 dimensional space as three Euler angles.
    *
    * Euler rotation is a set of three rotation of three angles over three fixed axes, defined by the EulerSystem given as a template parameter.
    * 
    * Here is how intrinsic Euler angles works:
    *  - first, rotate the axes system over the alpha axis in angle alpha
    *  - then, rotate the axes system over the beta axis(which was rotated in the first stage) in angle beta
    *  - then, rotate the axes system over the gamma axis(which was rotated in the two stages above) in angle gamma
    *
    * \note This class support only intrinsic Euler angles for simplicity,
    *  see EulerSystem how to easily overcome it for extrinsic systems.
    *
    * ### Rotation representation and conversions ###
    *
    * It has been proved(see Wikipedia link below) that every rotation can be represented
    *  by Euler angles, but there is no singular representation (e.g. unlike rotation matrices).
    * Therefore, you can convert from Eigen rotation and to them
    *  (including rotation matrices, which is not called "rotations" by Eigen design).
    *
    * Euler angles usually used for:
    *  - convenient human representation of rotation, especially in interactive GUI.
    *  - gimbal systems and robotics
    *  - efficient encoding(i.e. 3 floats only) of rotation for network protocols.
    *
    * However, Euler angles are slow comparing to quaternion or matrices,
    *  because their unnatural math definition, although it's simple for human.
    * To overcome this, this class provide easy movement from the math friendly representation
    *  to the human friendly representation, and vise-versa.
    *
    * All the user need to do is a safe simple C++ type conversion,
    *  and this class take care for the math.
    * Additionally, some axes related computation is done in compile time.
    *
    * ### Convenient user typedefs ###
    *
    * Convenient typedefs for EulerAngles exist for float and double scalar,
    *  in a form of EulerAngles{A}{B}{C}{scalar},
    *  e.g. EulerAnglesXYZd, EulerAnglesZYZf.
    *
    * !TODO! Add examples
    *
    * Only for positive axes{+x,+y,+z} euler systems are have convenient typedef.
    * If you need negative axes{-x,-y,-z}, it is recommended to create you own typedef with
    *  a word that represent what you need, e.g. EulerAnglesUTM (!TODO! make it more clear with example code).
    *
    * ### Additional reading ###
    *
    * If you're want to get more idea about how Euler system work in Eigen see EulerSystem.
    *
    * More information about Euler angles: https://en.wikipedia.org/wiki/Euler_angles
    *
    * \tparam _Scalar the scalar type, i.e., the type of the angles.
    *
    * \tparam _System the EulerSystem to use, which represents the axes of rotation.
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
      
      /** \returns the axis vector of the first (alpha) rotation */
      static Vector3 AlphaAxisVector() {
        const Vector3& u = Vector3::Unit(System::AlphaAxisAbs - 1);
        return System::IsAlphaOpposite ? -u : u;
      }
      
      /** \returns the axis vector of the second (beta) rotation */
      static Vector3 BetaAxisVector() {
        const Vector3& u = Vector3::Unit(System::BetaAxisAbs - 1);
        return System::IsBetaOpposite ? -u : u;
      }
      
      /** \returns the axis vector of the third (gamma) rotation */
      static Vector3 GammaAxisVector() {
        const Vector3& u = Vector3::Unit(System::GammaAxisAbs - 1);
        return System::IsGammaOpposite ? -u : u;
      }

    private:
      Vector3 m_angles;

    public:
      /** Default constructor without initialization. */
      EulerAngles() {}
      /** Constructs and initialize Euler angles(\p alpha, \p beta, \p gamma). */
      EulerAngles(const Scalar& alpha, const Scalar& beta, const Scalar& gamma) :
        m_angles(alpha, beta, gamma) {}
      
      /** Constructs and initialize Euler angles from a 3x3 rotation matrix \p m.
        *
        * \note All angles will be in the range [-PI, PI].
      */
      template<typename Derived>
      EulerAngles(const MatrixBase<Derived>& m) { *this = m; }
      
      /** Constructs and initialize Euler angles from a 3x3 rotation matrix \p m,
        *  with options to choose for each angle the requested range.
        *
        * If possitive range is true, then the specified angle will be in the range [0, +2*PI].
        * Otherwise, the specified angle will be in the range [-PI, +PI].
        *
        * \param m The 3x3 rotation matrix to convert
        * \param positiveRangeAlpha If true, alpha will be in [0, 2*PI]. Otherwise, in [-PI, +PI].
        * \param positiveRangeBeta If true, beta will be in [0, 2*PI]. Otherwise, in [-PI, +PI].
        * \param positiveRangeGamma If true, gamma will be in [0, 2*PI]. Otherwise, in [-PI, +PI].
      */
      template<typename Derived>
      EulerAngles(
        const MatrixBase<Derived>& m,
        bool positiveRangeAlpha,
        bool positiveRangeBeta,
        bool positiveRangeGamma) {
        
        System::CalcEulerAngles(*this, m, positiveRangeAlpha, positiveRangeBeta, positiveRangeGamma);
      }
      
      /** Constructs and initialize Euler angles from a rotation \p rot.
        *
        * \note All angles will be in the range [-PI, PI], unless \p rot is an EulerAngles.
        *  If rot is an EulerAngles, expected EulerAngles range is undefined.
        *  (Use other functions here for enforcing range if this effect is desired)
      */
      template<typename Derived>
      EulerAngles(const RotationBase<Derived, 3>& rot) { *this = rot; }
      
      /** Constructs and initialize Euler angles from a rotation \p rot,
        *  with options to choose for each angle the requested range.
        *
        * If possitive range is true, then the specified angle will be in the range [0, +2*PI].
        * Otherwise, the specified angle will be in the range [-PI, +PI].
        *
        * \param rot The 3x3 rotation matrix to convert
        * \param positiveRangeAlpha If true, alpha will be in [0, 2*PI]. Otherwise, in [-PI, +PI].
        * \param positiveRangeBeta If true, beta will be in [0, 2*PI]. Otherwise, in [-PI, +PI].
        * \param positiveRangeGamma If true, gamma will be in [0, 2*PI]. Otherwise, in [-PI, +PI].
      */
      template<typename Derived>
      EulerAngles(
        const RotationBase<Derived, 3>& rot,
        bool positiveRangeAlpha,
        bool positiveRangeBeta,
        bool positiveRangeGamma) {
        
        System::CalcEulerAngles(*this, rot.toRotationMatrix(), positiveRangeAlpha, positiveRangeBeta, positiveRangeGamma);
      }

      /** \returns The angle values stored in a vector (alpha, beta, gamma). */
      const Vector3& angles() const { return m_angles; }
      /** \returns A read-write reference to the angle values stored in a vector (alpha, beta, gamma). */
      Vector3& angles() { return m_angles; }

      /** \returns The value of the first angle. */
      Scalar alpha() const { return m_angles[0]; }
      /** \returns A read-write reference to the angle of the first angle. */
      Scalar& alpha() { return m_angles[0]; }

      /** \returns The value of the second angle. */
      Scalar beta() const { return m_angles[1]; }
      /** \returns A read-write reference to the angle of the second angle. */
      Scalar& beta() { return m_angles[1]; }

      /** \returns The value of the third angle. */
      Scalar gamma() const { return m_angles[2]; }
      /** \returns A read-write reference to the angle of the third angle. */
      Scalar& gamma() { return m_angles[2]; }

      /** \returns The Euler angles rotation inverse (which is as same as the negative),
        *  (-alpha, -beta, -gamma).
      */
      EulerAngles inverse() const
      {
        EulerAngles res;
        res.m_angles = -m_angles;
        return res;
      }

      /** \returns The Euler angles rotation negative (which is as same as the inverse),
        *  (-alpha, -beta, -gamma).
      */
      EulerAngles operator -() const
      {
        return inverse();
      }
      
      /** Constructs and initialize Euler angles from a 3x3 rotation matrix \p m,
        *  with options to choose for each angle the requested range (__only in compile time__).
        *
        * If possitive range is true, then the specified angle will be in the range [0, +2*PI].
        * Otherwise, the specified angle will be in the range [-PI, +PI].
        *
        * \param m The 3x3 rotation matrix to convert
        * \tparam positiveRangeAlpha If true, alpha will be in [0, 2*PI]. Otherwise, in [-PI, +PI].
        * \tparam positiveRangeBeta If true, beta will be in [0, 2*PI]. Otherwise, in [-PI, +PI].
        * \tparam positiveRangeGamma If true, gamma will be in [0, 2*PI]. Otherwise, in [-PI, +PI].
        */
      template<
        bool PositiveRangeAlpha,
        bool PositiveRangeBeta,
        bool PositiveRangeGamma,
        typename Derived>
      static EulerAngles FromRotation(const MatrixBase<Derived>& m)
      {
        EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 3)
        
        EulerAngles e;
        System::CalcEulerAngles<PositiveRangeAlpha, PositiveRangeBeta, PositiveRangeGamma>(e, m);
        return e;
      }
      
      /** Constructs and initialize Euler angles from a rotation \p rot,
        *  with options to choose for each angle the requested range (__only in compile time__).
        *
        * If possitive range is true, then the specified angle will be in the range [0, +2*PI].
        * Otherwise, the specified angle will be in the range [-PI, +PI].
        *
        * \param rot The 3x3 rotation matrix to convert
        * \tparam positiveRangeAlpha If true, alpha will be in [0, 2*PI]. Otherwise, in [-PI, +PI].
        * \tparam positiveRangeBeta If true, beta will be in [0, 2*PI]. Otherwise, in [-PI, +PI].
        * \tparam positiveRangeGamma If true, gamma will be in [0, 2*PI]. Otherwise, in [-PI, +PI].
      */
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
      
      /** Set \c *this from a rotation matrix(i.e. pure orthogonal matrix with determinant of +1). */
      template<typename Derived>
      EulerAngles& operator=(const MatrixBase<Derived>& m) {
        EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 3)
        
        System::CalcEulerAngles(*this, m);
        return *this;
      }

      // TODO: Assign and construct from another EulerAngles (with different system)
      
      /** Set \c *this from a rotation. */
      template<typename Derived>
      EulerAngles& operator=(const RotationBase<Derived, 3>& rot) {
        System::CalcEulerAngles(*this, rot.toRotationMatrix());
        return *this;
      }
      
      // TODO: Support isApprox function

      /** \returns an equivalent 3x3 rotation matrix. */
      Matrix3 toRotationMatrix() const
      {
        return static_cast<QuaternionType>(*this).toRotationMatrix();
      }

      /** Convert the Euler angles to quaternion. */
      operator QuaternionType() const
      {
        return
          AngleAxisType(alpha(), AlphaAxisVector()) *
          AngleAxisType(beta(), BetaAxisVector())   *
          AngleAxisType(gamma(), GammaAxisVector());
      }
  };

#define EIGEN_EULER_ANGLES_SINGLE_TYPEDEF(AXES, SCALAR_TYPE, SCALAR_POSTFIX) \
  typedef EulerAngles<SCALAR_TYPE, EulerSystem##AXES> EulerSystem##AXES##SCALAR_POSTFIX;

#define EIGEN_EULER_ANGLES_TYPEDEFS(SCALAR_TYPE, SCALAR_POSTFIX) \
  EIGEN_EULER_ANGLES_SINGLE_TYPEDEF(XYZ, SCALAR_TYPE, SCALAR_POSTFIX) \
  EIGEN_EULER_ANGLES_SINGLE_TYPEDEF(XYX, SCALAR_TYPE, SCALAR_POSTFIX) \
  EIGEN_EULER_ANGLES_SINGLE_TYPEDEF(XZY, SCALAR_TYPE, SCALAR_POSTFIX) \
  EIGEN_EULER_ANGLES_SINGLE_TYPEDEF(XZX, SCALAR_TYPE, SCALAR_POSTFIX) \
 \
  EIGEN_EULER_ANGLES_SINGLE_TYPEDEF(YZX, SCALAR_TYPE, SCALAR_POSTFIX) \
  EIGEN_EULER_ANGLES_SINGLE_TYPEDEF(YZY, SCALAR_TYPE, SCALAR_POSTFIX) \
  EIGEN_EULER_ANGLES_SINGLE_TYPEDEF(YXZ, SCALAR_TYPE, SCALAR_POSTFIX) \
  EIGEN_EULER_ANGLES_SINGLE_TYPEDEF(YXY, SCALAR_TYPE, SCALAR_POSTFIX) \
 \
  EIGEN_EULER_ANGLES_SINGLE_TYPEDEF(ZXY, SCALAR_TYPE, SCALAR_POSTFIX) \
  EIGEN_EULER_ANGLES_SINGLE_TYPEDEF(ZXZ, SCALAR_TYPE, SCALAR_POSTFIX) \
  EIGEN_EULER_ANGLES_SINGLE_TYPEDEF(ZYX, SCALAR_TYPE, SCALAR_POSTFIX) \
  EIGEN_EULER_ANGLES_SINGLE_TYPEDEF(ZYZ, SCALAR_TYPE, SCALAR_POSTFIX)

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
