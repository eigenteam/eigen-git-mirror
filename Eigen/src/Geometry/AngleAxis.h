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

#ifndef EIGEN_ANGLEAXIS_H
#define EIGEN_ANGLEAXIS_H

/** \class AngleAxis
  *
  * \brief Represents a rotation in a 3 dimensional space as a rotation angle around a 3D axis
  *
  * \param _Scalar the scalar type, i.e., the type of the coefficients.
  *
  * \sa class Quaternion, class EulerAngles, class Transform
  */
template<typename _Scalar>
class AngleAxis
{
public:
  enum { Dim = 3 };
  /** the scalar type of the coefficients */
  typedef _Scalar Scalar;
  typedef Matrix<Scalar,3,3> Matrix3;
  typedef Matrix<Scalar,3,1> Vector3;
  typedef Quaternion<Scalar> QuaternionType;
  typedef EulerAngles<Scalar> EulerAnglesType;

protected:

  Vector3 m_axis;
  Scalar m_angle;

public:

  AngleAxis() {}
  template<typename Derived>
  inline AngleAxis(Scalar angle, const MatrixBase<Derived>& axis) : m_axis(axis), m_angle(angle) {}
  inline AngleAxis(const QuaternionType& q) { *this = q; }
  inline AngleAxis(const EulerAnglesType& ea) { *this = ea; }
  template<typename Derived>
  inline AngleAxis(const MatrixBase<Derived>& m) { *this = m; }

  Scalar angle() const { return m_angle; }
  Scalar& angle() { return m_angle; }

  const Vector3& axis() const { return m_axis; }
  Vector3& axis() { return m_axis; }

  AngleAxis& operator=(const QuaternionType& q);
  AngleAxis& operator=(const EulerAnglesType& ea);
  template<typename Derived>
  AngleAxis& operator=(const MatrixBase<Derived>& m);

  template<typename Derived>
  AngleAxis& fromRotationMatrix(const MatrixBase<Derived>& m);
  Matrix3 toRotationMatrix(void) const;
};

/** Set \c *this from a quaternion.
  * The axis is normalized.
  */
template<typename Scalar>
AngleAxis<Scalar>& AngleAxis<Scalar>::operator=(const QuaternionType& q)
{
  Scalar n2 = q.vec().norm2();
  if (ei_isMuchSmallerThan(n2,Scalar(1)))
  {
    m_angle = 0;
    m_axis << 1, 0, 0;
  }
  else
  {
    m_angle = 2*std::acos(q.w());
    m_axis = q.vec() / ei_sqrt(n2);
  }
  return *this;
}

/** Set \c *this from Euler angles \a ea.
  */
template<typename Scalar>
AngleAxis<Scalar>& AngleAxis<Scalar>::operator=(const EulerAnglesType& ea)
{
  return *this = QuaternionType(ea);
}

/** Set \c *this from a 3x3 rotation matrix \a mat.
  */
template<typename Scalar>
template<typename Derived>
AngleAxis<Scalar>& AngleAxis<Scalar>::operator=(const MatrixBase<Derived>& mat)
{
  // Since a direct conversion would not be really faster,
  // let's use the robust Quaternion implementation:
  return *this = QuaternionType(mat);
}

/** Constructs and \returns an equivalent 3x3 rotation matrix.
  */
template<typename Scalar>
typename AngleAxis<Scalar>::Matrix3
AngleAxis<Scalar>::toRotationMatrix(void) const
{
  Matrix3 res;
  Vector3 sin_axis  = ei_sin(m_angle) * m_axis;
  Scalar c = ei_cos(m_angle);
  Vector3 cos1_axis = (Scalar(1)-c) * m_axis;

  Scalar tmp;
  tmp = cos1_axis.x() * m_axis.y();
  res.coeffRef(0,1) = tmp - sin_axis.z();
  res.coeffRef(1,0) = tmp + sin_axis.z();

  tmp = cos1_axis.x() * m_axis.z();
  res.coeffRef(0,2) = tmp + sin_axis.y();
  res.coeffRef(2,0) = tmp - sin_axis.y();

  tmp = cos1_axis.y() * m_axis.z();
  res.coeffRef(1,2) = tmp - sin_axis.x();
  res.coeffRef(2,1) = tmp + sin_axis.x();

  res.diagonal() = Vector3::constant(c) + cos1_axis.cwiseProduct(m_axis);

  return res;
}

#endif // EIGEN_ANGLEAXIS_H
