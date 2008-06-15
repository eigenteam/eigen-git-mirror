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

#ifndef EIGEN_ROTATION_H
#define EIGEN_ROTATION_H

// this file aims to contains the various representations of rotation/orientation
// in 2D and 3D space excepted Matrix and Quaternion.

/** \class ToRotationMatrix
  *
  * \brief Template static struct to convert any rotation representation to a matrix form
  *
  * \param Scalar the numeric type of the matrix coefficients
  * \param Dim the dimension of the current space
  * \param RotationType the input type of the rotation
  *
  * This class defines a single static member with the following prototype:
  * \code
  * static <MatrixExpression> convert(const RotationType& r);
  * \endcode
  * where \c <MatrixExpression> must be a fixed-size matrix expression of size Dim x Dim and
  * coefficient type Scalar.
  *
  * Default specializations are provided for:
  *   - any scalar type (2D),
  *   - any matrix expression,
  *   - Quaternion,
  *   - AngleAxis.
  *
  * Currently ToRotationMatrix is only used by Transform.
  *
  * \sa class Transform, class Rotation2D, class Quaternion, class AngleAxis
  *
  */
template<typename Scalar, int Dim, typename RotationType>
struct ToRotationMatrix;

// 2D rotation to matrix
template<typename Scalar, typename OtherScalarType>
struct ToRotationMatrix<Scalar, 2, OtherScalarType>
{
  inline static Matrix<Scalar,2,2> convert(const OtherScalarType& r)
  { return Rotation2D<Scalar>(r).toRotationMatrix(); }
};

// 2D rotation to matrix
template<typename Scalar, typename OtherScalarType>
struct ToRotationMatrix<Scalar, 2, Rotation2D<OtherScalarType> >
{
  inline static Matrix<Scalar,2,2> convert(const Rotation2D<OtherScalarType>& r)
  { return Rotation2D<Scalar>(r).toRotationMatrix(); }
};

// quaternion to matrix
template<typename Scalar, typename OtherScalarType>
struct ToRotationMatrix<Scalar, 3, Quaternion<OtherScalarType> >
{
  inline static Matrix<Scalar,3,3> convert(const Quaternion<OtherScalarType>& q)
  { return q.toRotationMatrix(); }
};

// angle axis to matrix
template<typename Scalar, typename OtherScalarType>
struct ToRotationMatrix<Scalar, 3, AngleAxis<OtherScalarType> >
{
  inline static Matrix<Scalar,3,3> convert(const AngleAxis<OtherScalarType>& aa)
  { return aa.toRotationMatrix(); }
};

// matrix xpr to matrix xpr
template<typename Scalar, int Dim, typename OtherDerived>
struct ToRotationMatrix<Scalar, Dim, MatrixBase<OtherDerived> >
{
  inline static const MatrixBase<OtherDerived>& convert(const MatrixBase<OtherDerived>& mat)
  {
    EIGEN_STATIC_ASSERT(OtherDerived::RowsAtCompileTime==Dim && OtherDerived::ColsAtCompileTime==Dim,
      you_did_a_programming_error);
    return mat;
  }
};

/** \class Rotation2D
  *
  * \brief Represents a rotation/orientation in a 2 dimensional space.
  *
  * \param _Scalar the scalar type, i.e., the type of the coefficients
  *
  * This class is equivalent to a single scalar representating the rotation angle
  * in radian with some additional features such as the conversion from/to
  * rotation matrix. Moreover this class aims to provide a similar interface
  * to Quaternion in order to facilitate the writting of generic algorithm
  * dealing with rotations.
  *
  * \sa class Quaternion, class Transform
  */
template<typename _Scalar>
class Rotation2D
{
public:
  enum { Dim = 2 };
  /** the scalar type of the coefficients */
  typedef _Scalar Scalar;
  typedef Matrix<Scalar,2,2> Matrix2;

protected:

  Scalar m_angle;

public:

  inline Rotation2D(Scalar a) : m_angle(a) {}
  inline operator Scalar& () { return m_angle; }
  inline operator Scalar () const { return m_angle; }

  template<typename Derived>
  Rotation2D& fromRotationMatrix(const MatrixBase<Derived>& m);
  Matrix2 toRotationMatrix(void) const;

  /** \returns the spherical interpolation between \c *this and \a other using
    * parameter \a t. It is equivalent to a linear interpolation.
    */
  inline Rotation2D slerp(Scalar t, const Rotation2D& other) const
  { return m_angle * (1-t) + t * other; }
};

/** Set \c *this from a 2x2 rotation matrix \a mat.
  * In other words, this function extract the rotation angle
  * from the rotation matrix.
  */
template<typename Scalar>
template<typename Derived>
Rotation2D<Scalar>& Rotation2D<Scalar>::fromRotationMatrix(const MatrixBase<Derived>& mat)
{
  EIGEN_STATIC_ASSERT(Derived::RowsAtCompileTime==2 && Derived::ColsAtCompileTime==2,you_did_a_programming_error);
  m_angle = ei_atan2(mat.coeff(1,0), mat.coeff(0,0));
  return *this;
}

/** Constructs and \returns an equivalent 2x2 rotation matrix.
  */
template<typename Scalar>
typename Rotation2D<Scalar>::Matrix2
Rotation2D<Scalar>::toRotationMatrix(void) const
{
  Scalar sinA = ei_sin(m_angle);
  Scalar cosA = ei_cos(m_angle);
  return (Matrix2() << cosA, -sinA, sinA, cosA).finished();
}

/** \class AngleAxis
  *
  * \brief Represents a rotation in a 3 dimensional space as a rotation angle around a 3D axis
  *
  * \param _Scalar the scalar type, i.e., the type of the coefficients.
  *
  * \sa class Quaternion, class Transform
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

protected:

  Vector3 m_axis;
  Scalar m_angle;

public:

  AngleAxis() {}
  template<typename Derived>
  inline AngleAxis(Scalar angle, const MatrixBase<Derived>& axis) : m_axis(axis), m_angle(angle) {}

  Scalar angle() const { return m_angle; }
  Scalar& angle() { return m_angle; }

  const Vector3& axis() const { return m_axis; }
  Vector3& axis() { return m_axis; }

  template<typename Derived>
  AngleAxis& fromRotationMatrix(const MatrixBase<Derived>& m);
  Matrix3 toRotationMatrix(void) const;
};

/** Set \c *this from a 3x3 rotation matrix \a mat.
  * In other words, this function extract the rotation angle
  * from the rotation matrix.
  */
template<typename Scalar>
template<typename Derived>
AngleAxis<Scalar>& AngleAxis<Scalar>::fromRotationMatrix(const MatrixBase<Derived>& mat)
{
  // Since a direct conversion would not be really faster,
  // let's use the robust Quaternion implementation:
  Quaternion<Scalar>().fromRotationMatrix(mat).toAngleAxis(m_angle, m_axis);

  return *this;
}

/** Constructs and \returns an equivalent 2x2 rotation matrix.
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

#endif // EIGEN_ROTATION_H
