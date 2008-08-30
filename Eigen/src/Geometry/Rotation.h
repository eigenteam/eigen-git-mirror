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

/** \class RotationBase
  *
  * \brief Common base class for compact rotation representations
  *
  * \param Derived is the derived type, i.e., a rotation type
  * \param _Dim the dimension of the space
  */
template<typename Derived, int _Dim>
class RotationBase
{
  public:
    enum { Dim = _Dim };
    /** the scalar type of the coefficients */
    typedef typename ei_traits<Derived>::Scalar Scalar;
    
    /** corresponding linear transformation matrix type */
    typedef Matrix<Scalar,Dim,Dim> RotationMatrixType;

    inline const Derived& derived() const { return *static_cast<const Derived*>(this); }
    inline Derived& derived() { return *static_cast<Derived*>(this); }

    /** \returns an equivalent rotation matrix */
    inline RotationMatrixType toRotationMatrix() const { return derived().toRotationMatrix(); }

    /** \returns the concatenation of the rotation \c *this with a translation \a t */
    inline Transform<Scalar,Dim> operator*(const Translation<Scalar,Dim>& t) const
    { return toRotationMatrix() * t; }

    /** \returns the concatenation of the rotation \c *this with a scaling \a s */
    inline RotationMatrixType operator*(const Scaling<Scalar,Dim>& s) const
    { return toRotationMatrix() * s; }

    /** \returns the concatenation of the rotation \c *this with an affine transformation \a t */
    inline Transform<Scalar,Dim> operator*(const Transform<Scalar,Dim>& t) const
    { return toRotationMatrix() * t; }
    
};

/** \geometry_module \ingroup GeometryModule
  *
  * \class Rotation2D
  *
  * \brief Represents a rotation/orientation in a 2 dimensional space.
  *
  * \param _Scalar the scalar type, i.e., the type of the coefficients
  *
  * This class is equivalent to a single scalar representing a counter clock wise rotation
  * as a single angle in radian. It provides some additional features such as the automatic
  * conversion from/to a 2x2 rotation matrix. Moreover this class aims to provide a similar
  * interface to Quaternion in order to facilitate the writing of generic algorithms
  * dealing with rotations.
  *
  * \sa class Quaternion, class Transform
  */
template<typename _Scalar> struct ei_traits<Rotation2D<_Scalar> >
{
  typedef _Scalar Scalar;
};

template<typename _Scalar>
class Rotation2D : public RotationBase<Rotation2D<_Scalar>,2>
{
  typedef RotationBase<Rotation2D<_Scalar>,2> Base;
  using Base::operator*;

public:
  enum { Dim = 2 };
  /** the scalar type of the coefficients */
  typedef _Scalar Scalar;
  typedef Matrix<Scalar,2,1> Vector2;
  typedef Matrix<Scalar,2,2> Matrix2;

protected:

  Scalar m_angle;

public:

  /** Construct a 2D counter clock wise rotation from the angle \a a in radian. */
  inline Rotation2D(Scalar a) : m_angle(a) {}

  /** \returns the rotation angle */
  inline Scalar angle() const { return m_angle; }

  /** \returns a read-write reference to the rotation angle */
  inline Scalar& angle() { return m_angle; }

  /** Automatic convertion to a 2D rotation matrix.
    * \sa toRotationMatrix()
    */
  inline operator Matrix2() const { return toRotationMatrix(); }

  /** \returns the inverse rotation */
  inline Rotation2D inverse() const { return -m_angle; }

  /** Concatenates two rotations */
  inline Rotation2D operator*(const Rotation2D& other) const
  { return m_angle + other.m_angle; }

  /** Concatenates two rotations */
  inline Rotation2D& operator*=(const Rotation2D& other)
  { return m_angle += other.m_angle; }

  /** Applies the rotation to a 2D vector */
  Vector2 operator* (const Vector2& vec) const
  { return toRotationMatrix() * vec; }

  template<typename Derived>
  Rotation2D& fromRotationMatrix(const MatrixBase<Derived>& m);
  Matrix2 toRotationMatrix(void) const;

  /** \returns the spherical interpolation between \c *this and \a other using
    * parameter \a t. It is in fact equivalent to a linear interpolation.
    */
  inline Rotation2D slerp(Scalar t, const Rotation2D& other) const
  { return m_angle * (1-t) + t * other; }
};

/** \ingroup GeometryModule
  * single precision 2D rotation type */
typedef Rotation2D<float> Rotation2Df;
/** \ingroup GeometryModule
  * double precision 2D rotation type */
typedef Rotation2D<double> Rotation2Dd;

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

/** \internal
  *
  * Helper function to return an arbitrary rotation object to a rotation matrix.
  *
  * \param Scalar the numeric type of the matrix coefficients
  * \param Dim the dimension of the current space
  *
  * It returns a Dim x Dim fixed size matrix.
  *
  * Default specializations are provided for:
  *   - any scalar type (2D),
  *   - any matrix expression,
  *   - any type based on RotationBase (e.g., Quaternion, AngleAxis, Rotation2D)
  *
  * Currently ei_toRotationMatrix is only used by Transform.
  *
  * \sa class Transform, class Rotation2D, class Quaternion, class AngleAxis
  */
template<typename Scalar, int Dim>
inline static Matrix<Scalar,2,2> ei_toRotationMatrix(const Scalar& s)
{
  EIGEN_STATIC_ASSERT(Dim==2,you_did_a_programming_error);
  return Rotation2D<Scalar>(s).toRotationMatrix();
}

template<typename Scalar, int Dim, typename OtherDerived>
inline static Matrix<Scalar,Dim,Dim> ei_toRotationMatrix(const RotationBase<OtherDerived,Dim>& r)
{
  return r.toRotationMatrix();
}

template<typename Scalar, int Dim, typename OtherDerived>
inline static const MatrixBase<OtherDerived>& ei_toRotationMatrix(const MatrixBase<OtherDerived>& mat)
{
  EIGEN_STATIC_ASSERT(OtherDerived::RowsAtCompileTime==Dim && OtherDerived::ColsAtCompileTime==Dim,
    you_did_a_programming_error);
  return mat;
}

#endif // EIGEN_ROTATION_H
