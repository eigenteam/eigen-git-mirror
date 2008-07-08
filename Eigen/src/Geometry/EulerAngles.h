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

#ifndef EIGEN_EULERANGLES_H
#define EIGEN_EULERANGLES_H

template<typename Other,
         int OtherRows=Other::RowsAtCompileTime,
         int OtherCols=Other::ColsAtCompileTime>
struct ei_eulerangles_assign_impl;

/** \class EulerAngles
  *
  * \brief Represents a rotation in a 3 dimensional space as three Euler angles
  *
  * \param _Scalar the scalar type, i.e., the type of the angles.
  *
  * \sa class Quaternion, class AngleAxis, class Transform
  */
template<typename _Scalar>
class EulerAngles
{
public:
  enum { Dim = 3 };
  /** the scalar type of the coefficients */
  typedef _Scalar Scalar;
  typedef Matrix<Scalar,3,3> Matrix3;
  typedef Matrix<Scalar,3,1> Vector3;
  typedef Quaternion<Scalar> QuaternionType;
  typedef AngleAxis<Scalar> AngleAxisType;

protected:

  Vector3 m_angles;

public:

  EulerAngles() {}
  template<typename Derived>
  inline EulerAngles(Scalar a0, Scalar a1, Scalar a2) : m_angles(a0, a1, a2) {}
  inline EulerAngles(const QuaternionType& q) { *this = q; }
  inline EulerAngles(const AngleAxisType& aa) { *this = aa; }
  template<typename Derived>
  inline EulerAngles(const MatrixBase<Derived>& m) { *this = m; }

  Scalar angle(int i) const { return m_angles.coeff(i); }
  Scalar& angle(int i) { return m_angles.coeffRef(i); }

  const Vector3& coeffs() const { return m_angles; }
  Vector3& coeffs() { return m_angles; }

  EulerAngles& operator=(const QuaternionType& q);
  EulerAngles& operator=(const AngleAxisType& ea);
  template<typename Derived>
  EulerAngles& operator=(const MatrixBase<Derived>& m);

  template<typename Derived>
  EulerAngles& fromRotationMatrix(const MatrixBase<Derived>& m);
  Matrix3 toRotationMatrix(void) const;
};

/** Set \c *this from a quaternion.
  * The axis is normalized.
  */
template<typename Scalar>
EulerAngles<Scalar>& EulerAngles<Scalar>::operator=(const QuaternionType& q)
{
  Scalar y2 = q.y() * q.y();
  m_angles.coeffRef(0) = std::atan2(2*(q.w()*q.x() + q.y()*q.z()), (1 - 2*(q.x()*q.x() + y2)));
  m_angles.coeffRef(1) = std::asin( 2*(q.w()*q.y() - q.z()*q.x()));
  m_angles.coeffRef(2) = std::atan2(2*(q.w()*q.z() + q.x()*q.y()), (1 - 2*(y2 + q.z()*q.z())));
  return *this;
}

/** Set \c *this from Euler angles \a ea.
  */
template<typename Scalar>
EulerAngles<Scalar>& EulerAngles<Scalar>::operator=(const AngleAxisType& aa)
{
  return *this = QuaternionType(aa);
}

/** Set \c *this from the expression \a xpr:
  *   - if \a xpr is a 3x1 vector, then \a xpr is assumed to be a vector of angles
  *   - if \a xpr is a 3x3 matrix, then \a xpr is assumed to be rotation matrix
  *     and \a xpr is converted to Euler angles
  */
template<typename Scalar>
template<typename Derived>
EulerAngles<Scalar>& EulerAngles<Scalar>::operator=(const MatrixBase<Derived>& other)
{
  ei_eulerangles_assign_impl<Derived>::run(*this,other.derived());
  return *this;
}

/** Constructs and \returns an equivalent 3x3 rotation matrix.
  */
template<typename Scalar>
typename EulerAngles<Scalar>::Matrix3
EulerAngles<Scalar>::toRotationMatrix(void) const
{
  Vector3 c = m_angles.cwise().cos();
  Vector3 s = m_angles.cwise().sin();
  return Matrix3() <<
    c.y()*c.z(),                    -c.y()*s.z(),                   s.y(),
    c.z()*s.x()*s.y()+c.x()*s.z(),  c.x()*c.z()-s.x()*s.y()*s.z(),  -c.y()*s.x(),
    -c.x()*c.z()*s.y()+s.x()*s.z(), c.z()*s.x()+c.x()*s.y()*s.z(),  c.x()*c.y();
}

// set from a rotation matrix
template<typename Other>
struct ei_eulerangles_assign_impl<Other,3,3>
{
  typedef typename Other::Scalar Scalar;
  inline static void run(EulerAngles<Scalar>& ea, const Other& mat)
  {
    // mat =  cy*cz          -cy*sz           sy
    //        cz*sx*sy+cx*sz  cx*cz-sx*sy*sz -cy*sx
    //       -cx*cz*sy+sx*sz  cz*sx+cx*sy*sz  cx*cy
    ea.angle(1) = std::asin(mat.coeff(0,2));
    ea.angle(0) = std::atan2(-mat.coeff(1,2),mat.coeff(2,2));
    ea.angle(2) = std::atan2(-mat.coeff(0,1),mat.coeff(0,0));
  }
};

// set from a vector of angles
template<typename Other>
struct ei_eulerangles_assign_impl<Other,3,1>
{
  typedef typename Other::Scalar Scalar;
  inline static void run(EulerAngles<Scalar>& ea, const Other& vec)
  {
    ea.coeffs() = vec;
  }
};

#endif // EIGEN_EULERANGLES_H
