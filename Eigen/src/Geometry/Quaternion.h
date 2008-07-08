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

#ifndef EIGEN_QUATERNION_H
#define EIGEN_QUATERNION_H

template<typename Other,
         int OtherRows=Other::RowsAtCompileTime,
         int OtherCols=Other::ColsAtCompileTime>
struct ei_quaternion_assign_impl;

/** \class Quaternion
  *
  * \brief The quaternion class used to represent 3D orientations and rotations
  *
  * \param _Scalar the scalar type, i.e., the type of the coefficients
  *
  * This class represents a quaternion that is a convenient representation of
  * orientations and rotations of objects in three dimensions. Compared to other
  * representations like Euler angles or 3x3 matrices, quatertions offer the
  * following advantages:
  *   - compact storage (4 scalars)
  *   - efficient to compose (28 flops),
  *   - stable spherical interpolation
  *
  * \sa  class AngleAxis, class EulerAngles, class Transform
  */
template<typename _Scalar>
class Quaternion
{
  typedef Matrix<_Scalar, 4, 1> Coefficients;
  Coefficients m_coeffs;

public:

  /** the scalar type of the coefficients */
  typedef _Scalar Scalar;

  typedef Matrix<Scalar,3,1> Vector3;
  typedef Matrix<Scalar,3,3> Matrix3;
  typedef AngleAxis<Scalar> AngleAxisType;
  typedef EulerAngles<Scalar> EulerAnglesType;

  inline Scalar x() const { return m_coeffs.coeff(0); }
  inline Scalar y() const { return m_coeffs.coeff(1); }
  inline Scalar z() const { return m_coeffs.coeff(2); }
  inline Scalar w() const { return m_coeffs.coeff(3); }

  inline Scalar& x() { return m_coeffs.coeffRef(0); }
  inline Scalar& y() { return m_coeffs.coeffRef(1); }
  inline Scalar& z() { return m_coeffs.coeffRef(2); }
  inline Scalar& w() { return m_coeffs.coeffRef(3); }

  /** \returns a read-only vector expression of the imaginary part (x,y,z) */
  inline const Block<Coefficients,3,1> vec() const { return m_coeffs.template start<3>(); }

  /** \returns a vector expression of the imaginary part (x,y,z) */
  inline Block<Coefficients,3,1> vec() { return m_coeffs.template start<3>(); }

  /** \returns a read-only vector expression of the coefficients */
  inline const Coefficients& coeffs() const { return m_coeffs; }

  /** \returns a vector expression of the coefficients */
  inline Coefficients& coeffs() { return m_coeffs; }

  // FIXME what is the prefered order: w x,y,z or x,y,z,w ?
  inline Quaternion(Scalar w = 1.0, Scalar x = 0.0, Scalar y = 0.0, Scalar z = 0.0)
  {
    m_coeffs.coeffRef(0) = x;
    m_coeffs.coeffRef(1) = y;
    m_coeffs.coeffRef(2) = z;
    m_coeffs.coeffRef(3) = w;
  }

  /** Copy constructor */
  inline Quaternion(const Quaternion& other) { m_coeffs = other.m_coeffs; }

  explicit inline Quaternion(const AngleAxisType& aa) { *this = aa; }
  explicit inline Quaternion(const EulerAnglesType& ea) { *this = ea; }
  template<typename Derived>
  explicit inline Quaternion(const MatrixBase<Derived>& other) { *this = other; }

  Quaternion& operator=(const Quaternion& other);
  Quaternion& operator=(const AngleAxisType& aa);
  Quaternion& operator=(EulerAnglesType ea);
  template<typename Derived>
  Quaternion& operator=(const MatrixBase<Derived>& m);

  /** \returns a quaternion representing an identity rotation
    * \sa MatrixBase::identity()
    */
  inline static Quaternion identity() { return Quaternion(1, 0, 0, 0); }

  /** \sa Quaternion::identity(), MatrixBase::setIdentity()
    */
  inline Quaternion& setIdentity() { m_coeffs << 1, 0, 0, 0; return *this; }

  /** \returns the squared norm of the quaternion's coefficients
    * \sa Quaternion::norm(), MatrixBase::norm2()
    */
  inline Scalar norm2() const { return m_coeffs.norm2(); }

  /** \returns the norm of the quaternion's coefficients
    * \sa Quaternion::norm2(), MatrixBase::norm()
    */
  inline Scalar norm() const { return m_coeffs.norm(); }

  Matrix3 toRotationMatrix(void) const;

  template<typename Derived1, typename Derived2>
  Quaternion& setFromTwoVectors(const MatrixBase<Derived1>& a, const MatrixBase<Derived2>& b);

  inline Quaternion operator* (const Quaternion& q) const;
  inline Quaternion& operator*= (const Quaternion& q);

  Quaternion inverse(void) const;
  Quaternion conjugate(void) const;

  Quaternion slerp(Scalar t, const Quaternion& other) const;

  template<typename Derived>
  Vector3 operator* (const MatrixBase<Derived>& vec) const;

};

/** \returns the concatenation of two rotations as a quaternion-quaternion product */
template <typename Scalar>
inline Quaternion<Scalar> Quaternion<Scalar>::operator* (const Quaternion& other) const
{
  return Quaternion
  (
    this->w() * other.w() - this->x() * other.x() - this->y() * other.y() - this->z() * other.z(),
    this->w() * other.x() + this->x() * other.w() + this->y() * other.z() - this->z() * other.y(),
    this->w() * other.y() + this->y() * other.w() + this->z() * other.x() - this->x() * other.z(),
    this->w() * other.z() + this->z() * other.w() + this->x() * other.y() - this->y() * other.x()
  );
}

template <typename Scalar>
inline Quaternion<Scalar>& Quaternion<Scalar>::operator*= (const Quaternion& other)
{
  return (*this = *this * other);
}

/** Rotation of a vector by a quaternion.
  * \remarks If the quaternion is used to rotate several points (>1)
  * then it is much more efficient to first convert it to a 3x3 Matrix.
  * Comparison of the operation cost for n transformations:
  *   - Quaternion:    30n
  *   - Via a Matrix3: 24 + 15n
  */
template <typename Scalar>
template<typename Derived>
inline typename Quaternion<Scalar>::Vector3
Quaternion<Scalar>::operator* (const MatrixBase<Derived>& v) const
{
    // Note that this algorithm comes from the optimization by hand
    // of the conversion to a Matrix followed by a Matrix/Vector product.
    // It appears to be much faster than the common algorithm found
    // in the litterature (30 versus 39 flops). It also requires two
    // Vector3 as temporaries.
    Vector3 uv;
    uv = 2 * this->vec().cross(v);
    return v + this->w() * uv + this->vec().cross(uv);
}

template<typename Scalar>
inline Quaternion<Scalar>& Quaternion<Scalar>::operator=(const Quaternion& other)
{
  m_coeffs = other.m_coeffs;
  return *this;
}

/** Set \c *this from an angle-axis \a aa
  * and returns a reference to \c *this
  */
template<typename Scalar>
inline Quaternion<Scalar>& Quaternion<Scalar>::operator=(const AngleAxisType& aa)
{
  Scalar ha = 0.5*aa.angle();
  this->w() = ei_cos(ha);
  this->vec() = ei_sin(ha) * aa.axis();
  return *this;
}

/** Set \c *this from the rotation defined by the Euler angles \a ea,
  * and returns a reference to \c *this
  */
template<typename Scalar>
inline Quaternion<Scalar>& Quaternion<Scalar>::operator=(EulerAnglesType ea)
{
  ea.coeffs() *= 0.5;

  Vector3 cosines = ea.coeffs().cwise().cos();
  Vector3 sines   = ea.coeffs().cwise().sin();

  Scalar cYcZ = cosines.y() * cosines.z();
  Scalar sYsZ = sines.y() * sines.z();
  Scalar sYcZ = sines.y() * cosines.z();
  Scalar cYsZ = cosines.y() * sines.z();

  this->w() = cosines.x() * cYcZ + sines.x()   * sYsZ;
  this->x() = sines.x()   * cYcZ - cosines.x() * sYsZ;
  this->y() = cosines.x() * sYcZ + sines.x()   * cYsZ;
  this->z() = cosines.x() * cYsZ - sines.x()   * sYcZ;

  return *this;
}

/** Set \c *this from the expression \a xpr:
  *   - if \a xpr is a 4x1 vector, then \a xpr is assumed to be a quaternion
  *   - if \a xpr is a 3x3 matrix, then \a xpr is assumed to be rotation matrix
  *     and \a xpr is converted to a quaternion
  */
template<typename Scalar>
template<typename Derived>
inline Quaternion<Scalar>& Quaternion<Scalar>::operator=(const MatrixBase<Derived>& xpr)
{
  ei_quaternion_assign_impl<Derived>::run(*this, xpr.derived());
  return *this;
}

/** Convert the quaternion to a 3x3 rotation matrix */
template<typename Scalar>
inline typename Quaternion<Scalar>::Matrix3
Quaternion<Scalar>::toRotationMatrix(void) const
{
  // NOTE if inlined, then gcc 4.2 and 4.4 get rid of the temporary (not gcc 4.3 !!)
  // if not inlined then the cost of the return by value is huge ~ +35%,
  // however, not inlining this function is an order of magnitude slower, so
  // it has to be inlined, and so the return by value is not an issue
  Matrix3 res;

  Scalar tx  = 2*this->x();
  Scalar ty  = 2*this->y();
  Scalar tz  = 2*this->z();
  Scalar twx = tx*this->w();
  Scalar twy = ty*this->w();
  Scalar twz = tz*this->w();
  Scalar txx = tx*this->x();
  Scalar txy = ty*this->x();
  Scalar txz = tz*this->x();
  Scalar tyy = ty*this->y();
  Scalar tyz = tz*this->y();
  Scalar tzz = tz*this->z();

  res.coeffRef(0,0) = 1-(tyy+tzz);
  res.coeffRef(0,1) = txy-twz;
  res.coeffRef(0,2) = txz+twy;
  res.coeffRef(1,0) = txy+twz;
  res.coeffRef(1,1) = 1-(txx+tzz);
  res.coeffRef(1,2) = tyz-twx;
  res.coeffRef(2,0) = txz-twy;
  res.coeffRef(2,1) = tyz+twx;
  res.coeffRef(2,2) = 1-(txx+tyy);

  return res;
}

/** Makes a quaternion representing the rotation between two vectors \a a and \a b.
  * \returns a reference to the actual quaternion
  * Note that the two input vectors have \b not to be normalized.
  */
template<typename Scalar>
template<typename Derived1, typename Derived2>
inline Quaternion<Scalar>& Quaternion<Scalar>::setFromTwoVectors(const MatrixBase<Derived1>& a, const MatrixBase<Derived2>& b)
{
  Vector3 v0 = a.normalized();
  Vector3 v1 = b.normalized();
  Vector3 axis = v0.cross(v1);
  Scalar c = v0.dot(v1);

  // if dot == 1, vectors are the same
  if (ei_isApprox(c,Scalar(1)))
  {
    // set to identity
    this->w() = 1; this->vec().setZero();
  }
  Scalar s = ei_sqrt((1+c)*2);
  Scalar invs = 1./s;
  this->vec() = axis * invs;
  this->w() = s * 0.5;

  return *this;
}

/** \returns the multiplicative inverse of \c *this
  * Note that in most cases, i.e., if you simply want the opposite
  * rotation, it is enough to use the conjugate.
  *
  * \sa Quaternion::conjugate()
  */
template <typename Scalar>
inline Quaternion<Scalar> Quaternion<Scalar>::inverse() const
{
  // FIXME should this funtion be called multiplicativeInverse and conjugate() be called inverse() or opposite()  ??
  Scalar n2 = this->norm2();
  if (n2 > 0)
    return Quaternion(conjugate().coeffs() / n2);
  else
  {
    // return an invalid result to flag the error
    return Quaternion(Coefficients::zero());
  }
}

/** \returns the conjugate of the \c *this which is equal to the multiplicative inverse
  * if the quaternion is normalized.
  * The conjugate of a quaternion represents the opposite rotation.
  *
  * \sa Quaternion::inverse()
  */
template <typename Scalar>
inline Quaternion<Scalar> Quaternion<Scalar>::conjugate() const
{
  return Quaternion(this->w(),-this->x(),-this->y(),-this->z());
}

/** \returns the spherical linear interpolation between the two quaternions
  * \c *this and \a other at the parameter \a t
  */
template <typename Scalar>
Quaternion<Scalar> Quaternion<Scalar>::slerp(Scalar t, const Quaternion& other) const
{
  // FIXME options for this function would be:
  // 1 - Quaternion& fromSlerp(Scalar t, const Quaternion& q0, const Quaternion& q1);
  //     which set *this from the s-lerp and returns *this
  // 2 - Quaternion slerp(Scalar t, const Quaternion& other) const
  //     which returns the s-lerp between this and other
  // ??
  if (*this == other)
    return *this;

  Scalar d = this->dot(other);

  // theta is the angle between the 2 quaternions
  Scalar theta = std::acos(ei_abs(d));
  Scalar sinTheta = ei_sin(theta);

  Scalar scale0 = ei_sin( ( 1 - t ) * theta) / sinTheta;
  Scalar scale1 = ei_sin( ( t * theta) ) / sinTheta;
  if (d<0)
    scale1 = -scale1;

  return scale0 * (*this) + scale1 * other;
}

// set from a rotation matrix
template<typename Other>
struct ei_quaternion_assign_impl<Other,3,3>
{
  typedef typename Other::Scalar Scalar;
  inline static void run(Quaternion<Scalar>& q, const Other& mat)
  {
    // This algorithm comes from  "Quaternion Calculus and Fast Animation",
    // Ken Shoemake, 1987 SIGGRAPH course notes
    Scalar t = mat.trace();
    if (t > 0)
    {
      t = ei_sqrt(t + 1.0);
      q.w() = 0.5*t;
      t = 0.5/t;
      q.x() = (mat.coeff(2,1) - mat.coeff(1,2)) * t;
      q.y() = (mat.coeff(0,2) - mat.coeff(2,0)) * t;
      q.z() = (mat.coeff(1,0) - mat.coeff(0,1)) * t;
    }
    else
    {
      int i = 0;
      if (mat.coeff(1,1) > mat.coeff(0,0))
        i = 1;
      if (mat.coeff(2,2) > mat.coeff(i,i))
        i = 2;
      int j = (i+1)%3;
      int k = (j+1)%3;

      t = ei_sqrt(mat.coeff(i,i)-mat.coeff(j,j)-mat.coeff(k,k) + 1.0);
      q.coeffs().coeffRef(i) = 0.5 * t;
      t = 0.5/t;
      q.w() = (mat.coeff(k,j)-mat.coeff(j,k))*t;
      q.coeffs().coeffRef(j) = (mat.coeff(j,i)+mat.coeff(i,j))*t;
      q.coeffs().coeffRef(k) = (mat.coeff(k,i)+mat.coeff(i,k))*t;
    }
  }
};

// set from a vector of coefficients assumed to be a quaternion
template<typename Other>
struct ei_quaternion_assign_impl<Other,4,1>
{
  typedef typename Other::Scalar Scalar;
  inline static void run(Quaternion<Scalar>& q, const Other& vec)
  {
    q.coeffs() = vec;
  }
};

#endif // EIGEN_QUATERNION_H
