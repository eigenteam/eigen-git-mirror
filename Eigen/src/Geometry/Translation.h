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

#ifndef EIGEN_TRANSLATION_H
#define EIGEN_TRANSLATION_H

/** \geometry_module \ingroup GeometryModule
  *
  * \class Translation
  *
  * \brief Represents a translation transformation
  *
  * \param _Scalar the scalar type, i.e., the type of the coefficients.
  * \param _Dim the  dimension of the space, can be a compile time value or Dynamic
  *
  *
  * \sa class Scaling, class Transform
  */
template<typename _Scalar, int _Dim>
class Translation
{
public:
  enum { Dim = _Dim };
  /** the scalar type of the coefficients */
  typedef _Scalar Scalar;
  typedef Matrix<Scalar,Dim,Dim> LinearMatrixType;
  typedef Matrix<Scalar,Dim,1> VectorType;
  typedef Scaling<Scalar,Dim> ScalingType;
  typedef Transform<Scalar,Dim> TransformType;

protected:

  VectorType m_coeffs;

public:

  /** Default constructor without initialization. */
  Translation() {}
  /**  */
  inline Translation(const Scalar& sx, const Scalar& sy)
  {
    ei_assert(Dim==2);
    m_coeffs.x() = sx;
    m_coeffs.y() = sy;
  }
  /**  */
  inline Translation(const Scalar& sx, const Scalar& sy, const Scalar& sz)
  {
    ei_assert(Dim==3);
    m_coeffs.x() = sx;
    m_coeffs.y() = sy;
    m_coeffs.z() = sz;
  }
  /** Constructs and initialize the scaling transformation from a vector of scaling coefficients */
  explicit inline Translation(const VectorType& vector) : m_coeffs(vector) {}

  const VectorType& vector() const { return m_coeffs; }
  VectorType& vector() { return m_coeffs; }

  /** Concatenates two translation */
  inline Translation operator* (const Translation& other) const
  { return Translation(m_coeffs + other.m_coeffs); }
  
  /** Concatenates a translation and a scaling */
  inline TransformType operator* (const ScalingType& other) const;

  /** Concatenates a translation and a linear transformation */
  inline TransformType operator* (const LinearMatrixType& linear) const;

  /** Concatenates a linear transformation and a translation */
  // its a nightmare to define a templated friend function outside its declaration
  friend inline TransformType operator* (const LinearMatrixType& linear, const Translation& t)
  {
    TransformType res;
    res.matrix().setZero();
    res.linear() = linear;
    res.translation() = linear * t.m_coeffs;
    res.matrix().row(Dim).setZero();
    res(Dim,Dim) = Scalar(1);
    return res;
  }

  /** Concatenates a translation and an affine transformation */
  inline TransformType operator* (const TransformType& t) const;

  /** Applies translation to vector */
  inline VectorType operator* (const VectorType& other) const
  { return m_coeffs + other; }

  /** \returns the inverse translation (opposite) */
  Translation inverse() const { return Translation(-m_coeffs); }

  Translation& operator=(const Translation& other)
  {
    m_coeffs = other.m_coeffs;
    return *this;
  }

};

/** \addtogroup GeometryModule */
//@{
typedef Translation<float, 2> Translation2f;
typedef Translation<double,2> Translation2d;
typedef Translation<float, 3> Translation3f;
typedef Translation<double,3> Translation3d;
//@}


template<typename Scalar, int Dim>
inline typename Translation<Scalar,Dim>::TransformType
Translation<Scalar,Dim>::operator* (const ScalingType& other) const
{
  TransformType res;
  res.matrix().setZero();
  res.linear().diagonal() = other.coeffs();
  res.translation() = m_coeffs;
  res(Dim,Dim) = Scalar(1);
  return res;
}

template<typename Scalar, int Dim>
inline typename Translation<Scalar,Dim>::TransformType
Translation<Scalar,Dim>::operator* (const LinearMatrixType& linear) const
{
  TransformType res;
  res.matrix().setZero();
  res.linear() = linear;
  res.translation() = m_coeffs;
  res.matrix().row(Dim).setZero();
  res(Dim,Dim) = Scalar(1);
  return res;
}

template<typename Scalar, int Dim>
inline typename Translation<Scalar,Dim>::TransformType
Translation<Scalar,Dim>::operator* (const TransformType& t) const
{
  TransformType res = t;
  res.pretranslate(m_coeffs);
  return res;
}

#endif // EIGEN_TRANSLATION_H
