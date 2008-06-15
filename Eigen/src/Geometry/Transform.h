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

#ifndef EIGEN_TRANSFORM_H
#define EIGEN_TRANSFORM_H

/** \class Orientation2D
  *
  * \brief Represents an orientation/rotation in a 2 dimensional space.
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
class Orientation2D
{
public:
  enum { Dim = 2 };
  /** the scalar type of the coefficients */
  typedef _Scalar Scalar;
  typedef Matrix<Scalar,2,2> Matrix2;

protected:

  Scalar m_angle;

public:

  inline Orientation2D(Scalar a) : m_angle(a) {}
  inline operator Scalar& () { return m_angle; }
  inline operator Scalar () const { return m_angle; }

  template<typename Derived>
  Orientation2D& fromRotationMatrix(const MatrixBase<Derived>& m);
  Matrix2 toRotationMatrix(void) const;

  Orientation2D slerp(Scalar t, const Orientation2D& other) const;
};

/** returns the default type used to represent an orientation.
  */
template<typename Scalar, int Dim>
struct ei_get_orientation_type;

template<typename Scalar>
struct ei_get_orientation_type<Scalar,2>
{ typedef Orientation2D<Scalar> type; };

template<typename Scalar>
struct ei_get_orientation_type<Scalar,3>
{ typedef Quaternion<Scalar> type; };

/** Set \c *this from a 2x2 rotation matrix \a mat.
  * In other words, this function extract the rotation angle
  * from the rotation matrix.
  */
template<typename Scalar>
template<typename Derived>
Orientation2D<Scalar>& Orientation2D<Scalar>::fromRotationMatrix(const MatrixBase<Derived>& mat)
{
  EIGEN_STATIC_ASSERT(Derived::RowsAtCompileTime==2 && Derived::ColsAtCompileTime==2,you_did_a_programming_error);
  m_angle = ei_atan2(mat.coeff(1,0), mat.coeff(0,0));
  return *this;
}

/** Constructs and \returns an equivalent 2x2 rotation matrix.
  */
template<typename Scalar>
typename Orientation2D<Scalar>::Matrix2
Orientation2D<Scalar>::toRotationMatrix(void) const
{
  Scalar sinA = ei_sin(m_angle);
  Scalar cosA = ei_cos(m_angle);
  return Matrix2(cosA, -sinA, sinA, cosA);
}

/** \returns the spherical interpolation between \c *this and \a other using
  * parameter \a t. It is equivalent to a linear interpolation.
  */
template<typename Scalar>
Orientation2D<Scalar>
Orientation2D<Scalar>::slerp(Scalar t, const Orientation2D& other) const
{
  return m_angle * (1-t) + t * other;
}


/** \class Transform
  *
  * \brief Represents an homogeneous transformation in a N dimensional space
  *
  * \param _Scalar the scalar type, i.e., the type of the coefficients
  * \param _Dim the dimension of the space
  *
  * The homography is internally represented and stored as a (Dim+1)^2 matrix which
  * is available through the matrix() method.
  *
  * Conversion methods from/to Qt's QMatrix are available if the preprocessor token
  * EIGEN_QT_SUPPORT is defined.
  *
  */
template<typename _Scalar, int _Dim>
class Transform
{
public:

  enum { Dim = _Dim, HDim = _Dim+1 };
  /** the scalar type of the coefficients */
  typedef _Scalar Scalar;
  typedef Matrix<Scalar,HDim,HDim> MatrixType;
  typedef Matrix<Scalar,Dim,Dim> AffineMatrixType;
  typedef Block<MatrixType,Dim,Dim> AffineMatrixRef;
  typedef Matrix<Scalar,Dim,1> VectorType;
  typedef Block<MatrixType,Dim,1> VectorRef;
  typedef typename ei_get_orientation_type<Scalar,Dim>::type Orientation;

protected:

  MatrixType m_matrix;

  template<typename Other,
  int OtherRows=Other::RowsAtCompileTime,
  int OtherCols=Other::ColsAtCompileTime>
  struct ei_transform_product_impl;

public:

  /** Default constructor without initialization of the coefficients. */
  Transform() { }

  inline Transform(const Transform& other)
  { m_matrix = other.m_matrix; }

  inline Transform& operator=(const Transform& other)
  { m_matrix = other.m_matrix; }

  template<typename OtherDerived>
  inline explicit Transform(const MatrixBase<OtherDerived>& other)
  { m_matrix = other; }

  template<typename OtherDerived>
  inline Transform& operator=(const MatrixBase<OtherDerived>& other)
  { m_matrix = other; }

  #ifdef EIGEN_QT_SUPPORT
  inline Transform(const QMatrix& other);
  inline Transform& operator=(const QMatrix& other);
  inline QMatrix toQMatrix(void) const;
  #endif

  /** \returns a read-only expression of the transformation matrix */
  inline const MatrixType matrix() const { return m_matrix; }
  /** \returns a writable expression of the transformation matrix */
  inline MatrixType matrix() { return m_matrix; }

  /** \returns a read-only expression of the affine (linear) part of the transformation */
  inline const AffineMatrixRef affine() const { return m_matrix.template block<Dim,Dim>(0,0); }
  /** \returns a writable expression of the affine (linear) part of the transformation */
  inline AffineMatrixRef affine() { return m_matrix.template block<Dim,Dim>(0,0); }

  /** \returns a read-only expression of the translation vector of the transformation */
  inline const VectorRef translation() const { return m_matrix.template block<Dim,1>(0,Dim); }
  /** \returns a writable expression of the translation vector of the transformation */
  inline VectorRef translation() { return m_matrix.template block<Dim,1>(0,Dim); }

  template<typename OtherDerived>
  struct ProductReturnType
  {
    typedef typename ei_transform_product_impl<OtherDerived>::ResultType Type;
  };

  template<typename OtherDerived>
  const typename ProductReturnType<OtherDerived>::Type
  operator * (const MatrixBase<OtherDerived> &other) const;

  /** Contatenates two transformations */
  Product<MatrixType,MatrixType>
  operator * (const Transform& other) const
  { return m_matrix * other.matrix(); }

  void setIdentity() { m_matrix.setIdentity(); }

  template<typename OtherDerived>
  Transform& scale(const MatrixBase<OtherDerived> &other);

  template<typename OtherDerived>
  Transform& prescale(const MatrixBase<OtherDerived> &other);

  template<typename OtherDerived>
  Transform& translate(const MatrixBase<OtherDerived> &other);

  template<typename OtherDerived>
  Transform& pretranslate(const MatrixBase<OtherDerived> &other);

  template<typename OtherDerived>
  Transform& shear(Scalar sx, Scalar sy);

  template<typename OtherDerived>
  Transform& preshear(Scalar sx, Scalar sy);

  AffineMatrixType extractRotation() const;
  AffineMatrixType extractRotationNoShear() const;

  template<typename PositionDerived, typename ScaleDerived>
  Transform& fromPositionOrientationScale(const MatrixBase<PositionDerived> &position,
    const Orientation& orientation, const MatrixBase<ScaleDerived> &scale);

  const Inverse<MatrixType, false> inverse() const
  { return m_matrix.inverse(); }

protected:

};

#ifdef EIGEN_QT_SUPPORT
/** Initialises \c *this from a QMatrix assuming the dimension is 2.
  */
template<typename Scalar, int Dim>
Transform<Scalar,Dim>::Transform(const QMatrix& other)
{
  *this = other;
}

/** Set \c *this from a QMatrix assuming the dimension is 2.
  */
template<typename Scalar, int Dim>
Transform<Scalar,Dim>& Transform<Scalar,Dim>::operator=(const QMatrix& other)
{
  EIGEN_STATIC_ASSERT(Dim==2, you_did_a_programming_error);
  m_matrix << other.m11(), other.m21(), other.dx(),
              other.m12(), other.m22(), other.dy(),
              0, 0, 1;
}

/** \returns a QMatrix from \c *this assuming the dimension is 2.
  */
template<typename Scalar, int Dim>
QMatrix Transform<Scalar,Dim>::toQMatrix(void) const
{
  EIGEN_STATIC_ASSERT(Dim==2, you_did_a_programming_error);
  return QMatrix( other.coeffRef(0,0), other.coeffRef(1,0),
                  other.coeffRef(0,1), other.coeffRef(1,1),
                  other.coeffRef(0,2), other.coeffRef(1,2),
}
#endif

template<typename Scalar, int Dim>
template<typename OtherDerived>
const typename Transform<Scalar,Dim>::template ProductReturnType<OtherDerived>::Type
Transform<Scalar,Dim>::operator*(const MatrixBase<OtherDerived> &other) const
{
  return ei_transform_product_impl<OtherDerived>::run(*this,other.derived());
}

/** Applies on the right the non uniform scale transformation represented
  * by the vector \a other to \c *this and returns a reference to \c *this.
  * \sa prescale()
  */
template<typename Scalar, int Dim>
template<typename OtherDerived>
Transform<Scalar,Dim>&
Transform<Scalar,Dim>::scale(const MatrixBase<OtherDerived> &other)
{
  EIGEN_STATIC_ASSERT(int(OtherDerived::IsVectorAtCompileTime)
    && int(OtherDerived::SizeAtCompileTime)==int(Dim), you_did_a_programming_error);
  affine() = (affine() * other.asDiagonal()).lazy();
  return *this;
}

/** Applies on the left the non uniform scale transformation represented
  * by the vector \a other to \c *this and returns a reference to \c *this.
  * \sa scale()
  */
template<typename Scalar, int Dim>
template<typename OtherDerived>
Transform<Scalar,Dim>&
Transform<Scalar,Dim>::prescale(const MatrixBase<OtherDerived> &other)
{
  EIGEN_STATIC_ASSERT(int(OtherDerived::IsVectorAtCompileTime)
    && int(OtherDerived::SizeAtCompileTime)==int(Dim), you_did_a_programming_error);
  m_matrix.template block<3,4>(0,0) = (other.asDiagonal() * m_matrix.template block<3,4>(0,0)).lazy();
  return *this;
}

/** Applies on the right translation matrix represented by the vector \a other
  * to \c *this and returns a reference to \c *this.
  * \sa pretranslate()
  */
template<typename Scalar, int Dim>
template<typename OtherDerived>
Transform<Scalar,Dim>&
Transform<Scalar,Dim>::translate(const MatrixBase<OtherDerived> &other)
{
  EIGEN_STATIC_ASSERT(int(OtherDerived::IsVectorAtCompileTime)
    && int(OtherDerived::SizeAtCompileTime)==int(Dim), you_did_a_programming_error);
  translation() += affine() * other;
  return *this;
}

/** Applies on the left translation matrix represented by the vector \a other
  * to \c *this and returns a reference to \c *this.
  * \sa translate()
  */
template<typename Scalar, int Dim>
template<typename OtherDerived>
Transform<Scalar,Dim>&
Transform<Scalar,Dim>::pretranslate(const MatrixBase<OtherDerived> &other)
{
  EIGEN_STATIC_ASSERT(int(OtherDerived::IsVectorAtCompileTime)
    && int(OtherDerived::SizeAtCompileTime)==int(Dim), you_did_a_programming_error);
  translation() += other;
  return *this;
}

/** Applies on the right the shear transformation represented
  * by the vector \a other to \c *this and returns a reference to \c *this.
  * \warning 2D only.
  * \sa preshear()
  */
template<typename Scalar, int Dim>
template<typename OtherDerived>
Transform<Scalar,Dim>&
Transform<Scalar,Dim>::shear(Scalar sx, Scalar sy)
{
  EIGEN_STATIC_ASSERT(int(OtherDerived::IsVectorAtCompileTime)
    && int(OtherDerived::SizeAtCompileTime)==int(Dim) && int(Dim)==2, you_did_a_programming_error);
  VectorType tmp = affine().col(0)*sy + affine().col(1);
  affine() << affine().col(0) + affine().col(1)*sx, tmp;
  return *this;
}

/** Applies on the left the shear transformation represented
  * by the vector \a other to \c *this and returns a reference to \c *this.
  * \warning 2D only.
  * \sa shear()
  */
template<typename Scalar, int Dim>
template<typename OtherDerived>
Transform<Scalar,Dim>&
Transform<Scalar,Dim>::preshear(Scalar sx, Scalar sy)
{
  EIGEN_STATIC_ASSERT(int(OtherDerived::IsVectorAtCompileTime)
    && int(OtherDerived::SizeAtCompileTime)==int(Dim), you_did_a_programming_error);
  m_matrix.template block<3,4>(0,0) = AffineMatrixType(1, sx, sy, 1) * m_matrix.template block<3,4>(0,0);
  return *this;
}

/** \returns the rotation part of the transformation using a QR decomposition.
  * \sa extractRotationNoShear()
  */
template<typename Scalar, int Dim>
typename Transform<Scalar,Dim>::AffineMatrixType
Transform<Scalar,Dim>::extractRotation() const
{
  return affine().qr().matrixQ();
}

/** \returns the rotation part of the transformation assuming no shear in
  * the affine part.
  * \sa extractRotation()
  */
template<typename Scalar, int Dim>
typename Transform<Scalar,Dim>::AffineMatrixType
Transform<Scalar,Dim>::extractRotationNoShear() const
{
  return affine().cwiseAbs2()
            .verticalRedux(ei_scalar_sum_op<Scalar>()).cwiseSqrt();
}

/** Convenient method to set \c *this from a position, orientation and scale
  * of a 3D object.
  */
template<typename Scalar, int Dim>
template<typename PositionDerived, typename ScaleDerived>
Transform<Scalar,Dim>&
Transform<Scalar,Dim>::fromPositionOrientationScale(const MatrixBase<PositionDerived> &position,
  const Orientation& orientation, const MatrixBase<ScaleDerived> &scale)
{
  affine() = orientation.toRotationMatrix();
  translation() = position;
  m_matrix(Dim,Dim) = 1.;
  m_matrix.template block<1,Dim>(Dim,0).setZero();
  affine() *= scale.asDiagonal();
}

//----------

template<typename Scalar, int Dim>
template<typename Other>
struct Transform<Scalar,Dim>::ei_transform_product_impl<Other,Dim+1,Dim+1>
{
  typedef typename Transform<Scalar,Dim>::MatrixType MatrixType;
  typedef Product<MatrixType,Other> ResultType;
  static ResultType run(const Transform<Scalar,Dim>& tr, const Other& other)
  { return tr.matrix() * other; }
};

template<typename Scalar, int Dim>
template<typename Other>
struct Transform<Scalar,Dim>::ei_transform_product_impl<Other,Dim+1,1>
{
  typedef typename Transform<Scalar,Dim>::MatrixType MatrixType;
  typedef Product<MatrixType,Other> ResultType;
  static ResultType run(const Transform<Scalar,Dim>& tr, const Other& other)
  { return tr.matrix() * other; }
};

template<typename Scalar, int Dim>
template<typename Other>
struct Transform<Scalar,Dim>::ei_transform_product_impl<Other,Dim,1>
{
  typedef typename Transform<Scalar,Dim>::AffineMatrixRef MatrixType;
  typedef const CwiseUnaryOp<
      ei_scalar_multiple_op<Scalar>,
      NestByValue<CwiseBinaryOp<
        ei_scalar_sum_op<Scalar>,
        NestByValue<Product<NestByValue<MatrixType>,Other> >,
        NestByValue<typename Transform<Scalar,Dim>::VectorRef> > >
      > ResultType;
  // FIXME shall we offer an optimized version when the last row is know to be 0,0...,0,1 ?
  static ResultType run(const Transform<Scalar,Dim>& tr, const Other& other)
  { return ((tr.affine().nestByValue() * other).nestByValue() + tr.translation().nestByValue()).nestByValue()
          * (Scalar(1) / ( (tr.matrix().template block<1,Dim>(Dim,0) * other).coeff(0) + tr.matrix().coeff(Dim,Dim))); }
};

#endif // EIGEN_TRANSFORM_H
