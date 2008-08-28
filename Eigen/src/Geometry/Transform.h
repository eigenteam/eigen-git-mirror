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

// Note that we have to pass Dim and HDim because it is not allowed to use a template
// parameter to define a template specialization. To be more precise, in the following
// specializations, it is not allowed to use Dim+1 instead of HDim.
template< typename Other,
          int Dim,
          int HDim,
          int OtherRows=Other::RowsAtCompileTime,
          int OtherCols=Other::ColsAtCompileTime>
struct ei_transform_product_impl;

/** \geometry_module \ingroup GeometryModule
  *
  * \class Transform
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
  * \sa class Matrix, class Quaternion
  */
template<typename _Scalar, int _Dim>
class Transform
{
public:

  enum {
    Dim = _Dim,     ///< space dimension in which the transformation holds
    HDim = _Dim+1   ///< size of a respective homogeneous vector
  };
  /** the scalar type of the coefficients */
  typedef _Scalar Scalar;
  /** type of the matrix used to represent the transformation */
  typedef Matrix<Scalar,HDim,HDim> MatrixType;
  /** type of the matrix used to represent the linear part of the transformation */
  typedef Matrix<Scalar,Dim,Dim> LinearMatrixType;
  /** type of read/write reference to the linear part of the transformation */
  typedef Block<MatrixType,Dim,Dim> LinearPart;
  /** type of a vector */
  typedef Matrix<Scalar,Dim,1> VectorType;
  /** type of a read/write reference to the translation part of the rotation */
  typedef Block<MatrixType,Dim,1> TranslationPart;

protected:

  MatrixType m_matrix;

public:

  /** Default constructor without initialization of the coefficients. */
  Transform() { }

  inline Transform(const Transform& other)
  { m_matrix = other.m_matrix; }

  inline Transform& operator=(const Transform& other)
  { m_matrix = other.m_matrix; return *this; }

  template<typename OtherDerived, bool select = OtherDerived::RowsAtCompileTime == Dim>
  struct construct_from_matrix
  {
    static inline void run(Transform *transform, const MatrixBase<OtherDerived>& other)
    {
      transform->matrix() = other;
    }
  };

  template<typename OtherDerived> struct construct_from_matrix<OtherDerived, true>
  {
    static inline void run(Transform *transform, const MatrixBase<OtherDerived>& other)
    {
      transform->linear() = other;
      transform->translation().setZero();
      transform->matrix()(Dim,Dim) = Scalar(1);
      transform->matrix().template block<1,Dim>(Dim,0).setZero();
    }
  };

  /** Constructs and initializes a transformation from a Dim^2 or a (Dim+1)^2 matrix. */
  template<typename OtherDerived>
  inline explicit Transform(const MatrixBase<OtherDerived>& other)
  {
    construct_from_matrix<OtherDerived>::run(this, other);
  }

  /** Set \c *this from a (Dim+1)^2 matrix. */
  template<typename OtherDerived>
  inline Transform& operator=(const MatrixBase<OtherDerived>& other)
  { m_matrix = other; return *this; }

  #ifdef EIGEN_QT_SUPPORT
  inline Transform(const QMatrix& other);
  inline Transform& operator=(const QMatrix& other);
  inline QMatrix toQMatrix(void) const;
  #endif

  /** shortcut for m_matrix(row,col);
    * \sa MatrixBase::operaror(int,int) const */
  Scalar operator() (int row, int col) const { return m_matrix(row,col); }
  /** shortcut for m_matrix(row,col);
    * \sa MatrixBase::operaror(int,int) */
  Scalar& operator() (int row, int col) { return m_matrix(row,col); }

  /** \returns a read-only expression of the transformation matrix */
  inline const MatrixType& matrix() const { return m_matrix; }
  /** \returns a writable expression of the transformation matrix */
  inline MatrixType& matrix() { return m_matrix; }

  /** \returns a read-only expression of the linear (linear) part of the transformation */
  inline const LinearPart linear() const { return m_matrix.template block<Dim,Dim>(0,0); }
  /** \returns a writable expression of the linear (linear) part of the transformation */
  inline LinearPart linear() { return m_matrix.template block<Dim,Dim>(0,0); }

  /** \returns a read-only expression of the translation vector of the transformation */
  inline const TranslationPart translation() const { return m_matrix.template block<Dim,1>(0,Dim); }
  /** \returns a writable expression of the translation vector of the transformation */
  inline TranslationPart translation() { return m_matrix.template block<Dim,1>(0,Dim); }

  /** \returns an expression of the product between the transform \c *this and a matrix expression \a other
  *
  * The right hand side \a other might be either:
  * \li a vector of size Dim,
  * \li an homogeneous vector of size Dim+1,
  * \li a transformation matrix of size Dim+1 x Dim+1.
  */
  // note: this function is defined here because some compilers cannot find the respective declaration
  template<typename OtherDerived>
  const typename ei_transform_product_impl<OtherDerived,_Dim,_Dim+1>::ResultType
  operator * (const MatrixBase<OtherDerived> &other) const
  { return ei_transform_product_impl<OtherDerived,Dim,HDim>::run(*this,other.derived()); }

  /** Contatenates two transformations */
  const typename ProductReturnType<MatrixType,MatrixType>::Type
  operator * (const Transform& other) const
  { return m_matrix * other.matrix(); }

  /** \sa MatrixBase::setIdentity() */
  void setIdentity() { m_matrix.setIdentity(); }

  template<typename OtherDerived>
  Transform& scale(const MatrixBase<OtherDerived> &other);

  template<typename OtherDerived>
  Transform& prescale(const MatrixBase<OtherDerived> &other);

  template<typename OtherDerived>
  Transform& translate(const MatrixBase<OtherDerived> &other);

  template<typename OtherDerived>
  Transform& pretranslate(const MatrixBase<OtherDerived> &other);

  template<typename RotationType>
  Transform& rotate(const RotationType& rotation);

  template<typename RotationType>
  Transform& prerotate(const RotationType& rotation);

  Transform& shear(Scalar sx, Scalar sy);
  Transform& preshear(Scalar sx, Scalar sy);

  LinearMatrixType extractRotation() const;
  LinearMatrixType extractRotationNoShear() const;

  template<typename PositionDerived, typename OrientationType, typename ScaleDerived>
  Transform& fromPositionOrientationScale(const MatrixBase<PositionDerived> &position,
    const OrientationType& orientation, const MatrixBase<ScaleDerived> &scale);

  /** \sa MatrixBase::inverse() */
  const MatrixType inverse() const
  { return m_matrix.inverse(); }

  const Scalar* data() const { return m_matrix.data(); }
  Scalar* data() { return m_matrix.data(); }

protected:

};

/** \ingroup GeometryModule */
typedef Transform<float,2> Transform2f;
/** \ingroup GeometryModule */
typedef Transform<float,3> Transform3f;
/** \ingroup GeometryModule */
typedef Transform<double,2> Transform2d;
/** \ingroup GeometryModule */
typedef Transform<double,3> Transform3d;

#ifdef EIGEN_QT_SUPPORT
/** Initialises \c *this from a QMatrix assuming the dimension is 2.
  *
  * This function is available only if the token EIGEN_QT_SUPPORT is defined.
  */
template<typename Scalar, int Dim>
Transform<Scalar,Dim>::Transform(const QMatrix& other)
{
  *this = other;
}

/** Set \c *this from a QMatrix assuming the dimension is 2.
  *
  * This function is available only if the token EIGEN_QT_SUPPORT is defined.
  */
template<typename Scalar, int Dim>
Transform<Scalar,Dim>& Transform<Scalar,Dim>::operator=(const QMatrix& other)
{
  EIGEN_STATIC_ASSERT(Dim==2, you_did_a_programming_error);
  m_matrix << other.m11(), other.m21(), other.dx(),
              other.m12(), other.m22(), other.dy(),
              0, 0, 1;
   return *this;
}

/** \returns a QMatrix from \c *this assuming the dimension is 2.
  *
  * This function is available only if the token EIGEN_QT_SUPPORT is defined.
  */
template<typename Scalar, int Dim>
QMatrix Transform<Scalar,Dim>::toQMatrix(void) const
{
  EIGEN_STATIC_ASSERT(Dim==2, you_did_a_programming_error);
  return QMatrix(other.coeffRef(0,0), other.coeffRef(1,0),
                 other.coeffRef(0,1), other.coeffRef(1,1),
                 other.coeffRef(0,2), other.coeffRef(1,2));
}
#endif

/** Applies on the right the non uniform scale transformation represented
  * by the vector \a other to \c *this and returns a reference to \c *this.
  * \sa prescale()
  */
template<typename Scalar, int Dim>
template<typename OtherDerived>
Transform<Scalar,Dim>&
Transform<Scalar,Dim>::scale(const MatrixBase<OtherDerived> &other)
{
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(OtherDerived,int(Dim));
  linear() = (linear() * other.asDiagonal()).lazy();
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
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(OtherDerived,int(Dim));
  m_matrix.template block<Dim,HDim>(0,0) = (other.asDiagonal() * m_matrix.template block<Dim,HDim>(0,0)).lazy();
  return *this;
}

/** Applies on the right the translation matrix represented by the vector \a other
  * to \c *this and returns a reference to \c *this.
  * \sa pretranslate()
  */
template<typename Scalar, int Dim>
template<typename OtherDerived>
Transform<Scalar,Dim>&
Transform<Scalar,Dim>::translate(const MatrixBase<OtherDerived> &other)
{
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(OtherDerived,int(Dim));
  translation() += linear() * other;
  return *this;
}

/** Applies on the left the translation matrix represented by the vector \a other
  * to \c *this and returns a reference to \c *this.
  * \sa translate()
  */
template<typename Scalar, int Dim>
template<typename OtherDerived>
Transform<Scalar,Dim>&
Transform<Scalar,Dim>::pretranslate(const MatrixBase<OtherDerived> &other)
{
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(OtherDerived,int(Dim));
  translation() += other;
  return *this;
}

/** Applies on the right the rotation represented by the rotation \a rotation
  * to \c *this and returns a reference to \c *this.
  *
  * The template parameter \a RotationType is the type of the rotation which
  * must be registered by ToRotationMatrix<>.
  *
  * Natively supported types includes:
  *   - any scalar (2D),
  *   - a Dim x Dim matrix expression,
  *   - a Quaternion (3D),
  *   - a AngleAxis (3D)
  *
  * This mechanism is easily extendable to support user types such as Euler angles,
  * or a pair of Quaternion for 4D rotations.
  *
  * \sa rotate(Scalar), class Quaternion, class AngleAxis, class ToRotationMatrix, prerotate(RotationType)
  */
template<typename Scalar, int Dim>
template<typename RotationType>
Transform<Scalar,Dim>&
Transform<Scalar,Dim>::rotate(const RotationType& rotation)
{
  linear() *= ToRotationMatrix<Scalar,Dim,RotationType>::convert(rotation);
  return *this;
}

/** Applies on the left the rotation represented by the rotation \a rotation
  * to \c *this and returns a reference to \c *this.
  *
  * See rotate() for further details.
  *
  * \sa rotate()
  */
template<typename Scalar, int Dim>
template<typename RotationType>
Transform<Scalar,Dim>&
Transform<Scalar,Dim>::prerotate(const RotationType& rotation)
{
  m_matrix.template block<Dim,HDim>(0,0) = ToRotationMatrix<Scalar,Dim,RotationType>::convert(rotation)
                                         * m_matrix.template block<Dim,HDim>(0,0);
  return *this;
}

/** Applies on the right the shear transformation represented
  * by the vector \a other to \c *this and returns a reference to \c *this.
  * \warning 2D only.
  * \sa preshear()
  */
template<typename Scalar, int Dim>
Transform<Scalar,Dim>&
Transform<Scalar,Dim>::shear(Scalar sx, Scalar sy)
{
  EIGEN_STATIC_ASSERT(int(Dim)==2, you_did_a_programming_error);
  VectorType tmp = linear().col(0)*sy + linear().col(1);
  linear() << linear().col(0) + linear().col(1)*sx, tmp;
  return *this;
}

/** Applies on the left the shear transformation represented
  * by the vector \a other to \c *this and returns a reference to \c *this.
  * \warning 2D only.
  * \sa shear()
  */
template<typename Scalar, int Dim>
Transform<Scalar,Dim>&
Transform<Scalar,Dim>::preshear(Scalar sx, Scalar sy)
{
  EIGEN_STATIC_ASSERT(int(Dim)==2, you_did_a_programming_error);
  m_matrix.template block<Dim,HDim>(0,0) = LinearMatrixType(1, sx, sy, 1) * m_matrix.template block<Dim,HDim>(0,0);
  return *this;
}

/** \returns the rotation part of the transformation using a QR decomposition.
  * \sa extractRotationNoShear(), class QR
  */
template<typename Scalar, int Dim>
typename Transform<Scalar,Dim>::LinearMatrixType
Transform<Scalar,Dim>::extractRotation() const
{
  return linear().qr().matrixQ();
}

/** \returns the rotation part of the transformation assuming no shear in
  * the linear part.
  * \sa extractRotation()
  */
template<typename Scalar, int Dim>
typename Transform<Scalar,Dim>::LinearMatrixType
Transform<Scalar,Dim>::extractRotationNoShear() const
{
  return linear().cwise().abs2()
            .verticalRedux(ei_scalar_sum_op<Scalar>()).cwise().sqrt();
}

/** Convenient method to set \c *this from a position, orientation and scale
  * of a 3D object.
  */
template<typename Scalar, int Dim>
template<typename PositionDerived, typename OrientationType, typename ScaleDerived>
Transform<Scalar,Dim>&
Transform<Scalar,Dim>::fromPositionOrientationScale(const MatrixBase<PositionDerived> &position,
  const OrientationType& orientation, const MatrixBase<ScaleDerived> &scale)
{
  linear() = ToRotationMatrix<Scalar,Dim,OrientationType>::convert(orientation);
  linear() *= scale.asDiagonal();
  translation() = position;
  m_matrix(Dim,Dim) = 1.;
  m_matrix.template block<1,Dim>(Dim,0).setZero();
  return *this;
}

/***********************************
*** Specializations of operator* ***
***********************************/

template<typename Other, int Dim, int HDim>
struct ei_transform_product_impl<Other,Dim,HDim, HDim,HDim>
{
  typedef Transform<typename Other::Scalar,Dim> TransformType;
  typedef typename TransformType::MatrixType MatrixType;
  typedef typename ProductReturnType<MatrixType,Other>::Type ResultType;
  static ResultType run(const TransformType& tr, const Other& other)
  { return tr.matrix() * other; }
};

template<typename Other, int Dim, int HDim>
struct ei_transform_product_impl<Other,Dim,HDim, HDim,1>
{
  typedef Transform<typename Other::Scalar,Dim> TransformType;
  typedef typename TransformType::MatrixType MatrixType;
  typedef typename ProductReturnType<MatrixType,Other>::Type ResultType;
  static ResultType run(const TransformType& tr, const Other& other)
  { return tr.matrix() * other; }
};

template<typename Other, int Dim, int HDim>
struct ei_transform_product_impl<Other,Dim,HDim, Dim,1>
{
  typedef typename Other::Scalar Scalar;
  typedef Transform<Scalar,Dim> TransformType;
  typedef typename TransformType::LinearPart MatrixType;
  typedef const CwiseUnaryOp<
      ei_scalar_multiple_op<Scalar>,
      NestByValue<CwiseBinaryOp<
        ei_scalar_sum_op<Scalar>,
        NestByValue<typename ProductReturnType<NestByValue<MatrixType>,Other>::Type >,
        NestByValue<typename TransformType::TranslationPart> > >
      > ResultType;
  // FIXME should we offer an optimized version when the last row is known to be 0,0...,0,1 ?
  static ResultType run(const TransformType& tr, const Other& other)
  { return ((tr.linear().nestByValue() * other).nestByValue() + tr.translation().nestByValue()).nestByValue()
          * (Scalar(1) / ( (tr.matrix().template block<1,Dim>(Dim,0) * other).coeff(0) + tr.matrix().coeff(Dim,Dim))); }
};

#endif // EIGEN_TRANSFORM_H
