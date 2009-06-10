// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
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
          int Mode,
          int Dim,
          int HDim,
          int OtherRows=Other::RowsAtCompileTime,
          int OtherCols=Other::ColsAtCompileTime>
struct ei_transform_right_product_impl;

template<typename TransformType> struct ei_transform_take_affine_part;

template< typename Other,
          int Mode,
          int Dim,
          int HDim,
          int OtherRows=Other::RowsAtCompileTime,
          int OtherCols=Other::ColsAtCompileTime>
struct ei_transform_left_product_impl;

template<typename Lhs,typename Rhs> struct ei_transform_transform_product_impl;

template< typename Other,
          int Mode,
          int Dim,
          int HDim,
          int OtherRows=Other::RowsAtCompileTime,
          int OtherCols=Other::ColsAtCompileTime>
struct ei_transform_construct_from_matrix;

/** \geometry_module \ingroup Geometry_Module
  *
  * \class Transform
  *
  * \brief Represents an homogeneous transformation in a N dimensional space
  *
  * \param _Scalar the scalar type, i.e., the type of the coefficients
  * \param _Dim the dimension of the space
  * \param _Mode the type of the transformation. Can be:
  *              - Affine: the transformation is stored as a (Dim+1)^2 matrix,
  *                        where the last row is assumed to be [0 ... 0 1].
  *                        This is the default.
  *              - AffineCompact: the transformation is stored as a (Dim)x(Dim+1) matrix.
  *              - Projective: the transformation is stored as a (Dim+1)^2 matrix
  *                            whithout any assumption.
  *
  * The homography is internally represented and stored by a matrix which
  * is available through the matrix() method. To understand the behavior of
  * this class you have to think a Transform object as its internal
  * matrix representation. The chosen convention is right multiply:
  *
  * \code v' = T * v \endcode
  *
  * Thefore, an affine transformation matrix M is shaped like this:
  *
  * \f$ \left( \begin{array}{cc}
  * linear & translation\\
  * 0 ... 0 & 1
  * \end{array} \right) \f$
  *
  * Note that for a provective transformation the last row can be anything,
  * and then the interpretation of different parts might be sighlty different.
  *
  * However, unlike a plain matrix, the Transform class provides many features
  * simplifying both its assembly and usage. In particular, it can be composed
  * with any other transformations (Transform,Trnaslation,RotationBase,Matrix)
  * and can be directly used to transform implicit homogeneous vectors. All these
  * operations are handled via the operator*. For the composition of transformations,
  * its principle consists to first convert the right/left hand sides of the product
  * to a compatible (Dim+1)^2 matrix and then perform a pure matrix product.
  * Of course, internally, operator* tries to perform the minimal number of operations
  * according to the nature of each terms. Likewise, when applying the transform
  * to non homogeneous vectors, the latters are automatically promoted to homogeneous
  * one before doing the matrix product. The convertions to homogeneous representations
  * are performed as follow:
  *
  * \b Translation t (Dim)x(1):
  * \f$ \left( \begin{array}{cc}
  * I & t \\
  * 0\,...\,0 & 1
  * \end{array} \right) \f$
  *
  * \b Rotation R (Dim)x(Dim):
  * \f$ \left( \begin{array}{cc}
  * R & 0\\
  * 0\,...\,0 & 1
  * \end{array} \right) \f$
  *
  * \b Linear \b Matrix L (Dim)x(Dim):
  * \f$ \left( \begin{array}{cc}
  * L & 0\\
  * 0\,...\,0 & 1
  * \end{array} \right) \f$
  *
  * \b Affine \b Matrix A (Dim)x(Dim+1):
  * \f$ \left( \begin{array}{c}
  * A\\
  * 0\,...\,0\,1
  * \end{array} \right) \f$
  *
  * \b Column \b vector v (Dim)x(1):
  * \f$ \left( \begin{array}{c}
  * v\\
  * 1
  * \end{array} \right) \f$
  *
  * \b Set \b of \b column \b vectors V1...Vn (Dim)x(n):
  * \f$ \left( \begin{array}{ccc}
  * v_1 & ... & v_n\\
  * 1 & ... & 1
  * \end{array} \right) \f$
  *
  * The concatenation of a Tranform object with any kind of other transformation
  * always returns a Transform object.
  *
  * A little execption to the "as pure matrix product" rule is the case of the
  * transformation of non homogeneous vectors by an affine transformation. In
  * that case the last matrix row can be ignored, and the product returns non
  * homogeneous vectors.
  *
  * Since, for instance, a Dim x Dim matrix is interpreted as a linear transformation,
  * it is not possible to directly transform Dim vectors stored in a Dim x Dim matrix.
  * The solution is either to use a Dim x Dynamic matrix or explicitely request a
  * vector transformation by making the vector homogeneous:
  * \code
  * m' = T * m.colwise().homogeneous();
  * \endcode
  * Note that there is zero overhead.
  *
  * Conversion methods from/to Qt's QMatrix and QTransform are available if the
  * preprocessor token EIGEN_QT_SUPPORT is defined.
  *
  * \sa class Matrix, class Quaternion
  */
template<typename _Scalar, int _Dim, int _Mode>
class Transform
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF_VECTORIZABLE_FIXED_SIZE(_Scalar,_Dim==Dynamic ? Dynamic : (_Dim+1)*(_Dim+1))
  enum {
    Mode = _Mode,
    Dim = _Dim,     ///< space dimension in which the transformation holds
    HDim = _Dim+1,  ///< size of a respective homogeneous vector
    Rows = int(Mode)==(AffineCompact) ? Dim : HDim
  };
  /** the scalar type of the coefficients */
  typedef _Scalar Scalar;
  /** type of the matrix used to represent the transformation */
  typedef Matrix<Scalar,Rows,HDim> MatrixType;
  /** type of the matrix used to represent the linear part of the transformation */
  typedef Matrix<Scalar,Dim,Dim> LinearMatrixType;
  /** type of read/write reference to the linear part of the transformation */
  typedef Block<MatrixType,Dim,Dim> LinearPart;
  /** type of read/write reference to the affine part of the transformation */
  typedef typename ei_meta_if<int(Mode)==int(AffineCompact),
                              MatrixType&,
                              Block<MatrixType,Dim,HDim> >::ret AffinePart;
  /** type of read/write reference to the affine part of the transformation */
  typedef typename ei_meta_if<int(Mode)==int(AffineCompact),
                              MatrixType&,
                              NestByValue<Block<MatrixType,Dim,HDim> > >::ret AffinePartNested;
  /** type of a vector */
  typedef Matrix<Scalar,Dim,1> VectorType;
  /** type of a read/write reference to the translation part of the rotation */
  typedef Block<MatrixType,Dim,1> TranslationPart;
  /** corresponding translation type */
  typedef Translation<Scalar,Dim> TranslationType;

protected:

  MatrixType m_matrix;

public:

  /** Default constructor without initialization of the meaningfull coefficients.
    * If Mode==Affine, then the last row is set to [0 ... 0 1] */
  inline Transform()
  {
    if (Mode==Affine)
      makeAffine();
  }

  inline Transform(const Transform& other)
  {
    m_matrix = other.m_matrix;
  }

  inline explicit Transform(const TranslationType& t) { *this = t; }
  inline explicit Transform(const UniformScaling<Scalar>& s) { *this = s; }
  template<typename Derived>
  inline explicit Transform(const RotationBase<Derived, Dim>& r) { *this = r; }

  inline Transform& operator=(const Transform& other)
  { m_matrix = other.m_matrix; return *this; }

  typedef ei_transform_take_affine_part<Transform> take_affine_part;

  /** Constructs and initializes a transformation from a Dim^2 or a (Dim+1)^2 matrix. */
  template<typename OtherDerived>
  inline explicit Transform(const MatrixBase<OtherDerived>& other)
  {
    ei_transform_construct_from_matrix<OtherDerived,Mode,Dim,HDim>::run(this, other.derived());
  }

  /** Set \c *this from a Dim^2 or (Dim+1)^2 matrix. */
  template<typename OtherDerived>
  inline Transform& operator=(const MatrixBase<OtherDerived>& other)
  {
    ei_transform_construct_from_matrix<OtherDerived,Mode,Dim,HDim>::run(this, other.derived());
    return *this;
  }

  template<int OtherMode>
  inline Transform(const Transform<Scalar,Dim,OtherMode>& other)
  {
    ei_assert(OtherMode!=Projective && "You cannot directly assign a projective transform to an affine one.");
    typedef typename Transform<Scalar,Dim,OtherMode>::MatrixType OtherMatrixType;
    ei_transform_construct_from_matrix<OtherMatrixType,Mode,Dim,HDim>::run(this, other.matrix());
  }

  template<typename OtherDerived,typename OtherEvalType>
  Transform(const ReturnByValue<OtherDerived,OtherEvalType>& other)
  {
    other.evalTo(*this);
  }

  template<typename OtherDerived,typename OtherEvalType>
  Transform& operator=(const ReturnByValue<OtherDerived,OtherEvalType>& other)
  {
    other.evalTo(*this);
    return *this;
  }

  #ifdef EIGEN_QT_SUPPORT
  inline Transform(const QMatrix& other);
  inline Transform& operator=(const QMatrix& other);
  inline QMatrix toQMatrix(void) const;
  inline Transform(const QTransform& other);
  inline Transform& operator=(const QTransform& other);
  inline QTransform toQTransform(void) const;
  #endif

  /** shortcut for m_matrix(row,col);
    * \sa MatrixBase::operaror(int,int) const */
  inline Scalar operator() (int row, int col) const { return m_matrix(row,col); }
  /** shortcut for m_matrix(row,col);
    * \sa MatrixBase::operaror(int,int) */
  inline Scalar& operator() (int row, int col) { return m_matrix(row,col); }

  /** \returns a read-only expression of the transformation matrix */
  inline const MatrixType& matrix() const { return m_matrix; }
  /** \returns a writable expression of the transformation matrix */
  inline MatrixType& matrix() { return m_matrix; }

  /** \returns a read-only expression of the linear part of the transformation */
  inline const LinearPart linear() const { return m_matrix.template block<Dim,Dim>(0,0); }
  /** \returns a writable expression of the linear part of the transformation */
  inline LinearPart linear() { return m_matrix.template block<Dim,Dim>(0,0); }

  /** \returns a read-only expression of the Dim x HDim affine part of the transformation */
  inline const AffinePart affine() const { return take_affine_part::run(m_matrix); }
  /** \returns a writable expression of the Dim x HDim affine part of the transformation */
  inline AffinePart affine() { return take_affine_part::run(m_matrix); }

  /** \returns a read-only expression of the translation vector of the transformation */
  inline const TranslationPart translation() const { return m_matrix.template block<Dim,1>(0,Dim); }
  /** \returns a writable expression of the translation vector of the transformation */
  inline TranslationPart translation() { return m_matrix.template block<Dim,1>(0,Dim); }

  /** \returns an expression of the product between the transform \c *this and a matrix expression \a other
    *
    * The right hand side \a other might be either:
    * \li a vector of size Dim,
    * \li an homogeneous vector of size Dim+1,
    * \li a set of vectors of size Dim x Dynamic,
    * \li a set of homogeneous vectors of size Dim+1 x Dynamic,
    * \li a linear transformation matrix of size Dim x Dim,
    * \li an affine transformation matrix of size Dim x Dim+1,
    * \li a transformation matrix of size Dim+1 x Dim+1.
    */
  // note: this function is defined here because some compilers cannot find the respective declaration
  template<typename OtherDerived>
  inline const typename ei_transform_right_product_impl<OtherDerived,Mode,_Dim,_Dim+1>::ResultType
  operator * (const MatrixBase<OtherDerived> &other) const
  { return ei_transform_right_product_impl<OtherDerived,Mode,Dim,HDim>::run(*this,other.derived()); }

  /** \returns the product expression of a transformation matrix \a a times a transform \a b
    *
    * The left hand side \a other might be either:
    * \li a linear transformation matrix of size Dim x Dim,
    * \li an affine transformation matrix of size Dim x Dim+1,
    * \li a general transformation matrix of size Dim+1 x Dim+1.
    */
  template<typename OtherDerived> friend
  inline const typename ei_transform_left_product_impl<OtherDerived,Mode,_Dim,_Dim+1>::ResultType
  operator * (const MatrixBase<OtherDerived> &a, const Transform &b)
  { return ei_transform_left_product_impl<OtherDerived,Mode,Dim,HDim>::run(a.derived(),b); }

  template<typename OtherDerived>
  inline Transform& operator*=(const MatrixBase<OtherDerived>& other) { return *this = *this * other; }

  /** Contatenates two transformations */
  inline const Transform operator * (const Transform& other) const
  {
    return ei_transform_transform_product_impl<Transform,Transform>::run(*this,other);
  }

  /** Contatenates two different transformations */
  template<int OtherMode>
  inline const typename ei_transform_transform_product_impl<
                          Transform,Transform<Scalar,Dim,OtherMode> >::ResultType
  operator * (const Transform<Scalar,Dim,OtherMode>& other) const
  {
    return ei_transform_transform_product_impl<Transform,Transform<Scalar,Dim,OtherMode> >::run(*this,other);
  }

  /** \sa MatrixBase::setIdentity() */
  void setIdentity() { m_matrix.setIdentity(); }

  template<typename OtherDerived>
  inline Transform& scale(const MatrixBase<OtherDerived> &other);

  template<typename OtherDerived>
  inline Transform& prescale(const MatrixBase<OtherDerived> &other);

  inline Transform& scale(Scalar s);
  inline Transform& prescale(Scalar s);

  template<typename OtherDerived>
  inline Transform& translate(const MatrixBase<OtherDerived> &other);

  template<typename OtherDerived>
  inline Transform& pretranslate(const MatrixBase<OtherDerived> &other);

  template<typename RotationType>
  inline Transform& rotate(const RotationType& rotation);

  template<typename RotationType>
  inline Transform& prerotate(const RotationType& rotation);

  Transform& shear(Scalar sx, Scalar sy);
  Transform& preshear(Scalar sx, Scalar sy);

  inline Transform& operator=(const TranslationType& t);
  inline Transform& operator*=(const TranslationType& t) { return translate(t.vector()); }
  inline Transform operator*(const TranslationType& t) const;

  inline Transform& operator=(const UniformScaling<Scalar>& t);
  inline Transform& operator*=(const UniformScaling<Scalar>& s) { return scale(s.factor()); }
  inline Transform operator*(const UniformScaling<Scalar>& s) const;

  template<typename Derived>
  inline Transform& operator=(const RotationBase<Derived,Dim>& r);
  template<typename Derived>
  inline Transform& operator*=(const RotationBase<Derived,Dim>& r) { return rotate(r.toRotationMatrix()); }
  template<typename Derived>
  inline Transform operator*(const RotationBase<Derived,Dim>& r) const;

  LinearMatrixType rotation() const;
  template<typename RotationMatrixType, typename ScalingMatrixType>
  void computeRotationScaling(RotationMatrixType *rotation, ScalingMatrixType *scaling) const;
  template<typename ScalingMatrixType, typename RotationMatrixType>
  void computeScalingRotation(ScalingMatrixType *scaling, RotationMatrixType *rotation) const;

  template<typename PositionDerived, typename OrientationType, typename ScaleDerived>
  Transform& fromPositionOrientationScale(const MatrixBase<PositionDerived> &position,
    const OrientationType& orientation, const MatrixBase<ScaleDerived> &scale);

  inline const MatrixType inverse(TransformTraits traits = (TransformTraits)Mode) const;

  /** \returns a const pointer to the column major internal matrix */
  const Scalar* data() const { return m_matrix.data(); }
  /** \returns a non-const pointer to the column major internal matrix */
  Scalar* data() { return m_matrix.data(); }

  /** \returns \c *this with scalar type casted to \a NewScalarType
    *
    * Note that if \a NewScalarType is equal to the current scalar type of \c *this
    * then this function smartly returns a const reference to \c *this.
    */
  template<typename NewScalarType>
  inline typename ei_cast_return_type<Transform,Transform<NewScalarType,Dim,Mode> >::type cast() const
  { return typename ei_cast_return_type<Transform,Transform<NewScalarType,Dim,Mode> >::type(*this); }

  /** Copy constructor with scalar type conversion */
  template<typename OtherScalarType>
  inline explicit Transform(const Transform<OtherScalarType,Dim,Mode>& other)
  { m_matrix = other.matrix().template cast<Scalar>(); }

  /** \returns \c true if \c *this is approximately equal to \a other, within the precision
    * determined by \a prec.
    *
    * \sa MatrixBase::isApprox() */
  bool isApprox(const Transform& other, typename NumTraits<Scalar>::Real prec = precision<Scalar>()) const
  { return m_matrix.isApprox(other.m_matrix, prec); }

  /** Sets the last row to [0 ... 0 1]
    */
  void makeAffine()
  {
    if(int(Mode)!=int(AffineCompact))
    {
      matrix().template block<1,Dim>(Dim,0).setZero();
      matrix().coeffRef(Dim,Dim) = 1;
    }
  }

  /** \internal
    * \returns the Dim x Dim linear part if the transformation is affine,
    *          and the HDim x Dim part for projective transformations.
    */
  inline Block<MatrixType,int(Mode)==int(Projective)?HDim:Dim,Dim> linearExt()
  { return m_matrix.template block<int(Mode)==int(Projective)?HDim:Dim,Dim>(0,0); }
  /** \internal
    * \returns the Dim x Dim linear part if the transformation is affine,
    *          and the HDim x Dim part for projective transformations.
    */
  inline const Block<MatrixType,int(Mode)==int(Projective)?HDim:Dim,Dim> linearExt() const
  { return m_matrix.template block<int(Mode)==int(Projective)?HDim:Dim,Dim>(0,0); }

  /** \internal
    * \returns the translation part if the transformation is affine,
    *          and the last column for projective transformations.
    */
  inline Block<MatrixType,int(Mode)==int(Projective)?HDim:Dim,1> translationExt()
  { return m_matrix.template block<int(Mode)==int(Projective)?HDim:Dim,1>(0,Dim); }
  /** \internal
    * \returns the translation part if the transformation is affine,
    *          and the last column for projective transformations.
    */
  inline const Block<MatrixType,int(Mode)==int(Projective)?HDim:Dim,1> translationExt() const
  { return m_matrix.template block<int(Mode)==int(Projective)?HDim:Dim,1>(0,Dim); }

};

/** \ingroup Geometry_Module */
typedef Transform<float,2> Transform2f;
/** \ingroup Geometry_Module */
typedef Transform<float,3> Transform3f;
/** \ingroup Geometry_Module */
typedef Transform<double,2> Transform2d;
/** \ingroup Geometry_Module */
typedef Transform<double,3> Transform3d;

/** \ingroup Geometry_Module */
typedef Transform<float,2> Affine2f;
/** \ingroup Geometry_Module */
typedef Transform<float,3> Affine3f;
/** \ingroup Geometry_Module */
typedef Transform<double,2> Affine2d;
/** \ingroup Geometry_Module */
typedef Transform<double,3> Affine3d;

/** \ingroup Geometry_Module */
typedef Transform<float,2,AffineCompact> AffineCompact2f;
/** \ingroup Geometry_Module */
typedef Transform<float,3,AffineCompact> AffineCompact3f;
/** \ingroup Geometry_Module */
typedef Transform<double,2,AffineCompact> AffineCompact2d;
/** \ingroup Geometry_Module */
typedef Transform<double,3,AffineCompact> AffineCompact3d;

/** \ingroup Geometry_Module */
typedef Transform<float,2,Projective> Projective2f;
/** \ingroup Geometry_Module */
typedef Transform<float,3,Projective> Projective3f;
/** \ingroup Geometry_Module */
typedef Transform<double,2,Projective> Projective2d;
/** \ingroup Geometry_Module */
typedef Transform<double,3,Projective> Projective3d;

/**************************
*** Optional QT support ***
**************************/

#ifdef EIGEN_QT_SUPPORT
/** Initialises \c *this from a QMatrix assuming the dimension is 2.
  *
  * This function is available only if the token EIGEN_QT_SUPPORT is defined.
  */
template<typename Scalar, int Dim, int Mode>
Transform<Scalar,Dim,Mode>::Transform(const QMatrix& other)
{
  *this = other;
}

/** Set \c *this from a QMatrix assuming the dimension is 2.
  *
  * This function is available only if the token EIGEN_QT_SUPPORT is defined.
  */
template<typename Scalar, int Dim, int Mode>
Transform<Scalar,Dim,Mode>& Transform<Scalar,Dim,Mode>::operator=(const QMatrix& other)
{
  EIGEN_STATIC_ASSERT(Dim==2, YOU_MADE_A_PROGRAMMING_MISTAKE)
  m_matrix << other.m11(), other.m21(), other.dx(),
              other.m12(), other.m22(), other.dy(),
              0, 0, 1;
   return *this;
}

/** \returns a QMatrix from \c *this assuming the dimension is 2.
  *
  * \warning this convertion might loss data if \c *this is not affine
  *
  * This function is available only if the token EIGEN_QT_SUPPORT is defined.
  */
template<typename Scalar, int Dim, int Mode>
QMatrix Transform<Scalar,Dim,Mode>::toQMatrix(void) const
{
  EIGEN_STATIC_ASSERT(Dim==2, YOU_MADE_A_PROGRAMMING_MISTAKE)
  return QMatrix(other.coeffRef(0,0), other.coeffRef(1,0),
                 other.coeffRef(0,1), other.coeffRef(1,1),
                 other.coeffRef(0,2), other.coeffRef(1,2));
}

/** Initialises \c *this from a QTransform assuming the dimension is 2.
  *
  * This function is available only if the token EIGEN_QT_SUPPORT is defined.
  */
template<typename Scalar, int Dim, int Mode>
Transform<Scalar,Dim,Mode>::Transform(const QTransform& other)
{
  *this = other;
}

/** Set \c *this from a QTransform assuming the dimension is 2.
  *
  * This function is available only if the token EIGEN_QT_SUPPORT is defined.
  */
template<typename Scalar, int Dim, int Mode>
Transform<Scalar,Dim,Mode>& Transform<Scalar,Dim,Mode>::operator=(const QTransform& other)
{
  EIGEN_STATIC_ASSERT(Dim==2, YOU_MADE_A_PROGRAMMING_MISTAKE)
  m_matrix << other.m11(), other.m21(), other.dx(),
              other.m12(), other.m22(), other.dy(),
              other.m13(), other.m23(), other.m33();
   return *this;
}

/** \returns a QTransform from \c *this assuming the dimension is 2.
  *
  * This function is available only if the token EIGEN_QT_SUPPORT is defined.
  */
template<typename Scalar, int Dim, int Mode>
QMatrix Transform<Scalar,Dim,Mode>::toQTransform(void) const
{
  EIGEN_STATIC_ASSERT(Dim==2, YOU_MADE_A_PROGRAMMING_MISTAKE)
  return QTransform(other.coeffRef(0,0), other.coeffRef(1,0), other.coeffRef(2,0)
                    other.coeffRef(0,1), other.coeffRef(1,1), other.coeffRef(2,1)
                    other.coeffRef(0,2), other.coeffRef(1,2), other.coeffRef(2,2);
}
#endif

/*********************
*** Procedural API ***
*********************/

/** Applies on the right the non uniform scale transformation represented
  * by the vector \a other to \c *this and returns a reference to \c *this.
  * \sa prescale()
  */
template<typename Scalar, int Dim, int Mode>
template<typename OtherDerived>
Transform<Scalar,Dim,Mode>&
Transform<Scalar,Dim,Mode>::scale(const MatrixBase<OtherDerived> &other)
{
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(OtherDerived,int(Dim))
  linearExt() = (linearExt() * other.asDiagonal()).lazy();
  return *this;
}

/** Applies on the right a uniform scale of a factor \a c to \c *this
  * and returns a reference to \c *this.
  * \sa prescale(Scalar)
  */
template<typename Scalar, int Dim, int Mode>
inline Transform<Scalar,Dim,Mode>& Transform<Scalar,Dim,Mode>::scale(Scalar s)
{
  linearExt() *= s;
  return *this;
}

/** Applies on the left the non uniform scale transformation represented
  * by the vector \a other to \c *this and returns a reference to \c *this.
  * \sa scale()
  */
template<typename Scalar, int Dim, int Mode>
template<typename OtherDerived>
Transform<Scalar,Dim,Mode>&
Transform<Scalar,Dim,Mode>::prescale(const MatrixBase<OtherDerived> &other)
{
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(OtherDerived,int(Dim))
  m_matrix.template block<Dim,HDim>(0,0) = (other.asDiagonal() * m_matrix.template block<Dim,HDim>(0,0)).lazy();
  return *this;
}

/** Applies on the left a uniform scale of a factor \a c to \c *this
  * and returns a reference to \c *this.
  * \sa scale(Scalar)
  */
template<typename Scalar, int Dim, int Mode>
inline Transform<Scalar,Dim,Mode>& Transform<Scalar,Dim,Mode>::prescale(Scalar s)
{
  m_matrix.template corner<Dim,HDim>(TopLeft) *= s;
  return *this;
}

/** Applies on the right the translation matrix represented by the vector \a other
  * to \c *this and returns a reference to \c *this.
  * \sa pretranslate()
  */
template<typename Scalar, int Dim, int Mode>
template<typename OtherDerived>
Transform<Scalar,Dim,Mode>&
Transform<Scalar,Dim,Mode>::translate(const MatrixBase<OtherDerived> &other)
{
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(OtherDerived,int(Dim))
  translationExt() += linearExt() * other;
  return *this;
}

/** Applies on the left the translation matrix represented by the vector \a other
  * to \c *this and returns a reference to \c *this.
  * \sa translate()
  */
template<typename Scalar, int Dim, int Mode>
template<typename OtherDerived>
Transform<Scalar,Dim,Mode>&
Transform<Scalar,Dim,Mode>::pretranslate(const MatrixBase<OtherDerived> &other)
{
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(OtherDerived,int(Dim))
  if(int(Mode)==int(Projective))
    affine() += other * m_matrix.row(Dim);
  else
    translation() += other;
  return *this;
}

/** Applies on the right the rotation represented by the rotation \a rotation
  * to \c *this and returns a reference to \c *this.
  *
  * The template parameter \a RotationType is the type of the rotation which
  * must be known by ei_toRotationMatrix<>.
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
  * \sa rotate(Scalar), class Quaternion, class AngleAxis, prerotate(RotationType)
  */
template<typename Scalar, int Dim, int Mode>
template<typename RotationType>
Transform<Scalar,Dim,Mode>&
Transform<Scalar,Dim,Mode>::rotate(const RotationType& rotation)
{
  linearExt() *= ei_toRotationMatrix<Scalar,Dim>(rotation);
  return *this;
}

/** Applies on the left the rotation represented by the rotation \a rotation
  * to \c *this and returns a reference to \c *this.
  *
  * See rotate() for further details.
  *
  * \sa rotate()
  */
template<typename Scalar, int Dim, int Mode>
template<typename RotationType>
Transform<Scalar,Dim,Mode>&
Transform<Scalar,Dim,Mode>::prerotate(const RotationType& rotation)
{
  m_matrix.template block<Dim,HDim>(0,0) = ei_toRotationMatrix<Scalar,Dim>(rotation)
                                         * m_matrix.template block<Dim,HDim>(0,0);
  return *this;
}

/** Applies on the right the shear transformation represented
  * by the vector \a other to \c *this and returns a reference to \c *this.
  * \warning 2D only.
  * \sa preshear()
  */
template<typename Scalar, int Dim, int Mode>
Transform<Scalar,Dim,Mode>&
Transform<Scalar,Dim,Mode>::shear(Scalar sx, Scalar sy)
{
  EIGEN_STATIC_ASSERT(int(Dim)==2, YOU_MADE_A_PROGRAMMING_MISTAKE)
  VectorType tmp = linear().col(0)*sy + linear().col(1);
  linear() << linear().col(0) + linear().col(1)*sx, tmp;
  return *this;
}

/** Applies on the left the shear transformation represented
  * by the vector \a other to \c *this and returns a reference to \c *this.
  * \warning 2D only.
  * \sa shear()
  */
template<typename Scalar, int Dim, int Mode>
Transform<Scalar,Dim,Mode>&
Transform<Scalar,Dim,Mode>::preshear(Scalar sx, Scalar sy)
{
  EIGEN_STATIC_ASSERT(int(Dim)==2, YOU_MADE_A_PROGRAMMING_MISTAKE)
  m_matrix.template block<Dim,HDim>(0,0) = LinearMatrixType(1, sx, sy, 1) * m_matrix.template block<Dim,HDim>(0,0);
  return *this;
}

/******************************************************
*** Scaling, Translation and Rotation compatibility ***
******************************************************/

template<typename Scalar, int Dim, int Mode>
inline Transform<Scalar,Dim,Mode>& Transform<Scalar,Dim,Mode>::operator=(const TranslationType& t)
{
  linear().setIdentity();
  translation() = t.vector();
  makeAffine();
  return *this;
}

template<typename Scalar, int Dim, int Mode>
inline Transform<Scalar,Dim,Mode> Transform<Scalar,Dim,Mode>::operator*(const TranslationType& t) const
{
  Transform res = *this;
  res.translate(t.vector());
  return res;
}

template<typename Scalar, int Dim, int Mode>
inline Transform<Scalar,Dim,Mode>& Transform<Scalar,Dim,Mode>::operator=(const UniformScaling<Scalar>& s)
{
  m_matrix.setZero();
  linear().diagonal().fill(s.factor());
  makeAffine();
  return *this;
}

template<typename Scalar, int Dim, int Mode>
inline Transform<Scalar,Dim,Mode> Transform<Scalar,Dim,Mode>::operator*(const UniformScaling<Scalar>& s) const
{
  Transform res = *this;
  res.scale(s.factor());
  return res;
}

template<typename Scalar, int Dim, int Mode>
template<typename Derived>
inline Transform<Scalar,Dim,Mode>& Transform<Scalar,Dim,Mode>::operator=(const RotationBase<Derived,Dim>& r)
{
  linear() = ei_toRotationMatrix<Scalar,Dim>(r);
  translation().setZero();
  makeAffine();
  return *this;
}

template<typename Scalar, int Dim, int Mode>
template<typename Derived>
inline Transform<Scalar,Dim,Mode> Transform<Scalar,Dim,Mode>::operator*(const RotationBase<Derived,Dim>& r) const
{
  Transform res = *this;
  res.rotate(r.derived());
  return res;
}

/************************
*** Special functions ***
************************/

/** \returns the rotation part of the transformation
  * \nonstableyet
  *
  * \svd_module
  *
  * \sa computeRotationScaling(), computeScalingRotation(), class SVD
  */
template<typename Scalar, int Dim, int Mode>
typename Transform<Scalar,Dim,Mode>::LinearMatrixType
Transform<Scalar,Dim,Mode>::rotation() const
{
  LinearMatrixType result;
  computeRotationScaling(&result, (LinearMatrixType*)0);
  return result;
}


/** decomposes the linear part of the transformation as a product rotation x scaling, the scaling being
  * not necessarily positive.
  *
  * If either pointer is zero, the corresponding computation is skipped.
  *
  * \nonstableyet
  *
  * \svd_module
  *
  * \sa computeScalingRotation(), rotation(), class SVD
  */
template<typename Scalar, int Dim, int Mode>
template<typename RotationMatrixType, typename ScalingMatrixType>
void Transform<Scalar,Dim,Mode>::computeRotationScaling(RotationMatrixType *rotation, ScalingMatrixType *scaling) const
{
  linear().svd().computeRotationScaling(rotation, scaling);
}

/** decomposes the linear part of the transformation as a product rotation x scaling, the scaling being
  * not necessarily positive.
  *
  * If either pointer is zero, the corresponding computation is skipped.
  *
  * \nonstableyet
  *
  * \svd_module
  *
  * \sa computeRotationScaling(), rotation(), class SVD
  */
template<typename Scalar, int Dim, int Mode>
template<typename ScalingMatrixType, typename RotationMatrixType>
void Transform<Scalar,Dim,Mode>::computeScalingRotation(ScalingMatrixType *scaling, RotationMatrixType *rotation) const
{
  linear().svd().computeScalingRotation(scaling, rotation);
}

/** Convenient method to set \c *this from a position, orientation and scale
  * of a 3D object.
  */
template<typename Scalar, int Dim, int Mode>
template<typename PositionDerived, typename OrientationType, typename ScaleDerived>
Transform<Scalar,Dim,Mode>&
Transform<Scalar,Dim,Mode>::fromPositionOrientationScale(const MatrixBase<PositionDerived> &position,
  const OrientationType& orientation, const MatrixBase<ScaleDerived> &scale)
{
  linear() = ei_toRotationMatrix<Scalar,Dim>(orientation);
  linear() *= scale.asDiagonal();
  translation() = position;
  makeAffine();
  return *this;
}

/** \nonstableyet
  *
  * \returns the inverse transformation matrix according to some given knowledge
  * on \c *this.
  *
  * \param traits allows to optimize the inversion process when the transformion
  * is known to be not a general transformation. The possible values are:
  *  - Projective if the transformation is not necessarily affine, i.e., if the
  *    last row is not guaranteed to be [0 ... 0 1]
  *  - Affine is the default, the last row is assumed to be [0 ... 0 1]
  *  - Isometry if the transformation is only a concatenations of translations
  *    and rotations.
  *
  * \warning unless \a traits is always set to NoShear or NoScaling, this function
  * requires the generic inverse method of MatrixBase defined in the LU module. If
  * you forget to include this module, then you will get hard to debug linking errors.
  *
  * \sa MatrixBase::inverse()
  */
template<typename Scalar, int Dim, int Mode>
const typename Transform<Scalar,Dim,Mode>::MatrixType
Transform<Scalar,Dim,Mode>::inverse(TransformTraits hint) const
{
  if (hint == Projective)
  {
    return m_matrix.inverse();
  }
  else
  {
    MatrixType res;
    if (hint == Isometry)
    {
      res.template corner<Dim,Dim>(TopLeft) = linear().transpose();
    }
    else if(hint&Affine)
    {
      res.template corner<Dim,Dim>(TopLeft) = linear().inverse();
    }
    else
    {
      ei_assert(false && "Invalid transform traits in Transform::Inverse");
    }
    // translation and remaining parts
    res.template corner<Dim,1>(TopRight) = - res.template corner<Dim,Dim>(TopLeft) * translation();
    res.template corner<1,Dim>(BottomLeft).setZero();
    res.coeffRef(Dim,Dim) = Scalar(1);
    return res;
  }
}

/*****************************************************
*** Specializations of take affine part            ***
*****************************************************/

template<typename TransformType> struct ei_transform_take_affine_part {
  typedef typename TransformType::MatrixType MatrixType;
  typedef typename TransformType::AffinePart AffinePart;
  static inline AffinePart run(MatrixType& m)
  { return m.template block<TransformType::Dim,TransformType::HDim>(0,0); }
  static inline const AffinePart run(const MatrixType& m)
  { return m.template block<TransformType::Dim,TransformType::HDim>(0,0); }
};

template<typename Scalar, int Dim>
struct ei_transform_take_affine_part<Transform<Scalar,Dim,AffineCompact> > {
  typedef typename Transform<Scalar,Dim,AffineCompact>::MatrixType MatrixType;
  static inline MatrixType& run(MatrixType& m) { return m; }
  static inline const MatrixType& run(const MatrixType& m) { return m; }
};

/*****************************************************
*** Specializations of construct from matix        ***
*****************************************************/

template<typename Other, int Mode, int Dim, int HDim>
struct ei_transform_construct_from_matrix<Other, Mode,Dim,HDim, Dim,Dim>
{
  static inline void run(Transform<typename ei_traits<Other>::Scalar,Dim,Mode> *transform, const Other& other)
  {
    transform->linear() = other;
    transform->translation().setZero();
    transform->makeAffine();
  }
};

template<typename Other, int Mode, int Dim, int HDim>
struct ei_transform_construct_from_matrix<Other, Mode,Dim,HDim, Dim,HDim>
{
  static inline void run(Transform<typename ei_traits<Other>::Scalar,Dim,Mode> *transform, const Other& other)
  {
    transform->affine() = other;
    transform->makeAffine();
  }
};

template<typename Other, int Mode, int Dim, int HDim>
struct ei_transform_construct_from_matrix<Other, Mode,Dim,HDim, HDim,HDim>
{
  static inline void run(Transform<typename ei_traits<Other>::Scalar,Dim,Mode> *transform, const Other& other)
  { transform->matrix() = other; }
};

template<typename Other, int Dim, int HDim>
struct ei_transform_construct_from_matrix<Other, AffineCompact,Dim,HDim, HDim,HDim>
{
  static inline void run(Transform<typename ei_traits<Other>::Scalar,Dim,AffineCompact> *transform, const Other& other)
  { transform->matrix() = other.template block<Dim,HDim>(0,0); }
};

/*****************************************************
*** Specializations of operator* with a MatrixBase ***
*****************************************************/

// Projective * set of homogeneous column vectors
template<typename Other, int Dim, int HDim>
struct ei_transform_right_product_impl<Other,Projective, Dim,HDim, HDim, Dynamic>
{
  typedef Transform<typename Other::Scalar,Dim,Projective> TransformType;
  typedef typename TransformType::MatrixType MatrixType;
  typedef typename ProductReturnType<MatrixType,Other>::Type ResultType;
  static ResultType run(const TransformType& tr, const Other& other)
  { return tr.matrix() * other; }
};

// Projective * homogeneous column vector
template<typename Other, int Dim, int HDim>
struct ei_transform_right_product_impl<Other,Projective, Dim,HDim, HDim, 1>
{
  typedef Transform<typename Other::Scalar,Dim,Projective> TransformType;
  typedef typename TransformType::MatrixType MatrixType;
  typedef typename ProductReturnType<MatrixType,Other>::Type ResultType;
  static ResultType run(const TransformType& tr, const Other& other)
  { return tr.matrix() * other; }
};

// Projective * column vector
template<typename Other, int Dim, int HDim>
struct ei_transform_right_product_impl<Other,Projective, Dim,HDim, Dim, 1>
{
  typedef Transform<typename Other::Scalar,Dim,Projective> TransformType;
  typedef Matrix<typename Other::Scalar,HDim,1> ResultType;
  static ResultType run(const TransformType& tr, const Other& other)
  { return tr.matrix().template block<HDim,Dim>(0,0) * other + tr.matrix().col(Dim); }
};

// Affine *  column vector
template<typename Other, int Mode, int Dim, int HDim>
struct ei_transform_right_product_impl<Other,Mode, Dim,HDim, Dim,1>
{
  typedef Transform<typename Other::Scalar,Dim,Mode> TransformType;
  typedef Matrix<typename Other::Scalar,Dim,1> ResultType;
  static ResultType run(const TransformType& tr, const Other& other)
  { return tr.linear() * other + tr.translation(); }
};

// Affine *  homogeneous column vector
// FIXME added for backward compatibility, but I'm not sure we should keep it
template<typename Other, int Mode, int Dim, int HDim>
struct ei_transform_right_product_impl<Other,Mode, Dim,HDim, HDim,1>
{
  typedef Transform<typename Other::Scalar,Dim,Mode> TransformType;
  typedef Matrix<typename Other::Scalar,HDim,1> ResultType;
  static ResultType run(const TransformType& tr, const Other& other)
  { return tr.matrix() * other; }
};
template<typename Other, int Dim, int HDim>
struct ei_transform_right_product_impl<Other,AffineCompact, Dim,HDim, HDim,1>
{
  typedef Transform<typename Other::Scalar,Dim,AffineCompact> TransformType;
  typedef Matrix<typename Other::Scalar,HDim,1> ResultType;
  static ResultType run(const TransformType& tr, const Other& other)
  {
    ResultType res;
    res.template start<HDim>() = tr.matrix() * other;
    res.coeffRef(Dim) = other.coeff(Dim);
  }
};

// T * linear matrix => T
template<typename Other, int Mode, int Dim, int HDim>
struct ei_transform_right_product_impl<Other,Mode, Dim,HDim, Dim,Dim>
{
  typedef Transform<typename Other::Scalar,Dim,Mode> TransformType;
  typedef typename TransformType::MatrixType MatrixType;
  typedef TransformType ResultType;
  static ResultType run(const TransformType& tr, const Other& other)
  {
    TransformType res;
    res.matrix().col(Dim) = tr.matrix().col(Dim);
    res.linearExt() = (tr.linearExt() * other).lazy();
    if(Mode==Affine)
      res.matrix().row(Dim).template start<Dim>() = tr.matrix().row(Dim).template start<Dim>();
    return res;
  }
};

// T * affine matrix => T
template<typename Other, int Mode, int Dim, int HDim>
struct ei_transform_right_product_impl<Other,Mode, Dim,HDim, Dim,HDim>
{
  typedef Transform<typename Other::Scalar,Dim,Mode> TransformType;
  typedef typename TransformType::MatrixType MatrixType;
  typedef TransformType ResultType;
  static ResultType run(const TransformType& tr, const Other& other)
  {
    TransformType res;
    const int Rows = Mode==Projective ? HDim : Dim;
    res.matrix().template block<Rows,HDim>(0,0) = (tr.linearExt() * other).lazy();
    res.translationExt() += tr.translationExt();
    if(Mode!=Affine)
      res.makeAffine();
    return res;
  }
};

// T * generic matrix => Projective
template<typename Other, int Mode, int Dim, int HDim>
struct ei_transform_right_product_impl<Other,Mode, Dim,HDim, HDim,HDim>
{
  typedef Transform<typename Other::Scalar,Dim,Mode> TransformType;
  typedef typename TransformType::MatrixType MatrixType;
  typedef Transform<typename Other::Scalar,Dim,Projective> ResultType;
  static ResultType run(const TransformType& tr, const Other& other)
  { return ResultType((tr.matrix() * other).lazy()); }
};

// AffineCompact * generic matrix => Projective
template<typename Other, int Dim, int HDim>
struct ei_transform_right_product_impl<Other,AffineCompact, Dim,HDim, HDim,HDim>
{
  typedef Transform<typename Other::Scalar,Dim,AffineCompact> TransformType;
  typedef Transform<typename Other::Scalar,Dim,Projective> ResultType;
  static ResultType run(const TransformType& tr, const Other& other)
  {
    ResultType res;
    res.affine() = (tr.matrix() * other).lazy();
    res.makeAffine();
    return res;
  }
};


// generic HDim x HDim matrix * T => Projective
template<typename Other,int Mode, int Dim, int HDim>
struct ei_transform_left_product_impl<Other,Mode,Dim,HDim, HDim,HDim>
{
  typedef Transform<typename Other::Scalar,Dim,Mode> TransformType;
  typedef typename TransformType::MatrixType MatrixType;
  typedef Transform<typename Other::Scalar,Dim,Projective> ResultType;
  static ResultType run(const Other& other,const TransformType& tr)
  { return ResultType((other * tr.matrix()).lazy()); }
};

// generic HDim x HDim matrix * AffineCompact => Projective
template<typename Other, int Dim, int HDim>
struct ei_transform_left_product_impl<Other,AffineCompact,Dim,HDim, HDim,HDim>
{
  typedef Transform<typename Other::Scalar,Dim,AffineCompact> TransformType;
  typedef typename TransformType::MatrixType MatrixType;
  typedef Transform<typename Other::Scalar,Dim,Projective> ResultType;
  static ResultType run(const Other& other,const TransformType& tr)
  {
    ResultType res;
    res.matrix() = (other.template block<HDim,Dim>(0,0) * tr.matrix()).lazy();
    res.matrix().col(Dim) += other.col(Dim);
    return res;
  }
};

// affine matrix * T
template<typename Other,int Mode, int Dim, int HDim>
struct ei_transform_left_product_impl<Other,Mode,Dim,HDim, Dim,HDim>
{
  typedef Transform<typename Other::Scalar,Dim,Mode> TransformType;
  typedef typename TransformType::MatrixType MatrixType;
  typedef TransformType ResultType;
  static ResultType run(const Other& other,const TransformType& tr)
  {
    ResultType res;
    res.affine() = (other * tr.matrix()).lazy();
    res.matrix().row(Dim) = tr.matrix().row(Dim);
    return res;
  }
};

// affine matrix * AffineCompact
template<typename Other, int Dim, int HDim>
struct ei_transform_left_product_impl<Other,AffineCompact,Dim,HDim, Dim,HDim>
{
  typedef Transform<typename Other::Scalar,Dim,AffineCompact> TransformType;
  typedef typename TransformType::MatrixType MatrixType;
  typedef TransformType ResultType;
  static ResultType run(const Other& other,const TransformType& tr)
  {
    ResultType res;
    res.matrix() = (other.template block<Dim,Dim>(0,0) * tr.matrix()).lazy();
    res.translation() += other.col(Dim);
    return res;
  }
};

// linear matrix * T
template<typename Other,int Mode, int Dim, int HDim>
struct ei_transform_left_product_impl<Other,Mode,Dim,HDim, Dim,Dim>
{
  typedef Transform<typename Other::Scalar,Dim,Mode> TransformType;
  typedef typename TransformType::MatrixType MatrixType;
  typedef TransformType ResultType;
  static ResultType run(const Other& other, const TransformType& tr)
  {
    TransformType res;
    if(Mode!=AffineCompact)
      res.matrix().row(Dim) = tr.matrix().row(Dim);
    res.matrix().template corner<Dim,HDim>(TopLeft)
      = (other * tr.matrix().template corner<Dim,HDim>(TopLeft)).lazy();
    return res;
  }
};

/**********************************************************
*** Specializations of operator* with another Transform ***
**********************************************************/

template<typename Scalar, int Dim, int Mode>
struct ei_transform_transform_product_impl<Transform<Scalar,Dim,Mode>,Transform<Scalar,Dim,Mode> >
{
  typedef Transform<Scalar,Dim,Mode> TransformType;
  typedef TransformType ResultType;
  static ResultType run(const TransformType& lhs, const TransformType& rhs)
  {
    return ResultType((lhs.matrix() * rhs.matrix()).lazy());
  }
};

template<typename Scalar, int Dim>
struct ei_transform_transform_product_impl<Transform<Scalar,Dim,AffineCompact>,Transform<Scalar,Dim,AffineCompact> >
{
  typedef Transform<Scalar,Dim,AffineCompact> TransformType;
  typedef TransformType ResultType;
  static ResultType run(const TransformType& lhs, const TransformType& rhs)
  {
    return ei_transform_right_product_impl<typename TransformType::MatrixType,
                                           AffineCompact,Dim,Dim+1>::run(lhs,rhs.matrix());
  }
};

template<typename Scalar, int Dim, int LhsMode, int RhsMode>
struct ei_transform_transform_product_impl<Transform<Scalar,Dim,LhsMode>,Transform<Scalar,Dim,RhsMode> >
{
  typedef Transform<Scalar,Dim,LhsMode> Lhs;
  typedef Transform<Scalar,Dim,RhsMode> Rhs;
  typedef typename ei_transform_right_product_impl<typename Rhs::MatrixType,
                                                   LhsMode,Dim,Dim+1>::ResultType ResultType;
  static ResultType run(const Lhs& lhs, const Rhs& rhs)
  {
    return ei_transform_right_product_impl<typename Rhs::MatrixType,LhsMode,Dim,Dim+1>::run(lhs,rhs.matrix());
  }
};

template<typename Scalar, int Dim>
struct ei_transform_transform_product_impl<Transform<Scalar,Dim,AffineCompact>,
                                           Transform<Scalar,Dim,Affine> >
{
  typedef Transform<Scalar,Dim,AffineCompact> Lhs;
  typedef Transform<Scalar,Dim,Affine> Rhs;
  typedef Transform<Scalar,Dim,AffineCompact> ResultType;
  static ResultType run(const Lhs& lhs, const Rhs& rhs)
  {
    return ResultType((lhs.matrix() * rhs.matrix()).lazy());
  }
};

#endif // EIGEN_TRANSFORM_H
