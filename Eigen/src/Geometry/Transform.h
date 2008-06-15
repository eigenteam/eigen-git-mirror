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

/** \class Transform
  *
  * \brief Represents an homogeneous transformation in a N dimensional space
  *
  * \param _Scalar the scalar type, i.e., the type of the coefficients
  * \param _Dim the dimension of the space
  *
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

protected:

  MatrixType m_matrix;

  template<typename Other,
  int OtherRows=Other::RowsAtCompileTime,
  int OtherCols=Other::ColsAtCompileTime>
  struct ei_transform_product_impl;

public:

  inline const MatrixType matrix() const { return m_matrix; }
  inline MatrixType matrix() { return m_matrix; }

  inline const AffineMatrixRef affine() const { return m_matrix.template block<Dim,Dim>(0,0); }
  inline AffineMatrixRef affine() { return m_matrix.template block<Dim,Dim>(0,0); }

  inline const VectorRef translation() const { return m_matrix.template block<Dim,1>(0,Dim); }
  inline VectorRef translation() { return m_matrix.template block<Dim,1>(0,Dim); }

  template<typename OtherDerived>
  struct ProductReturnType
  {
    typedef typename ei_transform_product_impl<OtherDerived>::ResultType Type;
  };

  template<typename OtherDerived>
  const typename ProductReturnType<OtherDerived>::Type
  operator * (const MatrixBase<OtherDerived> &other) const;

  void setIdentity() { m_matrix.setIdentity(); }

  template<typename OtherDerived>
  Transform& scale(const MatrixBase<OtherDerived> &other);

  template<typename OtherDerived>
  Transform& prescale(const MatrixBase<OtherDerived> &other);

  template<typename OtherDerived>
  Transform& translate(const MatrixBase<OtherDerived> &other);

  template<typename OtherDerived>
  Transform& pretranslate(const MatrixBase<OtherDerived> &other);

  AffineMatrixType extractRotation() const;
  AffineMatrixType extractRotationNoShear() const;

protected:

};

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
  m_matrix.template block<3,4>(0,0) = (other.asDiagonal().eval() * m_matrix.template block<3,4>(0,0)).lazy();
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
  typedef const CwiseBinaryOp<
    ei_scalar_sum_op<Scalar>,
    NestByValue<Product<NestByValue<MatrixType>,Other> >,
    NestByValue<typename Transform<Scalar,Dim>::VectorRef> > ResultType;
  static ResultType run(const Transform<Scalar,Dim>& tr, const Other& other)
  { return (tr.affine().nestByValue() * other).nestByValue() + tr.translation().nestByValue(); }
};

#endif // EIGEN_TRANSFORM_H
