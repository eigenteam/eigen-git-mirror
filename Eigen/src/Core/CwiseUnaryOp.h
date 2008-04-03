// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
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

#ifndef EIGEN_CWISE_UNARY_OP_H
#define EIGEN_CWISE_UNARY_OP_H

/** \class CwiseUnaryOp
  *
  * \brief Generic expression of a coefficient-wise unary operator of a matrix or a vector
  *
  * \param UnaryOp template functor implementing the operator
  * \param MatrixType the type of the matrix we are applying the unary operator
  *
  * This class represents an expression of a generic unary operator of a matrix or a vector.
  * It is the return type of the unary operator-, of a matrix or a vector, and most
  * of the time this is the only way it is used.
  *
  * \sa class CwiseBinaryOp
  */
template<typename UnaryOp, typename MatrixType>
struct ei_traits<CwiseUnaryOp<UnaryOp, MatrixType> >
{
  typedef typename ei_result_of<
                     UnaryOp(typename MatrixType::Scalar)
                   >::type Scalar;
  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime,
    Flags = MatrixType::Flags,
    CoeffReadCost = MatrixType::CoeffReadCost + UnaryOp::Cost
  };
};

template<typename UnaryOp, typename MatrixType>
class CwiseUnaryOp : ei_no_assignment_operator,
  public MatrixBase<CwiseUnaryOp<UnaryOp, MatrixType> >
{
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(CwiseUnaryOp)

    CwiseUnaryOp(const MatrixType& mat, const UnaryOp& func = UnaryOp())
      : m_matrix(mat), m_functor(func) {}

  private:

    int _rows() const { return m_matrix.rows(); }
    int _cols() const { return m_matrix.cols(); }

    const Scalar _coeff(int row, int col) const
    {
      return m_functor(m_matrix.coeff(row, col));
    }

  protected:
    const typename MatrixType::XprCopy m_matrix;
    const UnaryOp m_functor;
};

/** \returns an expression of a custom coefficient-wise unary operator \a func of *this
  *
  * The template parameter \a CustomUnaryOp is the type of the functor
  * of the custom unary operator.
  *
  * Here is an example:
  * \include class_CwiseUnaryOp.cpp
  *
  * \sa class CwiseUnaryOp, class CwiseBinarOp, MatrixBase::operator-, MatrixBase::cwiseAbs
  */
template<typename Derived>
template<typename CustomUnaryOp>
const CwiseUnaryOp<CustomUnaryOp, Derived>
MatrixBase<Derived>::cwise(const CustomUnaryOp& func) const
{
  return CwiseUnaryOp<CustomUnaryOp, Derived>(derived(), func);
}

/** \internal
  * \brief Template functor to compute the opposite of a scalar
  *
  * \sa class CwiseUnaryOp, MatrixBase::operator-
  */
template<typename Scalar> struct ei_scalar_opposite_op EIGEN_EMPTY_STRUCT {
  const Scalar operator() (const Scalar& a) const { return -a; }
  enum { Cost = NumTraits<Scalar>::AddCost };
};

/** \returns an expression of the opposite of \c *this
  */
template<typename Derived>
const CwiseUnaryOp<ei_scalar_opposite_op<typename ei_traits<Derived>::Scalar>,Derived>
MatrixBase<Derived>::operator-() const
{
  return CwiseUnaryOp<ei_scalar_opposite_op<Scalar>, Derived>(derived());
}

/** \internal
  * \brief Template functor to compute the absolute value of a scalar
  *
  * \sa class CwiseUnaryOp, MatrixBase::cwiseAbs
  */
template<typename Scalar> struct ei_scalar_abs_op EIGEN_EMPTY_STRUCT {
  const Scalar operator() (const Scalar& a) const { return ei_abs(a); }
  enum { Cost = NumTraits<Scalar>::AddCost };
};

/** \returns an expression of the coefficient-wise absolute value of \c *this
  */
template<typename Derived>
const CwiseUnaryOp<ei_scalar_abs_op<typename ei_traits<Derived>::Scalar>,Derived>
MatrixBase<Derived>::cwiseAbs() const
{
  return CwiseUnaryOp<ei_scalar_abs_op<Scalar>,Derived>(derived());
}

/** \internal
  * \brief Template functor to compute the squared absolute value of a scalar
  *
  * \sa class CwiseUnaryOp, MatrixBase::cwiseAbs2
  */
template<typename Scalar> struct ei_scalar_abs2_op EIGEN_EMPTY_STRUCT {
  const Scalar operator() (const Scalar& a) const { return ei_abs2(a); }
  enum { Cost = NumTraits<Scalar>::MulCost };
};

/** \returns an expression of the coefficient-wise squared absolute value of \c *this
  */
template<typename Derived>
const CwiseUnaryOp<ei_scalar_abs2_op<typename ei_traits<Derived>::Scalar>,Derived>
MatrixBase<Derived>::cwiseAbs2() const
{
  return CwiseUnaryOp<ei_scalar_abs2_op<Scalar>,Derived>(derived());
}

/** \internal
  * \brief Template functor to compute the conjugate of a complex value
  *
  * \sa class CwiseUnaryOp, MatrixBase::conjugate()
  */
template<typename Scalar> struct ei_scalar_conjugate_op EIGEN_EMPTY_STRUCT {
  const Scalar operator() (const Scalar& a) const { return ei_conj(a); }
  enum { Cost = NumTraits<Scalar>::IsComplex ? NumTraits<Scalar>::AddCost : 0 };
};

/** \returns an expression of the complex conjugate of *this.
  *
  * \sa adjoint() */
template<typename Derived>
const CwiseUnaryOp<ei_scalar_conjugate_op<typename ei_traits<Derived>::Scalar>, Derived>
MatrixBase<Derived>::conjugate() const
{
  return CwiseUnaryOp<ei_scalar_conjugate_op<Scalar>, Derived>(derived());
}

/** \internal
  * \brief Template functor to cast a scalar to another type
  *
  * \sa class CwiseUnaryOp, MatrixBase::cast()
  */
template<typename Scalar, typename NewType>
struct ei_scalar_cast_op EIGEN_EMPTY_STRUCT {
  typedef NewType result_type;
  const NewType operator() (const Scalar& a) const { return static_cast<NewType>(a); }
  enum { Cost = ei_is_same_type<Scalar, NewType>::ret ? 0 : NumTraits<NewType>::AddCost };
};

/** \returns an expression of *this with the \a Scalar type casted to
  * \a NewScalar.
  *
  * The template parameter \a NewScalar is the type we are casting the scalars to.
  *
  * \sa class CwiseUnaryOp
  */
template<typename Derived>
template<typename NewType>
const CwiseUnaryOp<ei_scalar_cast_op<typename ei_traits<Derived>::Scalar, NewType>, Derived>
MatrixBase<Derived>::cast() const
{
  return CwiseUnaryOp<ei_scalar_cast_op<Scalar, NewType>, Derived>(derived());
}

/** \internal
  * \brief Template functor to multiply a scalar by a fixed other one
  *
  * \sa class CwiseUnaryOp, MatrixBase::operator*, MatrixBase::operator/
  */
template<typename Scalar>
struct ei_scalar_multiple_op {
  ei_scalar_multiple_op(const Scalar& other) : m_other(other) {}
  Scalar operator() (const Scalar& a) const { return a * m_other; }
  const Scalar m_other;
  enum { Cost = NumTraits<Scalar>::MulCost };
};

template<typename Scalar, bool HasFloatingPoint>
struct ei_scalar_quotient1_impl {
  ei_scalar_quotient1_impl(const Scalar& other) : m_other(static_cast<Scalar>(1) / other) {}
  Scalar operator() (const Scalar& a) const { return a * m_other; }
  const Scalar m_other;
  enum { Cost = NumTraits<Scalar>::MulCost };
};

template<typename Scalar>
struct ei_scalar_quotient1_impl<Scalar,false> {
  ei_scalar_quotient1_impl(const Scalar& other) : m_other(other) {}
  Scalar operator() (const Scalar& a) const { return a / m_other; }
  const Scalar m_other;
  enum { Cost = 2 * NumTraits<Scalar>::MulCost };
};

/** \internal
  * \brief Template functor to divide a scalar by a fixed other one
  *
  * This functor is used to implement the quotient of a matrix by
  * a scalar where the scalar type is not a floating point type.
  *
  * \sa class CwiseUnaryOp, MatrixBase::operator/
  */
template<typename Scalar>
struct ei_scalar_quotient1_op : ei_scalar_quotient1_impl<Scalar, NumTraits<Scalar>::HasFloatingPoint > {
  ei_scalar_quotient1_op(const Scalar& other)
    : ei_scalar_quotient1_impl<Scalar, NumTraits<Scalar>::HasFloatingPoint >(other) {}
};

/** \relates MatrixBase */
template<typename Derived>
const CwiseUnaryOp<ei_scalar_multiple_op<typename ei_traits<Derived>::Scalar>, Derived>
MatrixBase<Derived>::operator*(const Scalar& scalar) const
{
  return CwiseUnaryOp<ei_scalar_multiple_op<Scalar>, Derived>
    (derived(), ei_scalar_multiple_op<Scalar>(scalar));
}

/** \relates MatrixBase */
template<typename Derived>
const CwiseUnaryOp<ei_scalar_quotient1_op<typename ei_traits<Derived>::Scalar>, Derived>
MatrixBase<Derived>::operator/(const Scalar& scalar) const
{
  return CwiseUnaryOp<ei_scalar_quotient1_op<Scalar>, Derived>
    (derived(), ei_scalar_quotient1_op<Scalar>(scalar));
}

template<typename Derived>
Derived&
MatrixBase<Derived>::operator*=(const Scalar& other)
{
  return *this = *this * other;
}

template<typename Derived>
Derived&
MatrixBase<Derived>::operator/=(const Scalar& other)
{
  return *this = *this / other;
}

/** \internal
  * \brief Template functor to compute the square root of a scalar
  *
  * \sa class CwiseUnaryOp, MatrixBase::cwiseSqrt()
  */
template<typename Scalar> struct ei_scalar_sqrt_op EIGEN_EMPTY_STRUCT {
  const Scalar operator() (const Scalar& a) const { return ei_sqrt(a); }
  enum { Cost = 5 * NumTraits<Scalar>::MulCost };
};

/** \returns an expression of the coefficient-wise square root of *this. */
template<typename Derived>
const CwiseUnaryOp<ei_scalar_sqrt_op<typename ei_traits<Derived>::Scalar>, Derived>
MatrixBase<Derived>::cwiseSqrt() const
{
  return CwiseUnaryOp<ei_scalar_sqrt_op<Scalar>, Derived>(derived());
}

/** \internal
  * \brief Template functor to compute the exponential of a scalar
  *
  * \sa class CwiseUnaryOp, MatrixBase::cwiseExp()
  */
template<typename Scalar> struct ei_scalar_exp_op EIGEN_EMPTY_STRUCT {
  const Scalar operator() (const Scalar& a) const { return ei_exp(a); }
  enum { Cost = 5 * NumTraits<Scalar>::MulCost };
};

/** \returns an expression of the coefficient-wise exponential of *this. */
template<typename Derived>
const CwiseUnaryOp<ei_scalar_exp_op<typename ei_traits<Derived>::Scalar>, Derived>
MatrixBase<Derived>::cwiseExp() const
{
  return CwiseUnaryOp<ei_scalar_exp_op<Scalar>, Derived>(derived());
}

/** \internal
  * \brief Template functor to compute the logarithm of a scalar
  *
  * \sa class CwiseUnaryOp, MatrixBase::cwiseLog()
  */
template<typename Scalar> struct ei_scalar_log_op EIGEN_EMPTY_STRUCT {
  const Scalar operator() (const Scalar& a) const { return ei_log(a); }
  enum { Cost = 5 * NumTraits<Scalar>::MulCost };
};

/** \returns an expression of the coefficient-wise logarithm of *this. */
template<typename Derived>
const CwiseUnaryOp<ei_scalar_log_op<typename ei_traits<Derived>::Scalar>, Derived>
MatrixBase<Derived>::cwiseLog() const
{
  return CwiseUnaryOp<ei_scalar_log_op<Scalar>, Derived>(derived());
}

/** \internal
  * \brief Template functor to compute the cosine of a scalar
  *
  * \sa class CwiseUnaryOp, MatrixBase::cwiseCos()
  */
template<typename Scalar> struct ei_scalar_cos_op EIGEN_EMPTY_STRUCT {
  const Scalar operator() (const Scalar& a) const { return ei_cos(a); }
  enum { Cost = 5 * NumTraits<Scalar>::MulCost };
};

/** \returns an expression of the coefficient-wise cosine of *this. */
template<typename Derived>
const CwiseUnaryOp<ei_scalar_cos_op<typename ei_traits<Derived>::Scalar>, Derived>
MatrixBase<Derived>::cwiseCos() const
{
  return CwiseUnaryOp<ei_scalar_cos_op<Scalar>, Derived>(derived());
}

/** \internal
  * \brief Template functor to compute the sine of a scalar
  *
  * \sa class CwiseUnaryOp, MatrixBase::cwiseSin()
  */
template<typename Scalar> struct ei_scalar_sin_op EIGEN_EMPTY_STRUCT {
  const Scalar operator() (const Scalar& a) const { return ei_sin(a); }
  enum { Cost = 5 * NumTraits<Scalar>::MulCost };
};

/** \returns an expression of the coefficient-wise sine of *this. */
template<typename Derived>
const CwiseUnaryOp<ei_scalar_sin_op<typename ei_traits<Derived>::Scalar>, Derived>
MatrixBase<Derived>::cwiseSin() const
{
  return CwiseUnaryOp<ei_scalar_sin_op<Scalar>, Derived>(derived());
}

/** \internal
  * \brief Template functor to raise a scalar to a power
  *
  * \sa class CwiseUnaryOp, MatrixBase::cwisePow
  */
template<typename Scalar>
struct ei_scalar_pow_op {
  ei_scalar_pow_op(const Scalar& exponent) : m_exponent(exponent) {}
  Scalar operator() (const Scalar& a) const { return ei_pow(a, m_exponent); }
  const Scalar m_exponent;
  enum { Cost = 5 * NumTraits<Scalar>::MulCost };
};

/** \relates MatrixBase */
template<typename Derived>
const CwiseUnaryOp<ei_scalar_pow_op<typename ei_traits<Derived>::Scalar>, Derived>
MatrixBase<Derived>::cwisePow(const Scalar& exponent) const
{
  return CwiseUnaryOp<ei_scalar_pow_op<Scalar>, Derived>
    (derived(), ei_scalar_pow_op<Scalar>(exponent));
}

#endif // EIGEN_CWISE_UNARY_OP_H
