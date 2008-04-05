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

#ifndef EIGEN_FUNCTORS_H
#define EIGEN_FUNCTORS_H

// associative functors:

/** \internal
  * \brief Template functor to compute the sum of two scalars
  *
  * \sa class CwiseBinaryOp, MatrixBase::operator+, class PartialRedux, MatrixBase::sum()
  */
template<typename Scalar> struct ei_scalar_sum_op EIGEN_EMPTY_STRUCT {
  const Scalar operator() (const Scalar& a, const Scalar& b) const { return a + b; }
};
template<typename Scalar>
struct ei_functor_traits<ei_scalar_sum_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost,
    IsVectorizable = NumTraits<Scalar>::PacketSize>0
  };
};

/** \internal
  * \brief Template functor to compute the product of two scalars
  *
  * \sa class CwiseBinaryOp, MatrixBase::cwiseProduct(), class PartialRedux, MatrixBase::redux()
  */
template<typename Scalar> struct ei_scalar_product_op EIGEN_EMPTY_STRUCT {
  const Scalar operator() (const Scalar& a, const Scalar& b) const { return a * b; }
};
template<typename Scalar>
struct ei_functor_traits<ei_scalar_product_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::MulCost,
    IsVectorizable = NumTraits<Scalar>::PacketSize>0
  };
};

/** \internal
  * \brief Template functor to compute the min of two scalars
  *
  * \sa class CwiseBinaryOp, MatrixBase::cwiseMin, class PartialRedux, MatrixBase::minCoeff()
  */
template<typename Scalar> struct ei_scalar_min_op EIGEN_EMPTY_STRUCT {
  const Scalar operator() (const Scalar& a, const Scalar& b) const { return std::min(a, b); }
};
template<typename Scalar>
struct ei_functor_traits<ei_scalar_min_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost,
    IsVectorizable = NumTraits<Scalar>::PacketSize>0
  };
};

/** \internal
  * \brief Template functor to compute the max of two scalars
  *
  * \sa class CwiseBinaryOp, MatrixBase::cwiseMax, class PartialRedux, MatrixBase::maxCoeff()
  */
template<typename Scalar> struct ei_scalar_max_op EIGEN_EMPTY_STRUCT {
  const Scalar operator() (const Scalar& a, const Scalar& b) const { return std::max(a, b); }
};
template<typename Scalar>
struct ei_functor_traits<ei_scalar_max_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost,
    IsVectorizable = NumTraits<Scalar>::PacketSize>0
  };
};


// other binary functors:

/** \internal
  * \brief Template functor to compute the difference of two scalars
  *
  * \sa class CwiseBinaryOp, MatrixBase::operator-
  */
template<typename Scalar> struct ei_scalar_difference_op EIGEN_EMPTY_STRUCT {
    const Scalar operator() (const Scalar& a, const Scalar& b) const { return a - b; }
};
template<typename Scalar>
struct ei_functor_traits<ei_scalar_difference_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost,
    IsVectorizable = NumTraits<Scalar>::PacketSize>0
  };
};

/** \internal
  * \brief Template functor to compute the quotient of two scalars
  *
  * \sa class CwiseBinaryOp, MatrixBase::cwiseQuotient()
  */
template<typename Scalar> struct ei_scalar_quotient_op EIGEN_EMPTY_STRUCT {
    const Scalar operator() (const Scalar& a, const Scalar& b) const { return a / b; }
};
template<typename Scalar>
struct ei_functor_traits<ei_scalar_quotient_op<Scalar> >
{ enum { Cost = 2 * NumTraits<Scalar>::MulCost, IsVectorizable = false }; };


// unary functors:

/** \internal
  * \brief Template functor to compute the opposite of a scalar
  *
  * \sa class CwiseUnaryOp, MatrixBase::operator-
  */
template<typename Scalar> struct ei_scalar_opposite_op EIGEN_EMPTY_STRUCT {
  const Scalar operator() (const Scalar& a) const { return -a; }
};
template<typename Scalar>
struct ei_functor_traits<ei_scalar_opposite_op<Scalar> >
{ enum { Cost = NumTraits<Scalar>::AddCost, IsVectorizable = false }; };

/** \internal
  * \brief Template functor to compute the absolute value of a scalar
  *
  * \sa class CwiseUnaryOp, MatrixBase::cwiseAbs
  */
template<typename Scalar> struct ei_scalar_abs_op EIGEN_EMPTY_STRUCT {
  const Scalar operator() (const Scalar& a) const { return ei_abs(a); }
};
template<typename Scalar>
struct ei_functor_traits<ei_scalar_abs_op<Scalar> >
{ enum { Cost = NumTraits<Scalar>::AddCost, IsVectorizable = false }; };

/** \internal
  * \brief Template functor to compute the squared absolute value of a scalar
  *
  * \sa class CwiseUnaryOp, MatrixBase::cwiseAbs2
  */
template<typename Scalar> struct ei_scalar_abs2_op EIGEN_EMPTY_STRUCT {
  const Scalar operator() (const Scalar& a) const { return ei_abs2(a); }
  enum { Cost = NumTraits<Scalar>::MulCost };
};
template<typename Scalar>
struct ei_functor_traits<ei_scalar_abs2_op<Scalar> >
{ enum { Cost = NumTraits<Scalar>::MulCost, IsVectorizable = false }; };

/** \internal
  * \brief Template functor to compute the conjugate of a complex value
  *
  * \sa class CwiseUnaryOp, MatrixBase::conjugate()
  */
template<typename Scalar> struct ei_scalar_conjugate_op EIGEN_EMPTY_STRUCT {
  const Scalar operator() (const Scalar& a) const { return ei_conj(a); }
};
template<typename Scalar>
struct ei_functor_traits<ei_scalar_conjugate_op<Scalar> >
{ enum { Cost = NumTraits<Scalar>::IsComplex ? NumTraits<Scalar>::AddCost : 0, IsVectorizable = false }; };

/** \internal
  * \brief Template functor to cast a scalar to another type
  *
  * \sa class CwiseUnaryOp, MatrixBase::cast()
  */
template<typename Scalar, typename NewType>
struct ei_scalar_cast_op EIGEN_EMPTY_STRUCT {
  typedef NewType result_type;
  const NewType operator() (const Scalar& a) const { return static_cast<NewType>(a); }
};
template<typename Scalar, typename NewType>
struct ei_functor_traits<ei_scalar_cast_op<Scalar,NewType> >
{ enum { Cost = ei_is_same_type<Scalar, NewType>::ret ? 0 : NumTraits<NewType>::AddCost, IsVectorizable = false }; };


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
};
template<typename Scalar>
struct ei_functor_traits<ei_scalar_multiple_op<Scalar> >
{ enum { Cost = NumTraits<Scalar>::MulCost, IsVectorizable = false }; };

template<typename Scalar, bool HasFloatingPoint>
struct ei_scalar_quotient1_impl {
  ei_scalar_quotient1_impl(const Scalar& other) : m_other(static_cast<Scalar>(1) / other) {}
  Scalar operator() (const Scalar& a) const { return a * m_other; }
  const Scalar m_other;
};
template<typename Scalar>
struct ei_functor_traits<ei_scalar_quotient1_impl<Scalar,true> >
{ enum { Cost = NumTraits<Scalar>::MulCost, IsVectorizable = false }; };

template<typename Scalar>
struct ei_scalar_quotient1_impl<Scalar,false> {
  ei_scalar_quotient1_impl(const Scalar& other) : m_other(other) {}
  Scalar operator() (const Scalar& a) const { return a / m_other; }
  const Scalar m_other;
  enum { Cost = 2 * NumTraits<Scalar>::MulCost };
};
template<typename Scalar>
struct ei_functor_traits<ei_scalar_quotient1_impl<Scalar,false> >
{ enum { Cost = 2 * NumTraits<Scalar>::MulCost, IsVectorizable = false }; };

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

/** \internal
  * \brief Template functor to compute the square root of a scalar
  *
  * \sa class CwiseUnaryOp, MatrixBase::cwiseSqrt()
  */
template<typename Scalar> struct ei_scalar_sqrt_op EIGEN_EMPTY_STRUCT {
  const Scalar operator() (const Scalar& a) const { return ei_sqrt(a); }
};
template<typename Scalar>
struct ei_functor_traits<ei_scalar_sqrt_op<Scalar> >
{ enum { Cost = 5 * NumTraits<Scalar>::MulCost, IsVectorizable = false }; };

/** \internal
  * \brief Template functor to compute the exponential of a scalar
  *
  * \sa class CwiseUnaryOp, MatrixBase::cwiseExp()
  */
template<typename Scalar> struct ei_scalar_exp_op EIGEN_EMPTY_STRUCT {
  const Scalar operator() (const Scalar& a) const { return ei_exp(a); }
};
template<typename Scalar>
struct ei_functor_traits<ei_scalar_exp_op<Scalar> >
{ enum { Cost = 5 * NumTraits<Scalar>::MulCost, IsVectorizable = false }; };

/** \internal
  * \brief Template functor to compute the logarithm of a scalar
  *
  * \sa class CwiseUnaryOp, MatrixBase::cwiseLog()
  */
template<typename Scalar> struct ei_scalar_log_op EIGEN_EMPTY_STRUCT {
  const Scalar operator() (const Scalar& a) const { return ei_log(a); }
};
template<typename Scalar>
struct ei_functor_traits<ei_scalar_log_op<Scalar> >
{ enum { Cost = 5 * NumTraits<Scalar>::MulCost, IsVectorizable = false }; };

/** \internal
  * \brief Template functor to compute the cosine of a scalar
  *
  * \sa class CwiseUnaryOp, MatrixBase::cwiseCos()
  */
template<typename Scalar> struct ei_scalar_cos_op EIGEN_EMPTY_STRUCT {
  const Scalar operator() (const Scalar& a) const { return ei_cos(a); }
};
template<typename Scalar>
struct ei_functor_traits<ei_scalar_cos_op<Scalar> >
{ enum { Cost = 5 * NumTraits<Scalar>::MulCost, IsVectorizable = false }; };

/** \internal
  * \brief Template functor to compute the sine of a scalar
  *
  * \sa class CwiseUnaryOp, MatrixBase::cwiseSin()
  */
template<typename Scalar> struct ei_scalar_sin_op EIGEN_EMPTY_STRUCT {
  const Scalar operator() (const Scalar& a) const { return ei_sin(a); }
};
template<typename Scalar>
struct ei_functor_traits<ei_scalar_sin_op<Scalar> >
{ enum { Cost = 5 * NumTraits<Scalar>::MulCost, IsVectorizable = false }; };

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
};
template<typename Scalar>
struct ei_functor_traits<ei_scalar_pow_op<Scalar> >
{ enum { Cost = 5 * NumTraits<Scalar>::MulCost, IsVectorizable = false }; };


// default ei_functor_traits for STL functors:

template<typename T>
struct ei_functor_traits<std::multiplies<T> >
{ enum { Cost = NumTraits<T>::MulCost, IsVectorizable = false }; };

template<typename T>
struct ei_functor_traits<std::divides<T> >
{ enum { Cost = NumTraits<T>::MulCost, IsVectorizable = false }; };

template<typename T>
struct ei_functor_traits<std::plus<T> >
{ enum { Cost = NumTraits<T>::AddCost, IsVectorizable = false }; };

template<typename T>
struct ei_functor_traits<std::minus<T> >
{ enum { Cost = NumTraits<T>::AddCost, IsVectorizable = false }; };

template<typename T>
struct ei_functor_traits<std::negate<T> >
{ enum { Cost = NumTraits<T>::AddCost, IsVectorizable = false }; };

template<typename T>
struct ei_functor_traits<std::logical_or<T> >
{ enum { Cost = 1, IsVectorizable = false }; };

template<typename T>
struct ei_functor_traits<std::logical_and<T> >
{ enum { Cost = 1, IsVectorizable = false }; };

template<typename T>
struct ei_functor_traits<std::logical_not<T> >
{ enum { Cost = 1, IsVectorizable = false }; };

template<typename T>
struct ei_functor_traits<std::greater<T> >
{ enum { Cost = 1, IsVectorizable = false }; };

template<typename T>
struct ei_functor_traits<std::less<T> >
{ enum { Cost = 1, IsVectorizable = false }; };

template<typename T>
struct ei_functor_traits<std::greater_equal<T> >
{ enum { Cost = 1, IsVectorizable = false }; };

template<typename T>
struct ei_functor_traits<std::less_equal<T> >
{ enum { Cost = 1, IsVectorizable = false }; };

template<typename T>
struct ei_functor_traits<std::equal_to<T> >
{ enum { Cost = 1, IsVectorizable = false }; };

template<typename T>
struct ei_functor_traits<std::not_equal_to<T> >
{ enum { Cost = 1, IsVectorizable = false }; };

template<typename T>
struct ei_functor_traits<std::binder2nd<T> >
{ enum { Cost = ei_functor_traits<T>::Cost, IsVectorizable = false }; };

template<typename T>
struct ei_functor_traits<std::binder1st<T> >
{ enum { Cost = ei_functor_traits<T>::Cost, IsVectorizable = false }; };

template<typename T>
struct ei_functor_traits<std::unary_negate<T> >
{ enum { Cost = 1 + ei_functor_traits<T>::Cost, IsVectorizable = false }; };

template<typename T>
struct ei_functor_traits<std::binary_negate<T> >
{ enum { Cost = 1 + ei_functor_traits<T>::Cost, IsVectorizable = false }; };

#ifdef EIGEN_STDEXT_SUPPORT

template<typename T0,typename T1>
struct ei_functor_traits<std::project1st<T0,T1> >
{ enum { Cost = 0, IsVectorizable = false }; };

template<typename T0,typename T1>
struct ei_functor_traits<std::project2nd<T0,T1> >
{ enum { Cost = 0, IsVectorizable = false }; };

template<typename T0,typename T1>
struct ei_functor_traits<std::select2nd<std::pair<T0,T1> > >
{ enum { Cost = 0, IsVectorizable = false }; };

template<typename T0,typename T1>
struct ei_functor_traits<std::select1st<std::pair<T0,T1> > >
{ enum { Cost = 0, IsVectorizable = false }; };

template<typename T0,typename T1>
struct ei_functor_traits<std::unary_compose<T0,T1> >
{ enum { Cost = ei_functor_traits<T0>::Cost + ei_functor_traits<T1>::Cost, IsVectorizable = false }; };

template<typename T0,typename T1,typename T2>
struct ei_functor_traits<std::binary_compose<T0,T1,T2> >
{ enum { Cost = ei_functor_traits<T0>::Cost + ei_functor_traits<T1>::Cost + ei_functor_traits<T2>::Cost, IsVectorizable = false }; };

#endif // EIGEN_STDEXT_SUPPORT

#endif // EIGEN_FUNCTORS_H
