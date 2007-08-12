/*
 * Tiny Vector Matrix Library
 * Dense Vector Matrix Libary of Tiny size using Expression Templates
 *
 * Copyright (C) 2001 - 2003 Olaf Petzold <opetzold@users.sourceforge.net>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * $Id: VectorFunctions.h,v 1.17 2005/03/25 07:11:29 opetzold Exp $
 */

#ifndef TVMET_XPR_VECTOR_FUNCTIONS_H
#define TVMET_XPR_VECTOR_FUNCTIONS_H

namespace tvmet {


/* forwards */
template<class T, int Sz> class Vector;


/*********************************************************
 * PART I: DECLARATION
 *********************************************************/


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Vector arithmetic functions add, sub, mul and div
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * function(XprVector<E1, Sz>, XprVector<E2, Sz>)
 */
#define TVMET_DECLARE_MACRO(NAME)					\
template<class E1, class E2, int Sz>				\
XprVector<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E1::value_type, typename E2::value_type>,	\
    XprVector<E1, Sz>,							\
    XprVector<E2, Sz>							\
  >,									\
  Sz									\
>									\
NAME (const XprVector<E1, Sz>& lhs,					\
      const XprVector<E2, Sz>& rhs) _tvmet_always_inline;

TVMET_DECLARE_MACRO(add)		// per se element wise
TVMET_DECLARE_MACRO(sub)		// per se element wise
TVMET_DECLARE_MACRO(mul)		// per se element wise
namespace element_wise {
  TVMET_DECLARE_MACRO(div)		// not defined for vectors
}

#undef TVMET_DECLARE_MACRO


/*
 * function(XprVector<E, Sz>, POD)
 * function(POD, XprVector<E, Sz>)
 * Note: - operations +,-,*,/ are per se element wise
 */
#define TVMET_DECLARE_MACRO(NAME, POD)				\
template<class E, int Sz>				\
XprVector<							\
  XprBinOp<							\
    Fcnl_##NAME< typename E::value_type, POD >,			\
    XprVector<E, Sz>,						\
    XprLiteral< POD >						\
  >,								\
  Sz								\
>								\
NAME (const XprVector<E, Sz>& lhs, 				\
      POD rhs) _tvmet_always_inline;				\
								\
template<class E, int Sz>				\
XprVector<							\
  XprBinOp<							\
    Fcnl_##NAME< POD, typename E::value_type>,			\
    XprLiteral< POD >,						\
    XprVector<E, Sz>						\
  >,								\
  Sz								\
>								\
NAME (POD lhs, 							\
      const XprVector<E, Sz>& rhs) _tvmet_always_inline;

TVMET_DECLARE_MACRO(add, int)
TVMET_DECLARE_MACRO(sub, int)
TVMET_DECLARE_MACRO(mul, int)
TVMET_DECLARE_MACRO(div, int)

TVMET_DECLARE_MACRO(add, float)
TVMET_DECLARE_MACRO(sub, float)
TVMET_DECLARE_MACRO(mul, float)
TVMET_DECLARE_MACRO(div, float)

TVMET_DECLARE_MACRO(add, double)
TVMET_DECLARE_MACRO(sub, double)
TVMET_DECLARE_MACRO(mul, double)
TVMET_DECLARE_MACRO(div, double)

#undef TVMET_DECLARE_MACRO


#if defined(EIGEN_USE_COMPLEX)
/*
 * function(XprMatrix<E, Rows, Cols>, complex<T>)
 * function(complex<T>, XprMatrix<E, Rows, Cols>)
 * Note: - operations +,-,*,/ are per se element wise
 * \todo type promotion
 */
#define TVMET_DECLARE_MACRO(NAME)				\
template<class E, int Sz, class T>			\
XprVector<							\
  XprBinOp<							\
    Fcnl_##NAME< typename E::value_type, std::complex<T> >,	\
    XprVector<E, Sz>,						\
    XprLiteral< std::complex<T> >				\
  >,								\
  Sz								\
>								\
NAME (const XprVector<E, Sz>& lhs,				\
      const std::complex<T>& rhs) _tvmet_always_inline;	\
								\
template<class E, int Sz, class T>			\
XprVector<							\
  XprBinOp<							\
    Fcnl_##NAME< std::complex<T>, typename E::value_type>,	\
    XprLiteral< std::complex<T> >,				\
    XprVector<E, Sz>						\
  >,								\
  Sz								\
>								\
NAME (const std::complex<T>& lhs, 				\
      const XprVector<E, Sz>& rhs) _tvmet_always_inline;

TVMET_DECLARE_MACRO(add)
TVMET_DECLARE_MACRO(sub)
TVMET_DECLARE_MACRO(mul)
TVMET_DECLARE_MACRO(div)

#undef TVMET_DECLARE_MACRO

#endif // defined(EIGEN_USE_COMPLEX)


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * vector specific functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


template<class E, int Sz>
typename Traits<typename E::value_type>::sum_type
sum(const XprVector<E, Sz>& v) _tvmet_always_inline;


template<class E, int Sz>
typename Traits<typename E::value_type>::sum_type
product(const XprVector<E, Sz>& v) _tvmet_always_inline;


template<class E1, class E2, int Sz>
typename PromoteTraits<
  typename E1::value_type,
  typename E2::value_type
>::value_type
dot(const XprVector<E1, Sz>& lhs,
    const XprVector<E2, Sz>& rhs) _tvmet_always_inline;


template<class T, class E, int Sz>
typename PromoteTraits<T, typename E::value_type>::value_type
dot(const Vector<T, Sz>& lhs,
    const XprVector<E, Sz>& rhs) _tvmet_always_inline;


template<class E, class T, int Sz>
typename PromoteTraits<T, typename E::value_type>::value_type
dot(const XprVector<E, Sz>& lhs,
    const Vector<T, Sz>& rhs) _tvmet_always_inline;


template<class E1, class E2>
Vector<
  typename PromoteTraits<
    typename E1::value_type,
    typename E2::value_type
  >::value_type,
  3
>
cross(const XprVector<E1, 3>& lhs,
      const XprVector<E2, 3>& rhs) _tvmet_always_inline;


template<class T, class E>
Vector<
  typename PromoteTraits<T, typename E::value_type>::value_type, 3>
cross(const Vector<T, 3>& lhs,
      const XprVector<E, 3>& rhs) _tvmet_always_inline;


template<class E, class T>
Vector<
  typename PromoteTraits<T, typename E::value_type>::value_type, 3>
cross(const XprVector<E, 3>& lhs,
      const Vector<T, 3>& rhs) _tvmet_always_inline;


template<class E, int Sz>
typename Traits<typename E::value_type>::sum_type
norm1(const XprVector<E, Sz>& v) _tvmet_always_inline;


template<class E, int Sz>
typename Traits<typename E::value_type>::sum_type
norm2(const XprVector<E, Sz>& v) _tvmet_always_inline;


template<class E, int Sz>
XprVector<
  XprBinOp<
    Fcnl_div<typename E::value_type, typename E::value_type>,
    XprVector<E, Sz>,
    XprLiteral<typename E::value_type>
  >,
  Sz
>
normalize(const XprVector<E, Sz>& v) _tvmet_always_inline;


/*********************************************************
 * PART II: IMPLEMENTATION
 *********************************************************/


/*
 * function(XprVector<E1, Sz>, XprVector<E2, Sz>)
 */
#define TVMET_IMPLEMENT_MACRO(NAME)					\
template<class E1, class E2, int Sz>				\
inline									\
XprVector<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E1::value_type, typename E2::value_type>,	\
    XprVector<E1, Sz>,							\
    XprVector<E2, Sz>							\
  >,									\
  Sz									\
>									\
NAME (const XprVector<E1, Sz>& lhs, const XprVector<E2, Sz>& rhs) {	\
  typedef XprBinOp<							\
    Fcnl_##NAME<typename E1::value_type, typename E2::value_type>,	\
    XprVector<E1, Sz>,							\
    XprVector<E2, Sz>							\
  > 							 expr_type;	\
  return XprVector<expr_type, Sz>(expr_type(lhs, rhs));			\
}

TVMET_IMPLEMENT_MACRO(add)		// per se element wise
TVMET_IMPLEMENT_MACRO(sub)		// per se element wise
TVMET_IMPLEMENT_MACRO(mul)		// per se element wise
namespace element_wise {
  TVMET_IMPLEMENT_MACRO(div)		// not defined for vectors
}

#undef TVMET_IMPLEMENT_MACRO


/*
 * function(XprVector<E, Sz>, POD)
 * function(POD, XprVector<E, Sz>)
 * Note: - operations +,-,*,/ are per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, POD)				\
template<class E, int Sz>					\
inline									\
XprVector<								\
  XprBinOp<								\
    Fcnl_##NAME< typename E::value_type, POD >,				\
    XprVector<E, Sz>,							\
    XprLiteral< POD >							\
  >,									\
  Sz									\
>									\
NAME (const XprVector<E, Sz>& lhs, POD rhs) {				\
  typedef XprBinOp<							\
    Fcnl_##NAME< typename E::value_type, POD >,				\
    XprVector<E, Sz>,							\
    XprLiteral< POD >							\
  >							expr_type;	\
  return XprVector<expr_type, Sz>(					\
    expr_type(lhs, XprLiteral< POD >(rhs)));				\
}									\
									\
template<class E, int Sz>					\
inline									\
XprVector<								\
  XprBinOp<								\
    Fcnl_##NAME< POD, typename E::value_type>,				\
    XprLiteral< POD >,							\
    XprVector<E, Sz>							\
  >,									\
  Sz									\
>									\
NAME (POD lhs, const XprVector<E, Sz>& rhs) {				\
  typedef XprBinOp<							\
    Fcnl_##NAME< POD, typename E::value_type>,				\
    XprLiteral< POD >,							\
    XprVector<E, Sz>							\
  >							expr_type;	\
  return XprVector<expr_type, Sz>(					\
    expr_type(XprLiteral< POD >(lhs), rhs));				\
}

TVMET_IMPLEMENT_MACRO(add, int)
TVMET_IMPLEMENT_MACRO(sub, int)
TVMET_IMPLEMENT_MACRO(mul, int)
TVMET_IMPLEMENT_MACRO(div, int)

TVMET_IMPLEMENT_MACRO(add, float)
TVMET_IMPLEMENT_MACRO(sub, float)
TVMET_IMPLEMENT_MACRO(mul, float)
TVMET_IMPLEMENT_MACRO(div, float)

TVMET_IMPLEMENT_MACRO(add, double)
TVMET_IMPLEMENT_MACRO(sub, double)
TVMET_IMPLEMENT_MACRO(mul, double)
TVMET_IMPLEMENT_MACRO(div, double)

#undef TVMET_IMPLEMENT_MACRO


#if defined(EIGEN_USE_COMPLEX)
/*
 * function(XprMatrix<E, Rows, Cols>, complex<T>)
 * function(complex<T>, XprMatrix<E, Rows, Cols>)
 * Note: - operations +,-,*,/ are per se element wise
 * \todo type promotion
 */
#define TVMET_IMPLEMENT_MACRO(NAME)				   \
template<class E, int Sz, class T>				\
inline									\
XprVector<								\
  XprBinOp<								\
    Fcnl_##NAME< typename E::value_type, std::complex<T> >,		\
    XprVector<E, Sz>,							\
    XprLiteral< std::complex<T> >					\
  >,									\
  Sz									\
>									\
NAME (const XprVector<E, Sz>& lhs, const std::complex<T>& rhs) {	\
  typedef XprBinOp<							\
    Fcnl_##NAME< typename E::value_type, std::complex<T> >,		\
    XprVector<E, Sz>,							\
    XprLiteral< std::complex<T> >					\
  >							expr_type;	\
  return XprVector<expr_type, Sz>(					\
    expr_type(lhs, XprLiteral< std::complex<T> >(rhs)));		\
}									\
									\
template<class E, int Sz, class T>				\
inline									\
XprVector<								\
  XprBinOp<								\
    Fcnl_##NAME< std::complex<T>, typename E::value_type>,		\
    XprLiteral< std::complex<T> >,					\
    XprVector<E, Sz>							\
  >,									\
  Sz									\
>									\
NAME (const std::complex<T>& lhs, const XprVector<E, Sz>& rhs) {	\
  typedef XprBinOp<							\
    Fcnl_##NAME< std::complex<T>, typename E::value_type>,		\
    XprLiteral< std::complex<T> >,					\
    XprVector<E, Sz>							\
  >							expr_type;	\
  return XprVector<expr_type, Sz>(					\
    expr_type(XprLiteral< std::complex<T> >(lhs), rhs));		\
}

TVMET_IMPLEMENT_MACRO(add)
TVMET_IMPLEMENT_MACRO(sub)
TVMET_IMPLEMENT_MACRO(mul)
TVMET_IMPLEMENT_MACRO(div)

#undef TVMET_IMPLEMENT_MACRO

#endif // defined(EIGEN_USE_COMPLEX)


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * vector specific functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/**
 * \fn sum(const XprVector<E, Sz>& v)
 * \brief Compute the sum of the vector expression.
 * \ingroup _unary_function
 *
 * Simply compute the sum of the given vector as:
 * \f[
 * \sum_{i = 0}^{Sz-1} v[i]
 * \f]
 */
template<class E, int Sz>
inline
typename Traits<typename E::value_type>::sum_type
sum(const XprVector<E, Sz>& v) {
  return meta::Vector<Sz>::sum(v);
}


/**
 * \fn product(const XprVector<E, Sz>& v)
 * \brief Compute the product of the vector elements.
 * \ingroup _unary_function
 *
 * Simply computer the product of the given vector expression as:
 * \f[
 * \prod_{i = 0}^{Sz - 1} v[i]
 * \f]
 */
template<class E, int Sz>
inline
typename Traits<typename E::value_type>::sum_type
product(const XprVector<E, Sz>& v) {
  return meta::Vector<Sz>::product(v);
}


/**
 * \fn dot(const XprVector<E1, Sz>& lhs, const XprVector<E2, Sz>& rhs)
 * \brief Compute the dot/inner product
 * \ingroup _binary_function
 *
 * Compute the dot product as:
 * \f[
 * \sum_{i = 0}^{Sz - 1} ( lhs[i] * rhs[i] )
 * \f]
 * where lhs is a column vector and rhs is a row vector, both vectors
 * have the same dimension.
 */
template<class E1, class E2, int Sz>
inline
typename PromoteTraits<
  typename E1::value_type,
  typename E2::value_type
>::value_type
dot(const XprVector<E1, Sz>& lhs, const XprVector<E2, Sz>& rhs) {
  return meta::Vector<Sz>::dot(lhs, rhs);
}


/**
 * \fn dot(const Vector<T, Sz>& lhs, const XprVector<E, Sz>& rhs)
 * \brief Compute the dot/inner product
 * \ingroup _binary_function
 *
 * Compute the dot product as:
 * \f[
 * \sum_{i = 0}^{Sz - 1} ( lhs[i] * rhs[i] )
 * \f]
 * where lhs is a column vector and rhs is a row vector, both vectors
 * have the same dimension.
 */
template<class T, class E, int Sz>
inline
typename PromoteTraits<T, typename E::value_type>::value_type
dot(const Vector<T, Sz>& lhs, const XprVector<E, Sz>& rhs) {
  return meta::Vector<Sz>::dot(lhs, rhs);
}


/**
 * \fn dot(const XprVector<E, Sz>& lhs, const Vector<T, Sz>& rhs)
 * \brief Compute the dot/inner product
 * \ingroup _binary_function
 *
 * Compute the dot product as:
 * \f[
 * \sum_{i = 0}^{Sz - 1} ( lhs[i] * rhs[i] )
 * \f]
 * where lhs is a column vector and rhs is a row vector, both vectors
 * have the same dimension.
 */
template<class E, class T, int Sz>
inline
typename PromoteTraits<T, typename E::value_type>::value_type
dot(const XprVector<E, Sz>& lhs, const Vector<T, Sz>& rhs) {
  return meta::Vector<Sz>::dot(lhs, rhs);
}


/**
 * \fn cross(const XprVector<E1, 3>& lhs, const XprVector<E2, 3>& rhs)
 * \brief Compute the cross/outer product
 * \ingroup _binary_function
 * \note working only for vectors of size = 3
 * \todo Implement vector outer product as ET and MT, returning a XprVector
 */
template<class E1, class E2>
inline
Vector<
  typename PromoteTraits<
    typename E1::value_type,
    typename E2::value_type
  >::value_type,
  3
>
cross(const XprVector<E1, 3>& lhs, const XprVector<E2, 3>& rhs) {
  typedef typename PromoteTraits<
    typename E1::value_type,
    typename E2::value_type
  >::value_type						value_type;
  return Vector<value_type, 3>(lhs(1)*rhs(2) - rhs(1)*lhs(2),
			       rhs(0)*lhs(2) - lhs(0)*rhs(2),
			       lhs(0)*rhs(1) - rhs(0)*lhs(1));
}


/**
 * \fn cross(const XprVector<E, 3>& lhs, const Vector<T, 3>& rhs)
 * \brief Compute the cross/outer product
 * \ingroup _binary_function
 * \note working only for vectors of size = 3
 * \todo Implement vector outer product as ET and MT, returning a XprVector
 */
template<class E, class T>
inline
Vector<
  typename PromoteTraits<T, typename E::value_type>::value_type, 3>
cross(const XprVector<E, 3>& lhs, const Vector<T, 3>& rhs) {
  typedef typename PromoteTraits<
    typename E::value_type, T>::value_type 		value_type;
  return Vector<value_type, 3>(lhs(1)*rhs(2) - rhs(1)*lhs(2),
			       rhs(0)*lhs(2) - lhs(0)*rhs(2),
			       lhs(0)*rhs(1) - rhs(0)*lhs(1));
}


/**
 * \fn cross(const Vector<T, 3>& lhs, const XprVector<E, 3>& rhs)
 * \brief Compute the cross/outer product
 * \ingroup _binary_function
 * \note working only for vectors of size = 3
 * \todo Implement vector outer product as ET and MT, returning a XprVector
 */
template<class T1, class E2>
inline
Vector<
  typename PromoteTraits<T1, typename E2::value_type>::value_type, 3>
cross(const Vector<T1, 3>& lhs, const XprVector<E2, 3>& rhs) {
  typedef typename PromoteTraits<
    typename E2::value_type, T1>::value_type 		value_type;
  return Vector<value_type, 3>(lhs(1)*rhs(2) - rhs(1)*lhs(2),
			       rhs(0)*lhs(2) - lhs(0)*rhs(2),
			       lhs(0)*rhs(1) - rhs(0)*lhs(1));
}


/**
 * \fn norm1(const XprVector<E, Sz>& v)
 * \brief The \f$l_1\f$ norm of a vector expression.
 * \ingroup _unary_function
 * The norm of any vector is just the square root of the dot product of
 * a vector with itself, or
 *
 * \f[
 * |Vector<T, Sz> v| = |v| = \sum_{i=0}^{Sz-1}\,|v[i]|
 * \f]
 */
template<class E, int Sz>
inline
typename Traits<typename E::value_type>::sum_type
norm1(const XprVector<E, Sz>& v) {
  return sum(abs(v));
}


/**
 * \fn norm2(const XprVector<E, Sz>& v)
 * \brief The euklidian norm (or \f$l_2\f$ norm) of a vector expression.
 * \ingroup _unary_function
 * The norm of any vector is just the square root of the dot product of
 * a vector with itself, or
 *
 * \f[
 * |Vector<T, Sz> v| = |v| = \sqrt{ \sum_{i=0}^{Sz-1}\,v[i]^2 }
 * \f]
 *
 * \note The internal cast for Vector<int> avoids warnings on sqrt.
 */
template<class E, int Sz>
inline
typename Traits<typename E::value_type>::sum_type
norm2(const XprVector<E, Sz>& v) {
  typedef typename E::value_type			value_type;
  return static_cast<value_type>( std::sqrt(static_cast<value_type>(dot(v, v))) );
}


/**
 * \fn normalize(const XprVector<E, Sz>& v)
 * \brief Normalize the given vector expression.
 * \ingroup _unary_function
 * \sa norm2
 *
 * using the equation:
 * \f[
 * \frac{Vector<T, Sz> v}{\sqrt{ \sum_{i=0}^{Sz-1}\,v[i]^2 }}
 * \f]
 */
template<class E, int Sz>
inline
XprVector<
  XprBinOp<
    Fcnl_div<typename E::value_type, typename E::value_type>,
    XprVector<E, Sz>,
    XprLiteral<typename E::value_type>
  >,
  Sz
>
normalize(const XprVector<E, Sz>& v) {
  typedef typename E::value_type			value_type;
  typedef XprBinOp<
    Fcnl_div<value_type, value_type>,
    XprVector<E, Sz>,
    XprLiteral<value_type>
  >							expr_type;
  return XprVector<expr_type, Sz>(
    expr_type(v, XprLiteral< value_type >(norm2(v))));
}


} // namespace tvmet

#endif // TVMET_XPR_VECTOR_FUNCTIONS_H

// Local Variables:
// mode:C++
// End:
