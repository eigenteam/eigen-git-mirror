/*
 * Tiny Vector Matrix Library
 * Dense Vector Matrix Libary of Tiny size using Expression Templates
 *
 * Copyright (C) 2001 - 2003 Olaf Petzold <opetzold@users.sourceforge.net>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * lesser General Public License for more details.
 *
 * You should have received a copy of the GNU lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * $Id: MatrixBinaryFunctions.h,v 1.12 2005/04/26 15:05:06 opetzold Exp $
 */

#ifndef TVMET_MATRIX_BINARY_FUNCTIONS_H
#define TVMET_MATRIX_BINARY_FUNCTIONS_H

namespace tvmet {

/*********************************************************
 * PART I: DECLARATION
 *********************************************************/

/*
 * binary_function(Matrix<T1, Rows, Cols>, Matrix<T2, Rows, Cols>)
 * binary_function(Matrix<T1, Rows, Cols>, XprMatrix<E, Rows, Cols>)
 * binary_function(XprMatrix<E, Rows, Cols>, Matrix<T, Rows, Cols>)
 */
#define TVMET_DECLARE_MACRO(NAME)					\
template<class T1, class T2, std::size_t Rows, std::size_t Cols>	\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<T1, T2>,						\
    MatrixConstReference<T1, Rows, Cols>,				\
    MatrixConstReference<T2, Rows, Cols>				\
  >,									\
  Rows, Cols								\
>									\
NAME(const Matrix<T1, Rows, Cols>& lhs, 				\
     const Matrix<T2, Cols, Cols>& rhs) TVMET_CXX_ALWAYS_INLINE;	\
									\
template<class E, class T, std::size_t Rows, std::size_t Cols>		\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, T>,				\
    MatrixConstReference<T, Rows, Cols>,				\
    XprMatrix<E, Rows, Cols>						\
  >,									\
  Rows, Cols								\
>									\
NAME(const XprMatrix<E, Rows, Cols>& lhs, 				\
     const Matrix<T, Rows, Cols>& rhs) TVMET_CXX_ALWAYS_INLINE;		\
									\
template<class E, class T, std::size_t Rows, std::size_t Cols>		\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<T, typename E::value_type>,				\
    MatrixConstReference<T, Rows, Cols>,				\
    XprMatrix<E, Rows, Cols>						\
  >,									\
  Rows, Cols								\
>									\
NAME(const Matrix<T, Rows, Cols>& lhs, 					\
     const XprMatrix<E, Rows, Cols>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(atan2)
TVMET_DECLARE_MACRO(drem)
TVMET_DECLARE_MACRO(fmod)
TVMET_DECLARE_MACRO(hypot)
TVMET_DECLARE_MACRO(jn)
TVMET_DECLARE_MACRO(yn)
TVMET_DECLARE_MACRO(pow)
#if defined(TVMET_HAVE_COMPLEX)
TVMET_DECLARE_MACRO(polar)
#endif

#undef TVMET_DECLARE_MACRO


/*
 * binary_function(Matrix<T, Rows, Cols>, POD)
 */
#define TVMET_DECLARE_MACRO(NAME, TP)					\
template<class T, std::size_t Rows, std::size_t Cols>			\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<T, TP >,						\
    MatrixConstReference<T, Rows, Cols>,				\
    XprLiteral< TP >							\
  >,									\
  Rows, Cols								\
>									\
NAME(const Matrix<T, Rows, Cols>& lhs, TP rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(atan2, int)
TVMET_DECLARE_MACRO(drem, int)
TVMET_DECLARE_MACRO(fmod, int)
TVMET_DECLARE_MACRO(hypot, int)
TVMET_DECLARE_MACRO(jn, int)
TVMET_DECLARE_MACRO(yn, int)
TVMET_DECLARE_MACRO(pow, int)

#if defined(TVMET_HAVE_LONG_LONG)
TVMET_DECLARE_MACRO(atan2, long long int)
TVMET_DECLARE_MACRO(drem, long long int)
TVMET_DECLARE_MACRO(fmod, long long int)
TVMET_DECLARE_MACRO(hypot, long long int)
TVMET_DECLARE_MACRO(jn, long long int)
TVMET_DECLARE_MACRO(yn, long long int)
TVMET_DECLARE_MACRO(pow, long long int)
#endif // defined(TVMET_HAVE_LONG_LONG)

TVMET_DECLARE_MACRO(atan2, float)
TVMET_DECLARE_MACRO(drem, float)
TVMET_DECLARE_MACRO(fmod, float)
TVMET_DECLARE_MACRO(hypot, float)
TVMET_DECLARE_MACRO(jn, float)
TVMET_DECLARE_MACRO(yn, float)
TVMET_DECLARE_MACRO(pow, float)

TVMET_DECLARE_MACRO(atan2, double)
TVMET_DECLARE_MACRO(drem, double)
TVMET_DECLARE_MACRO(fmod, double)
TVMET_DECLARE_MACRO(hypot, double)
TVMET_DECLARE_MACRO(jn, double)
TVMET_DECLARE_MACRO(yn, double)
TVMET_DECLARE_MACRO(pow, double)

#if defined(TVMET_HAVE_LONG_DOUBLE)
TVMET_DECLARE_MACRO(atan2, long double)
TVMET_DECLARE_MACRO(drem, long double)
TVMET_DECLARE_MACRO(fmod, long double)
TVMET_DECLARE_MACRO(hypot, long double)
TVMET_DECLARE_MACRO(jn, long double)
TVMET_DECLARE_MACRO(yn, long double)
TVMET_DECLARE_MACRO(pow, long double)
#endif // defined(TVMET_HAVE_LONG_DOUBLE)

#undef TVMET_DECLARE_MACRO


/*
 * complex math
 */

#if defined(TVMET_HAVE_COMPLEX) && defined(TVMET_HAVE_COMPLEX_MATH1)
template<class T, std::size_t Rows, std::size_t Cols>
XprMatrix<
  XprBinOp<
    Fcnl_pow<T, std::complex<T> >,
    MatrixConstReference<T, Rows, Cols>,
    XprLiteral< std::complex<T> >
  >,
  Rows, Cols
>
pow(const Matrix<T, Rows, Cols>& lhs,
    const std::complex<T>& rhs) TVMET_CXX_ALWAYS_INLINE;


template<class T, std::size_t Rows, std::size_t Cols>
XprMatrix<
  XprBinOp<
    Fcnl_pow< std::complex<T>, std::complex<T> >,
    MatrixConstReference<std::complex<T>, Rows, Cols>,
    XprLiteral< std::complex<T> >
  >,
  Rows, Cols
>
pow(const Matrix<std::complex<T>, Rows, Cols>& lhs,
    const std::complex<T>& rhs) TVMET_CXX_ALWAYS_INLINE;


/**
 * \fn pow(const Matrix<std::complex<T>, Rows, Cols>& lhs, const T& rhs)
 * \ingroup _binary_function
 */
template<class T, std::size_t Rows, std::size_t Cols>
XprMatrix<
  XprBinOp<
    Fcnl_pow<std::complex<T>, T>,
    MatrixConstReference<std::complex<T>, Rows, Cols>,
    XprLiteral<T>
  >,
  Rows, Cols
>
pow(const Matrix<std::complex<T>, Rows, Cols>& lhs,
    const T& rhs) TVMET_CXX_ALWAYS_INLINE;


/**
 * \fn pow(const Matrix<std::complex<T>, Rows, Cols>& lhs, int rhs)
 * \ingroup _binary_function
 */
template<class T, std::size_t Rows, std::size_t Cols>
XprMatrix<
  XprBinOp<
    Fcnl_pow<std::complex<T>, int>,
    MatrixConstReference<std::complex<T>, Rows, Cols>,
    XprLiteral<int>
  >,
  Rows, Cols
>
pow(const Matrix<std::complex<T>, Rows, Cols>& lhs,
    int rhs) TVMET_CXX_ALWAYS_INLINE;


template<class T, std::size_t Rows, std::size_t Cols>
XprMatrix<
  XprBinOp<
    Fcnl_polar<T, T>,
    MatrixConstReference<T, Rows, Cols>,
    XprLiteral<T>
  >,
  Rows, Cols
>
polar(const Matrix<T, Rows, Cols>& lhs,
      const T& rhs) TVMET_CXX_ALWAYS_INLINE;

#endif // defined(TVMET_HAVE_COMPLEX) && defined(TVMET_HAVE_COMPLEX_MATH1)


#if defined(TVMET_HAVE_COMPLEX) && defined(TVMET_HAVE_COMPLEX_MATH2)
// to be written (atan2)
#endif // defined(TVMET_HAVE_COMPLEX) && defined(TVMET_HAVE_COMPLEX_MATH2)


/*********************************************************
 * PART II: IMPLEMENTATION
 *********************************************************/

/*
 * binary_function(Matrix<T1, Rows, Cols>, Matrix<T2, Rows, Cols>)
 * binary_function(Matrix<T1, Rows, Cols>, XprMatrix<E, Rows, Cols>)
 * binary_function(XprMatrix<E, Rows, Cols>, Matrix<T, Rows, Cols>)
 */
#define TVMET_IMPLEMENT_MACRO(NAME)						\
template<class T1, class T2, std::size_t Rows, std::size_t Cols>		\
inline										\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME<T1, T2>,							\
    MatrixConstReference<T1, Rows, Cols>,					\
    MatrixConstReference<T2, Rows, Cols>					\
  >,										\
  Rows, Cols									\
>										\
NAME(const Matrix<T1, Rows, Cols>& lhs, const Matrix<T2, Cols, Cols>& rhs) {	\
  typedef XprBinOp <								\
    Fcnl_##NAME<T1, T2>,							\
    MatrixConstReference<T1, Rows, Cols>,					\
    MatrixConstReference<T2, Rows, Cols>					\
  >							expr_type;		\
  return XprMatrix<expr_type, Rows, Cols>(					\
    expr_type(lhs.const_ref(), rhs.const_ref()));				\
}										\
										\
template<class E, class T, std::size_t Rows, std::size_t Cols>			\
inline										\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME<typename E::value_type, T>,					\
    MatrixConstReference<T, Rows, Cols>,					\
    XprMatrix<E, Rows, Cols>							\
  >,										\
  Rows, Cols									\
>										\
NAME(const XprMatrix<E, Rows, Cols>& lhs, const Matrix<T, Rows, Cols>& rhs) {	\
  typedef XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, T>,					\
    XprMatrix<E, Rows, Cols>,							\
    MatrixConstReference<T, Rows, Cols>						\
  > 							 expr_type;		\
  return XprMatrix<expr_type, Rows, Cols>(					\
    expr_type(lhs, rhs.const_ref()));						\
}										\
										\
template<class E, class T, std::size_t Rows, std::size_t Cols>			\
inline										\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME<T, typename E::value_type>,					\
    MatrixConstReference<T, Rows, Cols>,					\
    XprMatrix<E, Rows, Cols>							\
  >,										\
  Rows, Cols									\
>										\
NAME(const Matrix<T, Rows, Cols>& lhs, const XprMatrix<E, Rows, Cols>& rhs) {	\
  typedef XprBinOp<								\
    Fcnl_##NAME<T, typename E::value_type>,					\
    MatrixConstReference<T, Rows, Cols>,					\
    XprMatrix<E, Rows, Cols>							\
  > 						 	expr_type;		\
  return XprMatrix<expr_type, Rows, Cols>(					\
    expr_type(lhs.const_ref(), rhs));						\
}

TVMET_IMPLEMENT_MACRO(atan2)
TVMET_IMPLEMENT_MACRO(drem)
TVMET_IMPLEMENT_MACRO(fmod)
TVMET_IMPLEMENT_MACRO(hypot)
TVMET_IMPLEMENT_MACRO(jn)
TVMET_IMPLEMENT_MACRO(yn)
TVMET_IMPLEMENT_MACRO(pow)
#if defined(TVMET_HAVE_COMPLEX)
TVMET_IMPLEMENT_MACRO(polar)
#endif
#undef TVMET_IMPLEMENT_MACRO


/*
 * binary_function(Matrix<T, Rows, Cols>, POD)
 */
#define TVMET_IMPLEMENT_MACRO(NAME, TP)					\
template<class T, std::size_t Rows, std::size_t Cols>			\
inline									\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<T, TP >,						\
    MatrixConstReference<T, Rows, Cols>,				\
    XprLiteral< TP >							\
  >,									\
  Rows, Cols								\
>									\
NAME(const Matrix<T, Rows, Cols>& lhs, TP rhs) {			\
  typedef XprBinOp<							\
    Fcnl_##NAME<T, TP >,						\
    MatrixConstReference<T, Rows, Cols>,				\
    XprLiteral< TP >							\
  >							expr_type;	\
  return XprMatrix<expr_type, Rows, Cols>(				\
    expr_type(lhs.const_ref(), XprLiteral< TP >(rhs)));			\
}

TVMET_IMPLEMENT_MACRO(atan2, int)
TVMET_IMPLEMENT_MACRO(drem, int)
TVMET_IMPLEMENT_MACRO(fmod, int)
TVMET_IMPLEMENT_MACRO(hypot, int)
TVMET_IMPLEMENT_MACRO(jn, int)
TVMET_IMPLEMENT_MACRO(yn, int)
TVMET_IMPLEMENT_MACRO(pow, int)

#if defined(TVMET_HAVE_LONG_LONG)
TVMET_IMPLEMENT_MACRO(atan2, long long int)
TVMET_IMPLEMENT_MACRO(drem, long long int)
TVMET_IMPLEMENT_MACRO(fmod, long long int)
TVMET_IMPLEMENT_MACRO(hypot, long long int)
TVMET_IMPLEMENT_MACRO(jn, long long int)
TVMET_IMPLEMENT_MACRO(yn, long long int)
TVMET_IMPLEMENT_MACRO(pow, long long int)
#endif // defined(TVMET_HAVE_LONG_LONG)

TVMET_IMPLEMENT_MACRO(atan2, float)
TVMET_IMPLEMENT_MACRO(drem, float)
TVMET_IMPLEMENT_MACRO(fmod, float)
TVMET_IMPLEMENT_MACRO(hypot, float)
TVMET_IMPLEMENT_MACRO(jn, float)
TVMET_IMPLEMENT_MACRO(yn, float)
TVMET_IMPLEMENT_MACRO(pow, float)

TVMET_IMPLEMENT_MACRO(atan2, double)
TVMET_IMPLEMENT_MACRO(drem, double)
TVMET_IMPLEMENT_MACRO(fmod, double)
TVMET_IMPLEMENT_MACRO(hypot, double)
TVMET_IMPLEMENT_MACRO(jn, double)
TVMET_IMPLEMENT_MACRO(yn, double)
TVMET_IMPLEMENT_MACRO(pow, double)

#if defined(TVMET_HAVE_LONG_DOUBLE)
TVMET_IMPLEMENT_MACRO(atan2, long double)
TVMET_IMPLEMENT_MACRO(drem, long double)
TVMET_IMPLEMENT_MACRO(fmod, long double)
TVMET_IMPLEMENT_MACRO(hypot, long double)
TVMET_IMPLEMENT_MACRO(jn, long double)
TVMET_IMPLEMENT_MACRO(yn, long double)
TVMET_IMPLEMENT_MACRO(pow, long double)
#endif // defined(TVMET_HAVE_LONG_DOUBLE)

#undef TVMET_IMPLEMENT_MACRO


/*
 * complex math
 */

#if defined(TVMET_HAVE_COMPLEX) && defined(TVMET_HAVE_COMPLEX_MATH1)
/**
 * \fn pow(const Matrix<T, Rows, Cols>& lhs, const std::complex<T>& rhs)
 * \ingroup _binary_function
 */
template<class T, std::size_t Rows, std::size_t Cols>
inline
XprMatrix<
  XprBinOp<
    Fcnl_pow<T, std::complex<T> >,
    MatrixConstReference<T, Rows, Cols>,
    XprLiteral< std::complex<T> >
  >,
  Rows, Cols
>
pow(const Matrix<T, Rows, Cols>& lhs, const std::complex<T>& rhs) {
  typedef XprBinOp<
    Fcnl_pow<T, std::complex<T> >,
    MatrixConstReference<T, Rows, Cols>,
    XprLiteral< std::complex<T> >
  >							expr_type;
  return XprMatrix<expr_type, Rows, Cols>(
      expr_type(lhs.const_ref(), XprLiteral< std::complex<T> >(rhs)));
}


/**
 * \fn pow(const Matrix<std::complex<T>, Rows, Cols>& lhs, const std::complex<T>& rhs)
 * \ingroup _binary_function
 */
template<class T, std::size_t Rows, std::size_t Cols>
inline
XprMatrix<
  XprBinOp<
    Fcnl_pow< std::complex<T>, std::complex<T> >,
    MatrixConstReference<std::complex<T>, Rows, Cols>,
    XprLiteral< std::complex<T> >
  >,
  Rows, Cols
>
pow(const Matrix<std::complex<T>, Rows, Cols>& lhs, const std::complex<T>& rhs) {
  typedef XprBinOp<
    Fcnl_pow< std::complex<T>, std::complex<T> >,
    MatrixConstReference<std::complex<T>, Rows, Cols>,
    XprLiteral< std::complex<T> >
  >							expr_type;
  return XprMatrix<expr_type, Rows, Cols>(
      expr_type(lhs.const_ref(), XprLiteral< std::complex<T> >(rhs)));
}


/**
 * \fn pow(const Matrix<std::complex<T>, Rows, Cols>& lhs, const T& rhs)
 * \ingroup _binary_function
 */
template<class T, std::size_t Rows, std::size_t Cols>
inline
XprMatrix<
  XprBinOp<
    Fcnl_pow<std::complex<T>, T>,
    MatrixConstReference<std::complex<T>, Rows, Cols>,
    XprLiteral<T>
  >,
  Rows, Cols
>
pow(const Matrix<std::complex<T>, Rows, Cols>& lhs, const T& rhs) {
  typedef XprBinOp<
    Fcnl_pow<std::complex<T>, T>,
    MatrixConstReference<std::complex<T>, Rows, Cols>,
    XprLiteral<T>
  >							expr_type;
  return XprMatrix<expr_type, Rows, Cols>(
      expr_type(lhs.const_ref(), XprLiteral<T>(rhs)));
}


/**
 * \fn pow(const Matrix<std::complex<T>, Rows, Cols>& lhs, int rhs)
 * \ingroup _binary_function
 */
template<class T, std::size_t Rows, std::size_t Cols>
inline
XprMatrix<
  XprBinOp<
    Fcnl_pow<std::complex<T>, int>,
    MatrixConstReference<std::complex<T>, Rows, Cols>,
    XprLiteral<int>
  >,
  Rows, Cols
>
pow(const Matrix<std::complex<T>, Rows, Cols>& lhs, int rhs) {
  typedef XprBinOp<
    Fcnl_pow<std::complex<T>, int>,
    MatrixConstReference<std::complex<T>, Rows, Cols>,
    XprLiteral<int>
  >							expr_type;
  return XprMatrix<expr_type, Rows, Cols>(
      expr_type(lhs.const_ref(), XprLiteral<int>(rhs)));
}


/**
 * \fn polar(const Matrix<T, Rows, Cols>& lhs, const T& rhs)
 * \ingroup _binary_function
 */
template<class T, std::size_t Rows, std::size_t Cols>
inline
XprMatrix<
  XprBinOp<
    Fcnl_polar<T, T>,
    MatrixConstReference<T, Rows, Cols>,
    XprLiteral<T>
  >,
  Rows, Cols
>
polar(const Matrix<T, Rows, Cols>& lhs, const T& rhs) {
  typedef XprBinOp<
    Fcnl_polar<T, T>,
    MatrixConstReference<T, Rows, Cols>,
    XprLiteral<T>
  >							expr_type;
  return XprMatrix<expr_type, Rows, Cols>(
      expr_type(lhs.const_ref(), XprLiteral<T>(rhs)));
}

#endif // defined(TVMET_HAVE_COMPLEX) && defined(TVMET_HAVE_COMPLEX_MATH1)

#if defined(TVMET_HAVE_COMPLEX) && defined(TVMET_HAVE_COMPLEX_MATH2)
// to be written (atan2)
#endif // defined(TVMET_HAVE_COMPLEX) && defined(TVMET_HAVE_COMPLEX_MATH2)


} // namespace tvmet

#endif // TVMET_MATRIX_BINARY_FUNCTIONS_H

// Local Variables:
// mode:C++
// End:
