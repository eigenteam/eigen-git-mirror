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
 * $Id: VectorBinaryFunctions.h,v 1.8 2004/06/10 16:36:55 opetzold Exp $
 */

#ifndef TVMET_XPR_VECTOR_BINARY_FUNCTIONS_H
#define TVMET_XPR_VECTOR_BINARY_FUNCTIONS_H

namespace tvmet {


/*********************************************************
 * PART I: DECLARATION
 *********************************************************/


/*
 * binary_function(XprVector<E1, Sz>, XprVector<E2, Sz>)
 */
#define TVMET_DECLARE_MACRO(NAME)					\
template<class E1, class E2, std::size_t Sz>				\
inline									\
XprVector<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E1::value_type, typename E2::value_type>,	\
    XprVector<E1, Sz>,							\
    XprVector<E2, Sz>							\
  >,									\
  Sz									\
>									\
NAME(const XprVector<E1, Sz>& lhs, 					\
     const XprVector<E2, Sz>& rhs) TVMET_CXX_ALWAYS_INLINE;

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
 * binary_function(XprVector<E, Sz>, POD)
 */
#define TVMET_DECLARE_MACRO(NAME, TP)		\
template<class E, std::size_t Sz>		\
inline						\
XprVector<					\
  XprBinOp<					\
    Fcnl_##NAME<typename E::value_type, TP >,	\
    XprVector<E, Sz>,				\
    XprLiteral< TP >				\
  >,						\
  Sz						\
>						\
NAME(const XprVector<E, Sz>& lhs, 		\
     TP rhs) TVMET_CXX_ALWAYS_INLINE;

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


#if defined(TVMET_HAVE_COMPLEX)
/*
 * binary_function(XprVector<E, Sz>, std::complex<>)
 */
#define TVMET_DECLARE_MACRO(NAME)				\
template<class E, std::size_t Sz, class T>			\
inline								\
XprVector<							\
  XprBinOp<							\
    Fcnl_##NAME<typename E::value_type, std::complex<T> >,	\
    XprVector<E, Sz>,						\
    XprLiteral< std::complex<T> >				\
  >,								\
  Sz								\
>								\
NAME(const XprVector<E, Sz>& lhs, 				\
     const std::complex<T>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(atan2)
TVMET_DECLARE_MACRO(drem)
TVMET_DECLARE_MACRO(fmod)
TVMET_DECLARE_MACRO(hypot)
TVMET_DECLARE_MACRO(jn)
TVMET_DECLARE_MACRO(yn)
TVMET_DECLARE_MACRO(pow)

#undef TVMET_DECLARE_MACRO

#endif // defined(TVMET_HAVE_COMPLEX)


/*********************************************************
 * PART II: IMPLEMENTATION
 *********************************************************/


/*
 * binary_function(XprVector<E1, Sz>, XprVector<E2, Sz>)
 */
#define TVMET_IMPLEMENT_MACRO(NAME)					\
template<class E1, class E2, std::size_t Sz>				\
inline									\
XprVector<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E1::value_type, typename E2::value_type>,	\
    XprVector<E1, Sz>,							\
    XprVector<E2, Sz>							\
  >,									\
  Sz									\
>									\
NAME(const XprVector<E1, Sz>& lhs, const XprVector<E2, Sz>& rhs) {	\
  typedef XprBinOp<							\
    Fcnl_##NAME<typename E1::value_type, typename E2::value_type>,	\
    XprVector<E1, Sz>,							\
    XprVector<E2, Sz>							\
  >		    					expr_type;	\
  return XprVector<expr_type, Sz>(					\
    expr_type(lhs, rhs));						\
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
 * binary_function(XprVector<E, Sz>, POD)
 */
#define TVMET_IMPLEMENT_MACRO(NAME, TP)					\
template<class E, std::size_t Sz>					\
inline									\
XprVector<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, TP >,				\
    XprVector<E, Sz>,							\
    XprLiteral< TP >							\
  >,									\
  Sz									\
>									\
NAME(const XprVector<E, Sz>& lhs, TP rhs) {				\
  typedef XprBinOp<							\
    Fcnl_##NAME<typename E::value_type, TP >,				\
    XprVector<E, Sz>,							\
    XprLiteral< TP >							\
  >							expr_type;	\
  return XprVector<expr_type, Sz>(					\
    expr_type(lhs, XprLiteral< TP >(rhs)));				\
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


#if defined(TVMET_HAVE_COMPLEX)
/*
 * binary_function(XprVector<E, Sz>, std::complex<>)
 */
#define TVMET_IMPLEMENT_MACRO(NAME)					\
template<class E, std::size_t Sz, class T>				\
inline									\
XprVector<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, std::complex<T> >,		\
    XprVector<E, Sz>,							\
    XprLiteral< std::complex<T> >					\
  >,									\
  Sz									\
>									\
NAME(const XprVector<E, Sz>& lhs, const std::complex<T>& rhs) {		\
  typedef XprBinOp<							\
    Fcnl_##NAME<typename E::value_type, std::complex<T> >,		\
    XprVector<E, Sz>,							\
    XprLiteral< std::complex<T> >					\
  >							expr_type;	\
  return XprVector<expr_type, Sz>(					\
    expr_type(lhs, XprLiteral< std::complex<T> >(rhs)));		\
}

TVMET_IMPLEMENT_MACRO(atan2)
TVMET_IMPLEMENT_MACRO(drem)
TVMET_IMPLEMENT_MACRO(fmod)
TVMET_IMPLEMENT_MACRO(hypot)
TVMET_IMPLEMENT_MACRO(jn)
TVMET_IMPLEMENT_MACRO(yn)
TVMET_IMPLEMENT_MACRO(pow)

#undef TVMET_IMPLEMENT_MACRO

#endif // defined(TVMET_HAVE_COMPLEX)


} // namespace tvmet

#endif // TVMET_XPR_VECTOR_BINARY_FUNCTIONS_H

// Local Variables:
// mode:C++
// End:
