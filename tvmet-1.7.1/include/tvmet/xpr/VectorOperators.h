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
 * $Id: VectorOperators.h,v 1.13 2004/06/10 16:36:55 opetzold Exp $
 */

#ifndef TVMET_XPR_VECTOR_OPERATORS_H
#define TVMET_XPR_VECTOR_OPERATORS_H

namespace tvmet {


/*********************************************************
 * PART I: DECLARATION
 *********************************************************/


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Vector arithmetic operators implemented by functions
 * add, sub, mul and div
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * operator(XprVector<E1, Sz>, XprVector<E2, Sz>)
 */
#define TVMET_DECLARE_MACRO(NAME, OP)					\
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
operator OP (const XprVector<E1, Sz>& lhs, 				\
	     const XprVector<E2, Sz>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(add, +)		// per se element wise
TVMET_DECLARE_MACRO(sub, -)		// per se element wise
TVMET_DECLARE_MACRO(mul, *)		// per se element wise
namespace element_wise {
  TVMET_DECLARE_MACRO(div, /)		// not defined for vectors
}

#undef TVMET_DECLARE_MACRO


/*
 * operator(XprVector<E, Sz>, POD)
 * operator(POD, XprVector<E, Sz>)
 * Note: operations +,-,*,/ are per se element wise
 */
#define TVMET_DECLARE_MACRO(NAME, OP, POD)				\
template<class E, int Sz>					\
inline									\
XprVector<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, POD >,				\
    XprVector<E, Sz>,							\
    XprLiteral< POD >							\
  >,									\
  Sz									\
>									\
operator OP (const XprVector<E, Sz>& lhs, 				\
	     POD rhs) TVMET_CXX_ALWAYS_INLINE;				\
									\
template<class E, int Sz>					\
inline									\
XprVector<								\
  XprBinOp<								\
    Fcnl_##NAME< POD, typename E::value_type >,				\
    XprLiteral< POD >,							\
    XprVector< E, Sz>							\
  >,									\
  Sz									\
>									\
operator OP (POD lhs, 							\
	     const XprVector<E, Sz>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(add, +, int)
TVMET_DECLARE_MACRO(sub, -, int)
TVMET_DECLARE_MACRO(mul, *, int)
TVMET_DECLARE_MACRO(div, /, int)

#if defined(TVMET_HAVE_LONG_LONG)
TVMET_DECLARE_MACRO(add, +, long long int)
TVMET_DECLARE_MACRO(sub, -, long long int)
TVMET_DECLARE_MACRO(mul, *, long long int)
TVMET_DECLARE_MACRO(div, /, long long int)
#endif

TVMET_DECLARE_MACRO(add, +, float)
TVMET_DECLARE_MACRO(sub, -, float)
TVMET_DECLARE_MACRO(mul, *, float)
TVMET_DECLARE_MACRO(div, /, float)

TVMET_DECLARE_MACRO(add, +, double)
TVMET_DECLARE_MACRO(sub, -, double)
TVMET_DECLARE_MACRO(mul, *, double)
TVMET_DECLARE_MACRO(div, /, double)

#if defined(TVMET_HAVE_LONG_DOUBLE)
TVMET_DECLARE_MACRO(add, +, long double)
TVMET_DECLARE_MACRO(sub, -, long double)
TVMET_DECLARE_MACRO(mul, *, long double)
TVMET_DECLARE_MACRO(div, /, long double)
#endif

#undef TVMET_DECLARE_MACRO


#if defined(TVMET_HAVE_COMPLEX)
/*
 * operator(XprVector<E, Sz>, complex<T>)
 * operator(complex<T>, XprVector<E, Sz>)
 * Note: operations +,-,*,/ are per se element wise
 */
#define TVMET_DECLARE_MACRO(NAME, OP)					\
template<class E, int Sz, class T>				\
inline									\
XprVector<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, std::complex<T> >,		\
    XprVector<E, Sz>,							\
    XprLiteral< std::complex<T> >					\
  >,									\
  Sz									\
>									\
operator OP (const XprVector<E, Sz>& lhs,				\
	     const std::complex<T>& rhs) TVMET_CXX_ALWAYS_INLINE;	\
									\
template<class E, int Sz, class T>				\
inline									\
XprVector<								\
  XprBinOp<								\
    Fcnl_##NAME< std::complex<T>, typename E::value_type >,		\
    XprLiteral< std::complex<T> >,					\
    XprVector< E, Sz>							\
  >,									\
  Sz									\
>									\
operator OP (const std::complex<T>& lhs,				\
	     const XprVector<E, Sz>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(add, +)		// per se element wise
TVMET_DECLARE_MACRO(sub, -)		// per se element wise
TVMET_DECLARE_MACRO(mul, *)		// per se element wise
TVMET_DECLARE_MACRO(div, /)		// per se element wise

#undef TVMET_DECLARE_MACRO

#endif // defined(TVMET_HAVE_COMPLEX)


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * global unary operators
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * Unary Operator on XprVector<E, Sz>
 */
#define TVMET_DECLARE_MACRO(NAME, OP)					\
template <class E, int Sz>					\
inline									\
XprVector<								\
  XprUnOp<								\
    Fcnl_##NAME<typename E::value_type>,				\
    XprVector<E, Sz>							\
  >,									\
  Sz									\
>									\
operator OP (const XprVector<E, Sz>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(neg, -)

#undef TVMET_DECLARE_MACRO


/*********************************************************
 * PART II: IMPLEMENTATION
 *********************************************************/


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Vector arithmetic operators implemented by functions
 * add, sub, mul and div
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * operator(XprVector<E1, Sz>, XprVector<E2, Sz>)
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)					\
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
operator OP (const XprVector<E1, Sz>& lhs, 				\
	     const XprVector<E2, Sz>& rhs) {				\
  return NAME (lhs, rhs);						\
}

TVMET_IMPLEMENT_MACRO(add, +)		// per se element wise
TVMET_IMPLEMENT_MACRO(sub, -)		// per se element wise
TVMET_IMPLEMENT_MACRO(mul, *)		// per se element wise
namespace element_wise {
  TVMET_IMPLEMENT_MACRO(div, /)		// not defined for vectors
}

#undef TVMET_IMPLEMENT_MACRO


/*
 * operator(XprVector<E, Sz>, POD)
 * operator(POD, XprVector<E, Sz>)
 * Note: operations +,-,*,/ are per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP, POD)		\
template<class E, int Sz>			\
inline							\
XprVector<						\
  XprBinOp<						\
    Fcnl_##NAME<typename E::value_type, POD >,		\
    XprVector<E, Sz>,					\
    XprLiteral< POD >					\
  >,							\
  Sz							\
>							\
operator OP (const XprVector<E, Sz>& lhs, POD rhs) {	\
  return NAME (lhs, rhs);				\
}							\
							\
template<class E, int Sz>			\
inline							\
XprVector<						\
  XprBinOp<						\
    Fcnl_##NAME< POD, typename E::value_type >,		\
    XprLiteral< POD >,					\
    XprVector< E, Sz>					\
  >,							\
  Sz							\
>							\
operator OP (POD lhs, const XprVector<E, Sz>& rhs) {	\
  return NAME (lhs, rhs);				\
}

TVMET_IMPLEMENT_MACRO(add, +, int)
TVMET_IMPLEMENT_MACRO(sub, -, int)
TVMET_IMPLEMENT_MACRO(mul, *, int)
TVMET_IMPLEMENT_MACRO(div, /, int)

#if defined(TVMET_HAVE_LONG_LONG)
TVMET_IMPLEMENT_MACRO(add, +, long long int)
TVMET_IMPLEMENT_MACRO(sub, -, long long int)
TVMET_IMPLEMENT_MACRO(mul, *, long long int)
TVMET_IMPLEMENT_MACRO(div, /, long long int)
#endif

TVMET_IMPLEMENT_MACRO(add, +, float)
TVMET_IMPLEMENT_MACRO(sub, -, float)
TVMET_IMPLEMENT_MACRO(mul, *, float)
TVMET_IMPLEMENT_MACRO(div, /, float)

TVMET_IMPLEMENT_MACRO(add, +, double)
TVMET_IMPLEMENT_MACRO(sub, -, double)
TVMET_IMPLEMENT_MACRO(mul, *, double)
TVMET_IMPLEMENT_MACRO(div, /, double)

#if defined(TVMET_HAVE_LONG_DOUBLE)
TVMET_IMPLEMENT_MACRO(add, +, long double)
TVMET_IMPLEMENT_MACRO(sub, -, long double)
TVMET_IMPLEMENT_MACRO(mul, *, long double)
TVMET_IMPLEMENT_MACRO(div, /, long double)
#endif

#undef TVMET_IMPLEMENT_MACRO


#if defined(TVMET_HAVE_COMPLEX)
/*
 * operator(XprVector<E, Sz>, complex<T>)
 * operator(complex<T>, XprVector<E, Sz>)
 * Note: operations +,-,*,/ are per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)				\
template<class E, int Sz, class T>			\
inline								\
XprVector<							\
  XprBinOp<							\
    Fcnl_##NAME<typename E::value_type, std::complex<T> >,	\
    XprVector<E, Sz>,						\
    XprLiteral< std::complex<T> >				\
  >,								\
  Sz								\
>								\
operator OP (const XprVector<E, Sz>& lhs,			\
	     const std::complex<T>& rhs) {			\
  return NAME (lhs, rhs);					\
}								\
								\
template<class E, int Sz, class T>			\
inline								\
XprVector<							\
  XprBinOp<							\
    Fcnl_##NAME< std::complex<T>, typename E::value_type >,	\
    XprLiteral< std::complex<T> >,				\
    XprVector< E, Sz>						\
  >,								\
  Sz								\
>								\
operator OP (const std::complex<T>& lhs,			\
	     const XprVector<E, Sz>& rhs) {			\
  return NAME (lhs, rhs);					\
}

TVMET_IMPLEMENT_MACRO(add, +)		// per se element wise
TVMET_IMPLEMENT_MACRO(sub, -)		// per se element wise
TVMET_IMPLEMENT_MACRO(mul, *)		// per se element wise
TVMET_IMPLEMENT_MACRO(div, /)		// per se element wise

#undef TVMET_IMPLEMENT_MACRO

#endif // defined(TVMET_HAVE_COMPLEX)


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * global unary operators
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * Unary Operator on XprVector<E, Sz>
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)					\
template <class E, int Sz>					\
inline									\
XprVector<								\
  XprUnOp<								\
    Fcnl_##NAME<typename E::value_type>,				\
    XprVector<E, Sz>							\
  >,									\
  Sz									\
>									\
operator OP (const XprVector<E, Sz>& rhs) {				\
  typedef XprUnOp<							\
    Fcnl_##NAME<typename E::value_type>,				\
    XprVector<E, Sz>							\
  >  							 expr_type;	\
  return XprVector<expr_type, Sz>(expr_type(rhs));			\
}

TVMET_IMPLEMENT_MACRO(neg, -)

#undef TVMET_IMPLEMENT_MACRO


} // namespace tvmet

#endif // TVMET_XPR_VECTOR_OPERATORS_H

// Local Variables:
// mode:C++
// End:
