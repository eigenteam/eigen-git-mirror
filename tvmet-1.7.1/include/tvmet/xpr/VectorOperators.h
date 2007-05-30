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
template<class E, std::size_t Sz>					\
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
template<class E, std::size_t Sz>					\
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
operator OP (const XprVector<E, Sz>& lhs,				\
	     const std::complex<T>& rhs) TVMET_CXX_ALWAYS_INLINE;	\
									\
template<class E, std::size_t Sz, class T>				\
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
 * Vector integer and compare operators
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * operator(XprVector, XprVector)
 */
#define TVMET_DECLARE_MACRO(NAME, OP)					\
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
operator OP (const XprVector<E1, Sz>& lhs,				\
	     const XprVector<E2, Sz>& rhs) TVMET_CXX_ALWAYS_INLINE;

// integer operators only, e.g used on double you wil get an error
namespace element_wise {
  TVMET_DECLARE_MACRO(mod, %)
  TVMET_DECLARE_MACRO(bitxor, ^)
  TVMET_DECLARE_MACRO(bitand, &)
  TVMET_DECLARE_MACRO(bitor, |)
  TVMET_DECLARE_MACRO(shl, <<)
  TVMET_DECLARE_MACRO(shr, >>)
}

// necessary operators for eval functions
TVMET_DECLARE_MACRO(greater, >)
TVMET_DECLARE_MACRO(less, <)
TVMET_DECLARE_MACRO(greater_eq, >=)
TVMET_DECLARE_MACRO(less_eq, <=)
TVMET_DECLARE_MACRO(eq, ==)
TVMET_DECLARE_MACRO(not_eq, !=)
TVMET_DECLARE_MACRO(and, &&)
TVMET_DECLARE_MACRO(or, ||)

#undef TVMET_DECLARE_MACRO


/*
 * operator(Vector<T, Sz>, POD)
 * operator(POD, Vector<T, Sz>)
 * Note: operations are per se element_wise
 */
#define TVMET_DECLARE_MACRO(NAME, OP, TP)				\
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
operator OP (const XprVector<E, Sz>& lhs, 				\
	     TP rhs) TVMET_CXX_ALWAYS_INLINE;				\
									\
template<class E, std::size_t Sz>					\
inline									\
XprVector<								\
  XprBinOp<								\
    Fcnl_##NAME<TP, typename E::value_type>,				\
    XprLiteral< TP >,							\
    XprVector<E, Sz>							\
  >,									\
  Sz									\
>									\
operator OP (TP lhs, 							\
	     const XprVector<E, Sz>& rhs) TVMET_CXX_ALWAYS_INLINE;

// integer operators only, e.g used on double you wil get an error
namespace element_wise {
  TVMET_DECLARE_MACRO(mod, %, int)
  TVMET_DECLARE_MACRO(bitxor, ^, int)
  TVMET_DECLARE_MACRO(bitand, &, int)
  TVMET_DECLARE_MACRO(bitor, |, int)
  TVMET_DECLARE_MACRO(shl, <<, int)
  TVMET_DECLARE_MACRO(shr, >>, int)
}

// necessary operators for eval functions
TVMET_DECLARE_MACRO(greater, >, int)
TVMET_DECLARE_MACRO(less, <, int)
TVMET_DECLARE_MACRO(greater_eq, >=, int)
TVMET_DECLARE_MACRO(less_eq, <=, int)
TVMET_DECLARE_MACRO(eq, ==, int)
TVMET_DECLARE_MACRO(not_eq, !=, int)
TVMET_DECLARE_MACRO(and, &&, int)
TVMET_DECLARE_MACRO(or, ||, int)


#if defined(TVMET_HAVE_LONG_LONG)
// integer operators only
namespace element_wise {
  TVMET_DECLARE_MACRO(mod, %, long long int)
  TVMET_DECLARE_MACRO(bitxor, ^, long long int)
  TVMET_DECLARE_MACRO(bitand, &, long long int)
  TVMET_DECLARE_MACRO(bitor, |, long long int)
  TVMET_DECLARE_MACRO(shl, <<, long long int)
  TVMET_DECLARE_MACRO(shr, >>, long long int)
}

// necessary operators for eval functions
TVMET_DECLARE_MACRO(greater, >, long long int)
TVMET_DECLARE_MACRO(less, <, long long int)
TVMET_DECLARE_MACRO(greater_eq, >=, long long int)
TVMET_DECLARE_MACRO(less_eq, <=, long long int)
TVMET_DECLARE_MACRO(eq, ==, long long int)
TVMET_DECLARE_MACRO(not_eq, !=, long long int)
TVMET_DECLARE_MACRO(and, &&, long long int)
TVMET_DECLARE_MACRO(or, ||, long long int)
#endif // defined(TVMET_HAVE_LONG_LONG)

// necessary operators for eval functions
TVMET_DECLARE_MACRO(greater, >, float)
TVMET_DECLARE_MACRO(less, <, float)
TVMET_DECLARE_MACRO(greater_eq, >=, float)
TVMET_DECLARE_MACRO(less_eq, <=, float)
TVMET_DECLARE_MACRO(eq, ==, float)
TVMET_DECLARE_MACRO(not_eq, !=, float)

// necessary operators for eval functions
TVMET_DECLARE_MACRO(greater, >, double)
TVMET_DECLARE_MACRO(less, <, double)
TVMET_DECLARE_MACRO(greater_eq, >=, double)
TVMET_DECLARE_MACRO(less_eq, <=, double)
TVMET_DECLARE_MACRO(eq, ==, double)
TVMET_DECLARE_MACRO(not_eq, !=, double)

#if defined(TVMET_HAVE_LONG_DOUBLE)
// necessary operators for eval functions
TVMET_DECLARE_MACRO(greater, >, long double)
TVMET_DECLARE_MACRO(less, <, long double)
TVMET_DECLARE_MACRO(greater_eq, >=, long double)
TVMET_DECLARE_MACRO(less_eq, <=, long double)
TVMET_DECLARE_MACRO(eq, ==, long double)
TVMET_DECLARE_MACRO(not_eq, !=, long double)
#endif // defined(TVMET_HAVE_LONG_DOUBLE)

#undef TVMET_DECLARE_MACRO


#if defined(TVMET_HAVE_COMPLEX)
/*
 * operator(Vector<std::complex<T>, Sz>, std::complex<T>)
 * operator(std::complex<T>, Vector<std::complex<T>, Sz>)
 * Note: - per se element wise
 *       - bit ops on complex<int> doesn't make sense, stay away
 * \todo type promotion
 */
#define TVMET_DECLARE_MACRO(NAME, OP)					\
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
operator OP (const XprVector<E, Sz>& lhs,				\
	     const std::complex<T>& rhs) TVMET_CXX_ALWAYS_INLINE;	\
									\
template<class E, std::size_t Sz, class T>				\
inline									\
XprVector<								\
  XprBinOp<								\
    Fcnl_##NAME<std::complex<T>, typename E::value_type>,		\
    XprLiteral< std::complex<T> >,					\
    XprVector<E, Sz>							\
  >,									\
  Sz									\
>									\
operator OP (const std::complex<T>& lhs,				\
	     const XprVector<E, Sz>& rhs) TVMET_CXX_ALWAYS_INLINE;

// necessary operators for eval functions
TVMET_DECLARE_MACRO(greater, >)
TVMET_DECLARE_MACRO(less, <)
TVMET_DECLARE_MACRO(greater_eq, >=)
TVMET_DECLARE_MACRO(less_eq, <=)
TVMET_DECLARE_MACRO(eq, ==)
TVMET_DECLARE_MACRO(not_eq, !=)
TVMET_DECLARE_MACRO(and, &&)
TVMET_DECLARE_MACRO(or, ||)

#undef TVMET_DECLARE_MACRO

#endif // defined(TVMET_HAVE_COMPLEX)


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * global unary operators
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * Unary Operator on XprVector<E, Sz>
 */
#define TVMET_DECLARE_MACRO(NAME, OP)					\
template <class E, std::size_t Sz>					\
inline									\
XprVector<								\
  XprUnOp<								\
    Fcnl_##NAME<typename E::value_type>,				\
    XprVector<E, Sz>							\
  >,									\
  Sz									\
>									\
operator OP (const XprVector<E, Sz>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(not, !)
TVMET_DECLARE_MACRO(compl, ~)
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
template<class E, std::size_t Sz>			\
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
template<class E, std::size_t Sz>			\
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
operator OP (const XprVector<E, Sz>& lhs,			\
	     const std::complex<T>& rhs) {			\
  return NAME (lhs, rhs);					\
}								\
								\
template<class E, std::size_t Sz, class T>			\
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
 * Vector integer and compare operators
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * operator(XprVector, XprVector)
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)					\
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
operator OP (const XprVector<E1, Sz>& lhs,				\
	     const XprVector<E2, Sz>& rhs) {				\
  typedef XprBinOp<							\
    Fcnl_##NAME<typename E1::value_type, typename E2::value_type>,	\
    XprVector<E1, Sz>,							\
    XprVector<E2, Sz>							\
  >		    					expr_type;	\
  return XprVector<expr_type, Sz>(expr_type(lhs, rhs));			\
}

// integer operators only, e.g used on double you wil get an error
namespace element_wise {
  TVMET_IMPLEMENT_MACRO(mod, %)
  TVMET_IMPLEMENT_MACRO(bitxor, ^)
  TVMET_IMPLEMENT_MACRO(bitand, &)
  TVMET_IMPLEMENT_MACRO(bitor, |)
  TVMET_IMPLEMENT_MACRO(shl, <<)
  TVMET_IMPLEMENT_MACRO(shr, >>)
}

// necessary operators for eval functions
TVMET_IMPLEMENT_MACRO(greater, >)
TVMET_IMPLEMENT_MACRO(less, <)
TVMET_IMPLEMENT_MACRO(greater_eq, >=)
TVMET_IMPLEMENT_MACRO(less_eq, <=)
TVMET_IMPLEMENT_MACRO(eq, ==)
TVMET_IMPLEMENT_MACRO(not_eq, !=)
TVMET_IMPLEMENT_MACRO(and, &&)
TVMET_IMPLEMENT_MACRO(or, ||)

#undef TVMET_IMPLEMENT_MACRO


/*
 * operator(Vector<T, Sz>, POD)
 * operator(POD, Vector<T, Sz>)
 * Note: operations are per se element_wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP, TP)				\
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
operator OP (const XprVector<E, Sz>& lhs, TP rhs) {			\
  typedef XprBinOp<							\
    Fcnl_##NAME<typename E::value_type, TP >,				\
    XprVector<E, Sz>,							\
    XprLiteral< TP >							\
  >							expr_type;	\
  return XprVector<expr_type, Sz>(					\
    expr_type(lhs, XprLiteral< TP >(rhs)));				\
}									\
									\
template<class E, std::size_t Sz>					\
inline									\
XprVector<								\
  XprBinOp<								\
    Fcnl_##NAME<TP, typename E::value_type>,				\
    XprLiteral< TP >,							\
    XprVector<E, Sz>							\
  >,									\
  Sz									\
>									\
operator OP (TP lhs, const XprVector<E, Sz>& rhs) {			\
  typedef XprBinOp<							\
    Fcnl_##NAME< TP, typename E::value_type>,				\
    XprLiteral< TP >,							\
    XprVector<E, Sz>							\
  >							expr_type;	\
  return XprVector<expr_type, Sz>(					\
    expr_type(XprLiteral< TP >(lhs), rhs));				\
}

// integer operators only, e.g used on double you wil get an error
namespace element_wise {
  TVMET_IMPLEMENT_MACRO(mod, %, int)
  TVMET_IMPLEMENT_MACRO(bitxor, ^, int)
  TVMET_IMPLEMENT_MACRO(bitand, &, int)
  TVMET_IMPLEMENT_MACRO(bitor, |, int)
  TVMET_IMPLEMENT_MACRO(shl, <<, int)
  TVMET_IMPLEMENT_MACRO(shr, >>, int)
}

// necessary operators for eval functions
TVMET_IMPLEMENT_MACRO(greater, >, int)
TVMET_IMPLEMENT_MACRO(less, <, int)
TVMET_IMPLEMENT_MACRO(greater_eq, >=, int)
TVMET_IMPLEMENT_MACRO(less_eq, <=, int)
TVMET_IMPLEMENT_MACRO(eq, ==, int)
TVMET_IMPLEMENT_MACRO(not_eq, !=, int)
TVMET_IMPLEMENT_MACRO(and, &&, int)
TVMET_IMPLEMENT_MACRO(or, ||, int)


#if defined(TVMET_HAVE_LONG_LONG)
// integer operators only
namespace element_wise {
  TVMET_IMPLEMENT_MACRO(mod, %, long long int)
  TVMET_IMPLEMENT_MACRO(bitxor, ^, long long int)
  TVMET_IMPLEMENT_MACRO(bitand, &, long long int)
  TVMET_IMPLEMENT_MACRO(bitor, |, long long int)
  TVMET_IMPLEMENT_MACRO(shl, <<, long long int)
  TVMET_IMPLEMENT_MACRO(shr, >>, long long int)
}

// necessary operators for eval functions
TVMET_IMPLEMENT_MACRO(greater, >, long long int)
TVMET_IMPLEMENT_MACRO(less, <, long long int)
TVMET_IMPLEMENT_MACRO(greater_eq, >=, long long int)
TVMET_IMPLEMENT_MACRO(less_eq, <=, long long int)
TVMET_IMPLEMENT_MACRO(eq, ==, long long int)
TVMET_IMPLEMENT_MACRO(not_eq, !=, long long int)
TVMET_IMPLEMENT_MACRO(and, &&, long long int)
TVMET_IMPLEMENT_MACRO(or, ||, long long int)
#endif // defined(TVMET_HAVE_LONG_LONG)

// necessary operators for eval functions
TVMET_IMPLEMENT_MACRO(greater, >, float)
TVMET_IMPLEMENT_MACRO(less, <, float)
TVMET_IMPLEMENT_MACRO(greater_eq, >=, float)
TVMET_IMPLEMENT_MACRO(less_eq, <=, float)
TVMET_IMPLEMENT_MACRO(eq, ==, float)
TVMET_IMPLEMENT_MACRO(not_eq, !=, float)

// necessary operators for eval functions
TVMET_IMPLEMENT_MACRO(greater, >, double)
TVMET_IMPLEMENT_MACRO(less, <, double)
TVMET_IMPLEMENT_MACRO(greater_eq, >=, double)
TVMET_IMPLEMENT_MACRO(less_eq, <=, double)
TVMET_IMPLEMENT_MACRO(eq, ==, double)
TVMET_IMPLEMENT_MACRO(not_eq, !=, double)

#if defined(TVMET_HAVE_LONG_DOUBLE)
// necessary operators for eval functions
TVMET_IMPLEMENT_MACRO(greater, >, long double)
TVMET_IMPLEMENT_MACRO(less, <, long double)
TVMET_IMPLEMENT_MACRO(greater_eq, >=, long double)
TVMET_IMPLEMENT_MACRO(less_eq, <=, long double)
TVMET_IMPLEMENT_MACRO(eq, ==, long double)
TVMET_IMPLEMENT_MACRO(not_eq, !=, long double)
#endif // defined(TVMET_HAVE_LONG_DOUBLE)

#undef TVMET_IMPLEMENT_MACRO


#if defined(TVMET_HAVE_COMPLEX)
/*
 * operator(Vector<std::complex<T>, Sz>, std::complex<T>)
 * operator(std::complex<T>, Vector<std::complex<T>, Sz>)
 * Note: - per se element wise
 *       - bit ops on complex<int> doesn't make sense, stay away
 * \todo type promotion
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)					\
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
operator OP (const XprVector<E, Sz>& lhs,				\
	     const std::complex<T>& rhs) {				\
  typedef XprBinOp<							\
    Fcnl_##NAME<typename E::value_type, std::complex<T> >,		\
    XprVector<E, Sz>,							\
    XprLiteral< std::complex<T> >					\
  >							expr_type;	\
  return XprVector<expr_type, Sz>(					\
    expr_type(lhs, XprLiteral< std::complex<T> >(rhs)));		\
}									\
									\
template<class E, std::size_t Sz, class T>				\
inline									\
XprVector<								\
  XprBinOp<								\
    Fcnl_##NAME<std::complex<T>, typename E::value_type>,		\
    XprLiteral< std::complex<T> >,					\
    XprVector<E, Sz>							\
  >,									\
  Sz									\
>									\
operator OP (const std::complex<T>& lhs,				\
	     const XprVector<E, Sz>& rhs) {				\
  typedef XprBinOp<							\
    Fcnl_##NAME< std::complex<T>, typename E::value_type>,		\
    XprLiteral< std::complex<T> >,					\
    XprVector<E, Sz>							\
  >							expr_type;	\
  return XprVector<expr_type, Sz>(					\
    expr_type(XprLiteral< std::complex<T> >(lhs), rhs));		\
}

// necessary operators for eval functions
TVMET_IMPLEMENT_MACRO(greater, >)
TVMET_IMPLEMENT_MACRO(less, <)
TVMET_IMPLEMENT_MACRO(greater_eq, >=)
TVMET_IMPLEMENT_MACRO(less_eq, <=)
TVMET_IMPLEMENT_MACRO(eq, ==)
TVMET_IMPLEMENT_MACRO(not_eq, !=)
TVMET_IMPLEMENT_MACRO(and, &&)
TVMET_IMPLEMENT_MACRO(or, ||)

#undef TVMET_IMPLEMENT_MACRO

#endif // defined(TVMET_HAVE_COMPLEX)


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * global unary operators
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * Unary Operator on XprVector<E, Sz>
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)					\
template <class E, std::size_t Sz>					\
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

TVMET_IMPLEMENT_MACRO(not, !)
TVMET_IMPLEMENT_MACRO(compl, ~)
TVMET_IMPLEMENT_MACRO(neg, -)

#undef TVMET_IMPLEMENT_MACRO


} // namespace tvmet

#endif // TVMET_XPR_VECTOR_OPERATORS_H

// Local Variables:
// mode:C++
// End:
