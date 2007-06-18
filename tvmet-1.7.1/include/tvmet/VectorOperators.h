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
 * $Id: VectorOperators.h,v 1.14 2004/06/10 16:36:55 opetzold Exp $
 */

#ifndef TVMET_VECTOR_OPERATORS_H
#define TVMET_VECTOR_OPERATORS_H

namespace tvmet {


/*********************************************************
 * PART I: DECLARATION
 *********************************************************/


template<class T, int Sz>
inline
std::ostream& operator<<(std::ostream& os,
			 const Vector<T, Sz>& rhs) _tvmet_always_inline;


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Member operators (arithmetic and bit ops)
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * update_operator(Vector<T1, Sz>,  Vector<T2, Sz>)
 * update_operator(Vector<T1, Sz>,  XprVector<E, Sz>)
 * Note: per se element wise
 */
#define TVMET_DECLARE_MACRO(NAME, OP)					\
template<class T1, class T2, int Sz>				\
Vector<T1, Sz>&								\
operator OP (Vector<T1, Sz>& lhs,					\
	     const Vector<T2, Sz>& rhs) _tvmet_always_inline;	\
									\
template<class T, class E, int Sz>				\
Vector<T, Sz>&								\
operator OP (Vector<T, Sz>& lhs,					\
	     const XprVector<E, Sz>& rhs) _tvmet_always_inline;

TVMET_DECLARE_MACRO(add_eq, +=)		// per se element wise
TVMET_DECLARE_MACRO(sub_eq, -=)		// per se element wise
TVMET_DECLARE_MACRO(mul_eq, *=)		// per se element wise
namespace element_wise {
  TVMET_DECLARE_MACRO(div_eq, /=)		// not defined for vectors
}

#undef TVMET_DECLARE_MACRO


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Vector arithmetic operators implemented by functions
 * add, sub, mul and div
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * operator(Vector<T1, Sz>, Vector<T2, Sz>)
 * operator(Vector<T1, Sz>, XprVector<E, Sz>)
 * operator(XprVector<E, Sz>, Vector<T1, Sz>)
 */
#define TVMET_DECLARE_MACRO(NAME, OP)					\
template<class T1, class T2, int Sz>				\
XprVector<								\
  XprBinOp<								\
    Fcnl_##NAME<T1, T2>,						\
    VectorConstReference<T1, Sz>,					\
    VectorConstReference<T2, Sz>					\
  >,									\
  Sz									\
>									\
operator OP (const Vector<T1, Sz>& lhs, 				\
	     const Vector<T2, Sz>& rhs) _tvmet_always_inline;	\
									\
template<class E, class T, int Sz>				\
XprVector<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, T>,				\
    XprVector<E, Sz>,							\
    VectorConstReference<T, Sz>						\
  >,									\
  Sz									\
>									\
operator OP (const XprVector<E, Sz>& lhs,				\
	     const Vector<T, Sz>& rhs) _tvmet_always_inline;		\
									\
template<class E, class T, int Sz>				\
XprVector<								\
  XprBinOp<								\
    Fcnl_##NAME<T, typename E::value_type>,				\
    VectorConstReference<T, Sz>,					\
    XprVector<E, Sz>							\
  >,									\
  Sz									\
>									\
operator OP (const Vector<T, Sz>& lhs, 					\
	     const XprVector<E, Sz>& rhs) _tvmet_always_inline;

TVMET_DECLARE_MACRO(add, +)		// per se element wise
TVMET_DECLARE_MACRO(sub, -)		// per se element wise
TVMET_DECLARE_MACRO(mul, *)		// per se element wise
namespace element_wise {
  TVMET_DECLARE_MACRO(div, /)		// not defined for vectors
}

#undef TVMET_DECLARE_MACRO


/*
 * operator(Vector<T, Sz>, POD)
 * operator(POD, Vector<T, Sz>)
 * Note: operations +,-,*,/ are per se element wise
 */
#define TVMET_DECLARE_MACRO(NAME, OP, POD)			\
template<class T, int Sz>				\
XprVector<							\
  XprBinOp<							\
    Fcnl_##NAME< T, POD >,					\
    VectorConstReference<T, Sz>,				\
    XprLiteral< POD >						\
  >,								\
  Sz								\
>								\
operator OP (const Vector<T, Sz>& lhs, 				\
	     POD rhs) _tvmet_always_inline;			\
								\
template<class T, int Sz>				\
XprVector<							\
  XprBinOp<							\
    Fcnl_##NAME< POD, T>,					\
    XprLiteral< POD >,						\
    VectorConstReference<T, Sz>					\
  >,								\
  Sz								\
>								\
operator OP (POD lhs, 						\
	     const Vector<T, Sz>& rhs) _tvmet_always_inline;

TVMET_DECLARE_MACRO(add, +, int)
TVMET_DECLARE_MACRO(sub, -, int)
TVMET_DECLARE_MACRO(mul, *, int)
TVMET_DECLARE_MACRO(div, /, int)

TVMET_DECLARE_MACRO(add, +, float)
TVMET_DECLARE_MACRO(sub, -, float)
TVMET_DECLARE_MACRO(mul, *, float)
TVMET_DECLARE_MACRO(div, /, float)

TVMET_DECLARE_MACRO(add, +, double)
TVMET_DECLARE_MACRO(sub, -, double)
TVMET_DECLARE_MACRO(mul, *, double)
TVMET_DECLARE_MACRO(div, /, double)

#undef TVMET_DECLARE_MACRO


#if defined(EIGEN_USE_COMPLEX)
/*
 * operator(Vector<std::complex<T>, Sz>, std::complex<T>)
 * operator(std::complex<T>, Vector<std::complex<T>, Sz>)
 * Note: operations +,-,*,/ are per se element wise
 * \todo type promotion
 */
#define TVMET_DECLARE_MACRO(NAME, OP)						\
template<class T, int Sz>						\
XprVector<									\
  XprBinOp<									\
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,				\
    VectorConstReference< std::complex<T>, Sz>,					\
    XprLiteral< std::complex<T> >						\
  >,										\
  Sz										\
>										\
operator OP (const Vector<std::complex<T>, Sz>& lhs, 				\
	     const std::complex<T>& rhs) _tvmet_always_inline;		\
										\
template<class T, int Sz>						\
XprVector<									\
  XprBinOp<									\
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,				\
    XprLiteral< std::complex<T> >,						\
    VectorConstReference< std::complex<T>, Sz>					\
  >,										\
  Sz										\
>										\
operator OP (const std::complex<T>& lhs, 					\
	     const Vector< std::complex<T>, Sz>& rhs) _tvmet_always_inline;

TVMET_DECLARE_MACRO(add, +)		// per se element wise
TVMET_DECLARE_MACRO(sub, -)		// per se element wise
TVMET_DECLARE_MACRO(mul, *)		// per se element wise
TVMET_DECLARE_MACRO(div, /)		// per se element wise
#undef TVMET_DECLARE_MACRO

#endif // defined(EIGEN_USE_COMPLEX)



/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * global unary operators
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * unary_operator(Vector<T, Sz>)
 * Note: per se element wise
 */
#define TVMET_DECLARE_MACRO(NAME, OP)				\
template <class T, int Sz>				\
XprVector<							\
  XprUnOp<							\
    Fcnl_##NAME<T>,						\
    VectorConstReference<T, Sz>					\
  >,								\
  Sz								\
>								\
operator OP (const Vector<T, Sz>& rhs) _tvmet_always_inline;

TVMET_DECLARE_MACRO(neg, -)
#undef TVMET_DECLARE_MACRO


/*********************************************************
 * PART II: IMPLEMENTATION
 *********************************************************/


/**
 * \fn operator<<(std::ostream& os, const Vector<T, Sz>& rhs)
 * \brief Overload operator for i/o
 * \ingroup _binary_operator
 */
template<class T, int Sz>
inline
std::ostream& operator<<(std::ostream& os, const Vector<T, Sz>& rhs) {
  return rhs.print_on(os);
}


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Member operators (arithmetic and bit ops)
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * update_operator(Vector<T1, Sz>,  Vector<T2, Sz>)
 * update_operator(Vector<T1, Sz>,  XprVector<E, Sz>)
 * Note: per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)				\
template<class T1, class T2, int Sz>			\
inline Vector<T1, Sz>&						\
operator OP (Vector<T1, Sz>& lhs, const Vector<T2, Sz>& rhs) {	\
  return lhs.M_##NAME(rhs);					\
}								\
								\
template<class T, class E, int Sz>			\
inline Vector<T, Sz>&						\
operator OP (Vector<T, Sz>& lhs, const XprVector<E, Sz>& rhs) {	\
  return lhs.M_##NAME(rhs);					\
}

TVMET_IMPLEMENT_MACRO(add_eq, +=)		// per se element wise
TVMET_IMPLEMENT_MACRO(sub_eq, -=)		// per se element wise
TVMET_IMPLEMENT_MACRO(mul_eq, *=)		// per se element wise
namespace element_wise {
  TVMET_IMPLEMENT_MACRO(div_eq, /=)		// not defined for vectors
}

#undef TVMET_IMPLEMENT_MACRO


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Vector arithmetic operators implemented by functions
 * add, sub, mul and div
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * operator(Vector<T1, Sz>, Vector<T2, Sz>)
 * operator(Vector<T1, Sz>, XprVector<E, Sz>)
 * operator(XprVector<E, Sz>, Vector<T1, Sz>)
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)					\
template<class T1, class T2, int Sz>				\
inline									\
XprVector<								\
  XprBinOp<								\
    Fcnl_##NAME<T1, T2>,						\
    VectorConstReference<T1, Sz>,					\
    VectorConstReference<T2, Sz>					\
  >,									\
  Sz									\
>									\
operator OP (const Vector<T1, Sz>& lhs, const Vector<T2, Sz>& rhs) {	\
  return NAME (lhs, rhs);						\
}									\
									\
template<class E, class T, int Sz>				\
inline									\
XprVector<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, T>,				\
    XprVector<E, Sz>,							\
    VectorConstReference<T, Sz>						\
  >,									\
  Sz									\
>									\
operator OP (const XprVector<E, Sz>& lhs, const Vector<T, Sz>& rhs) {	\
  return NAME (lhs, rhs);						\
}									\
									\
template<class E, class T, int Sz>				\
inline									\
XprVector<								\
  XprBinOp<								\
    Fcnl_##NAME<T, typename E::value_type>,				\
    VectorConstReference<T, Sz>,					\
    XprVector<E, Sz>							\
  >,									\
  Sz									\
>									\
operator OP (const Vector<T, Sz>& lhs, const XprVector<E, Sz>& rhs) {	\
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
 * operator(Vector<T, Sz>, POD)
 * operator(POD, Vector<T, Sz>)
 * Note: operations +,-,*,/ are per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP, POD)		\
template<class T, int Sz>			\
inline							\
XprVector<						\
  XprBinOp<						\
    Fcnl_##NAME< T, POD >,				\
    VectorConstReference<T, Sz>,			\
    XprLiteral< POD >					\
  >,							\
  Sz							\
>							\
operator OP (const Vector<T, Sz>& lhs, POD rhs) {	\
  return NAME (lhs, rhs);				\
}							\
							\
template<class T, int Sz>			\
inline							\
XprVector<						\
  XprBinOp<						\
    Fcnl_##NAME< POD, T>,				\
    XprLiteral< POD >,					\
    VectorConstReference<T, Sz>				\
  >,							\
  Sz							\
>							\
operator OP (POD lhs, const Vector<T, Sz>& rhs) {	\
  return NAME (lhs, rhs);				\
}

TVMET_IMPLEMENT_MACRO(add, +, int)
TVMET_IMPLEMENT_MACRO(sub, -, int)
TVMET_IMPLEMENT_MACRO(mul, *, int)
TVMET_IMPLEMENT_MACRO(div, /, int)

TVMET_IMPLEMENT_MACRO(add, +, float)
TVMET_IMPLEMENT_MACRO(sub, -, float)
TVMET_IMPLEMENT_MACRO(mul, *, float)
TVMET_IMPLEMENT_MACRO(div, /, float)

TVMET_IMPLEMENT_MACRO(add, +, double)
TVMET_IMPLEMENT_MACRO(sub, -, double)
TVMET_IMPLEMENT_MACRO(mul, *, double)
TVMET_IMPLEMENT_MACRO(div, /, double)

#undef TVMET_IMPLEMENT_MACRO


#if defined(EIGEN_USE_COMPLEX)
/*
 * operator(Vector<std::complex<T>, Sz>, std::complex<T>)
 * operator(std::complex<T>, Vector<std::complex<T>, Sz>)
 * Note: operations +,-,*,/ are per se element wise
 * \todo type promotion
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)			\
template<class T, int Sz>			\
inline							\
XprVector<						\
  XprBinOp<						\
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,	\
    VectorConstReference< std::complex<T>, Sz>,		\
    XprLiteral< std::complex<T> >			\
  >,							\
  Sz							\
>							\
operator OP (const Vector<std::complex<T>, Sz>& lhs, 	\
	     const std::complex<T>& rhs) {		\
  return NAME (lhs, rhs);				\
}							\
							\
template<class T, int Sz>			\
inline							\
XprVector<						\
  XprBinOp<						\
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,	\
    XprLiteral< std::complex<T> >,			\
    VectorConstReference< std::complex<T>, Sz>		\
  >,							\
  Sz							\
>							\
operator OP (const std::complex<T>& lhs, 		\
	     const Vector< std::complex<T>, Sz>& rhs) {	\
  return NAME (lhs, rhs);				\
}

TVMET_IMPLEMENT_MACRO(add, +)		// per se element wise
TVMET_IMPLEMENT_MACRO(sub, -)		// per se element wise
TVMET_IMPLEMENT_MACRO(mul, *)		// per se element wise
TVMET_IMPLEMENT_MACRO(div, /)		// per se element wise

#undef TVMET_IMPLEMENT_MACRO

#endif // defined(EIGEN_USE_COMPLEX)


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * global unary operators
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * unary_operator(Vector<T, Sz>)
 * Note: per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)					\
template <class T, int Sz>					\
inline									\
XprVector<								\
  XprUnOp<								\
    Fcnl_##NAME<T>,							\
    VectorConstReference<T, Sz>						\
  >,									\
  Sz									\
>									\
operator OP (const Vector<T, Sz>& rhs) {				\
  typedef XprUnOp<							\
    Fcnl_##NAME<T>,							\
    VectorConstReference<T, Sz>						\
  >  							 expr_type;	\
  return XprVector<expr_type, Sz>(expr_type(rhs.const_ref()));		\
}

TVMET_IMPLEMENT_MACRO(neg, -)

#undef TVMET_IMPLEMENT_MACRO


} // namespace tvmet

#endif // TVMET_VECTOR_OPERATORS_H

// Local Variables:
// mode:C++
// End:
