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
 * $Id: MatrixOperators.h,v 1.19 2005/03/09 09:48:03 opetzold Exp $
 */

#ifndef TVMET_XPR_MATRIX_OPERATORS_H
#define TVMET_XPR_MATRIX_OPERATORS_H

namespace tvmet {


/*********************************************************
 * PART I: DECLARATION
 *********************************************************/


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Matrix arithmetic operators implemented by functions
 * add, sub, mul and div
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * operator(const XprMatrix<E1, Rows1, Cols1>& lhs, const XprMatrix<E2, Cols1,Cols2>& rhs)
 *
 * Note: operations +,-,*,/ are per se element wise. Further more,
 * element wise operations make sense only for matrices of the same
 * size [varg].
 */
#define TVMET_DECLARE_MACRO(NAME, OP)						\
template<class E1, int Rows1, int Cols1,			\
         class E2>								\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME<typename E1::value_type, typename E2::value_type>,		\
    XprMatrix<E1, Rows1, Cols1>,						\
    XprMatrix<E2, Rows1, Cols1>							\
  >,										\
  Rows1, Cols1									\
>										\
operator OP (const XprMatrix<E1, Rows1, Cols1>& lhs,				\
	     const XprMatrix<E2, Rows1, Cols1>& rhs) _tvmet_always_inline;

TVMET_DECLARE_MACRO(add, +)		// per se element wise
TVMET_DECLARE_MACRO(sub, -)		// per se element wise
namespace element_wise {
  TVMET_DECLARE_MACRO(mul, *)		// see as prod()
  TVMET_DECLARE_MACRO(div, /)		// not defined for matrizes, must be element_wise
}
#undef TVMET_DECLARE_MACRO


/*
 * operator(XprMatrix<E, Rows, Cols>,  POD)
 * operator(POD, XprMatrix<E, Rows, Cols>)
 * Note: operations +,-,*,/ are per se element wise
 */
#define TVMET_DECLARE_MACRO(NAME, OP, POD)					\
template<class E, int Rows, int Cols>				\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME<typename E::value_type, POD >,					\
    XprMatrix<E, Rows, Cols>,							\
    XprLiteral< POD >								\
  >,										\
  Rows, Cols									\
>										\
operator OP (const XprMatrix<E, Rows, Cols>& lhs, 				\
	     POD rhs) _tvmet_always_inline;					\
										\
template<class E,int Rows, int Cols>				\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME<POD, typename E::value_type>,					\
    XprLiteral< POD >,								\
    XprMatrix<E, Rows, Cols>							\
  >,										\
  Rows, Cols									\
>										\
operator OP (POD lhs, 								\
	     const XprMatrix<E, Rows, Cols>& rhs) _tvmet_always_inline;

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
 * operator(XprMatrix<E, Rows, Cols>, complex<>)
 * operator(complex<>, XprMatrix<E, Rows, Cols>)
 * Note: operations +,-,*,/ are per se element wise
 * \todo type promotion
 */
#define TVMET_DECLARE_MACRO(NAME, OP)						\
template<class E, int Rows, int Cols, class T>			\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME<typename E::value_type, std::complex<T> >,			\
    XprMatrix<E, Rows, Cols>,							\
    XprLiteral< std::complex<T> >						\
  >,										\
  Rows, Cols									\
>										\
operator OP (const XprMatrix<E, Rows, Cols>& lhs,				\
	     const std::complex<T>& rhs) _tvmet_always_inline;		\
										\
template<class E, int Rows, int Cols, class T>			\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME<std::complex<T>, typename E::value_type>,			\
    XprLiteral< std::complex<T> >,						\
    XprMatrix<E, Rows, Cols>							\
  >,										\
  Rows, Cols									\
>										\
operator OP (const std::complex<T>& lhs,					\
	     const XprMatrix<E, Rows, Cols>& rhs) _tvmet_always_inline;

TVMET_DECLARE_MACRO(add, +)
TVMET_DECLARE_MACRO(sub, -)
TVMET_DECLARE_MACRO(mul, *)
TVMET_DECLARE_MACRO(div, /)

#undef TVMET_DECLARE_MACRO

#endif // defined(EIGEN_USE_COMPLEX)


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix specific operator*() = prod() operations
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/**
 * \fn operator*(const XprMatrix<E1, Rows1, Cols1>& lhs, const XprMatrix<E2, Cols1, Cols2>& rhs)
 * \brief Evaluate the product of two XprMatrix.
 * \ingroup _binary_operator
 * \sa prod(XprMatrix<E1, Rows1, Cols1> lhs, XprMatrix<E2, Cols1, Cols2> rhs)
 */
template<class E1, int Rows1, int Cols1,
	 class E2, int Cols2>
XprMatrix<
  XprMMProduct<
    XprMatrix<E1, Rows1, Cols1>, Rows1, Cols1,	// M1(Rows1, Cols1)
    XprMatrix<E2, Cols1, Cols2>, Cols2		// M2(Cols1, Cols2)
  >,
  Rows1, Cols2
>
operator*(const XprMatrix<E1, Rows1, Cols1>& lhs,
	  const XprMatrix<E2, Cols1, Cols2>& rhs) _tvmet_always_inline;


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix-vector specific prod( ... ) operators
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/**
 * \fn operator*(const XprMatrix<E1, Rows, Cols>& lhs, const XprVector<E2, Cols>& rhs)
 * \brief Evaluate the product of XprMatrix and XprVector.
 * \ingroup _binary_operator
 * \sa prod(XprMatrix<E1, Rows, Cols> lhs, XprVector<E2, Cols> rhs)
 */
template<class E1, int Rows, int Cols,
	 class E2>
XprVector<
  XprMVProduct<
    XprMatrix<E1, Rows, Cols>, Rows, Cols,
    XprVector<E2, Cols>
  >,
  Rows
>
operator*(const XprMatrix<E1, Rows, Cols>& lhs,
	  const XprVector<E2, Cols>& rhs) _tvmet_always_inline;



/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * global unary operators
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * unary_operator(const XprMatrix<E, Rows, Cols>& m)
 * Note: per se element wise
 */
#define TVMET_DECLARE_MACRO(NAME, OP)						\
template <class E, int Rows, int Cols>				\
XprMatrix<									\
  XprUnOp<									\
    Fcnl_##NAME<typename E::value_type>,					\
    XprMatrix<E, Rows, Cols>							\
  >,										\
  Rows, Cols									\
>										\
operator OP (const XprMatrix<E, Rows, Cols>& m) _tvmet_always_inline;

TVMET_DECLARE_MACRO(neg, -)

#undef TVMET_DECLARE_MACRO


/*********************************************************
 * PART II: IMPLEMENTATION
 *********************************************************/



/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Matrix arithmetic operators implemented by functions
 * add, sub, mul and div
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * operator(const XprMatrix<E1, Rows1, Cols1>& lhs, const XprMatrix<E2, Cols1,Cols2>& rhs)
 *
 * Note: operations +,-,*,/ are per se element wise. Further more,
 * element wise operations make sense only for matrices of the same
 * size [varg].
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)					\
template<class E1, int Rows1, int Cols1,		\
         class E2>							\
inline									\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E1::value_type, typename E2::value_type>,	\
    XprMatrix<E1, Rows1, Cols1>,					\
    XprMatrix<E2, Rows1, Cols1>						\
  >,									\
  Rows1, Cols1								\
>									\
operator OP (const XprMatrix<E1, Rows1, Cols1>& lhs, 			\
	     const XprMatrix<E2, Rows1, Cols1>& rhs) {			\
  return NAME (lhs, rhs);						\
}

TVMET_IMPLEMENT_MACRO(add, +)		// per se element wise
TVMET_IMPLEMENT_MACRO(sub, -)		// per se element wise
namespace element_wise {
  TVMET_IMPLEMENT_MACRO(mul, *)		// see as prod()
  TVMET_IMPLEMENT_MACRO(div, /)		// not defined for matrizes, must be element_wise
}
#undef TVMET_IMPLEMENT_MACRO


/*
 * operator(XprMatrix<E, Rows, Cols>,  POD)
 * operator(POD, XprMatrix<E, Rows, Cols>)
 * Note: operations +,-,*,/ are per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP, POD)			\
template<class E, int Rows, int Cols>		\
inline								\
XprMatrix<							\
  XprBinOp<							\
    Fcnl_##NAME<typename E::value_type, POD >,			\
    XprMatrix<E, Rows, Cols>,					\
    XprLiteral< POD >						\
  >,								\
  Rows, Cols							\
>								\
operator OP (const XprMatrix<E, Rows, Cols>& lhs, POD rhs) {	\
  return NAME (lhs, rhs);					\
}								\
								\
template<class E,int Rows, int Cols>		\
inline								\
XprMatrix<							\
  XprBinOp<							\
    Fcnl_##NAME<POD, typename E::value_type>,			\
    XprLiteral< POD >,						\
    XprMatrix<E, Rows, Cols>					\
  >,								\
  Rows, Cols							\
>								\
operator OP (POD lhs, const XprMatrix<E, Rows, Cols>& rhs) {	\
  return NAME (lhs, rhs);					\
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
 * operator(XprMatrix<E, Rows, Cols>, complex<>)
 * operator(complex<>, XprMatrix<E, Rows, Cols>)
 * Note: operations +,-,*,/ are per se element wise
 * \todo type promotion
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)				\
template<class E, int Rows, int Cols, class T>	\
inline								\
XprMatrix<							\
  XprBinOp<							\
    Fcnl_##NAME<typename E::value_type, std::complex<T> >,	\
    XprMatrix<E, Rows, Cols>,					\
    XprLiteral< std::complex<T> >				\
  >,								\
  Rows, Cols							\
>								\
operator OP (const XprMatrix<E, Rows, Cols>& lhs,		\
	     const std::complex<T>& rhs) {			\
  return NAME (lhs, rhs);					\
}								\
								\
template<class E, int Rows, int Cols, class T>	\
inline								\
XprMatrix<							\
  XprBinOp<							\
    Fcnl_##NAME<std::complex<T>, typename E::value_type>,	\
    XprLiteral< std::complex<T> >,				\
    XprMatrix<E, Rows, Cols>					\
  >,								\
  Rows, Cols							\
>								\
operator OP (const std::complex<T>& lhs,			\
	     const XprMatrix<E, Rows, Cols>& rhs) {		\
  return NAME (lhs, rhs);					\
}

TVMET_IMPLEMENT_MACRO(add, +)
TVMET_IMPLEMENT_MACRO(sub, -)
TVMET_IMPLEMENT_MACRO(mul, *)
TVMET_IMPLEMENT_MACRO(div, /)

#undef TVMET_IMPLEMENT_MACRO

#endif // defined(EIGEN_USE_COMPLEX)


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix specific operator*() = prod() operations
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/**
 * \fn operator*(const XprMatrix<E1, Rows1, Cols1>& lhs, const XprMatrix<E2, Cols1, Cols2>& rhs)
 * \brief Evaluate the product of two XprMatrix.
 * \ingroup _binary_operator
 * \sa prod(XprMatrix<E1, Rows1, Cols1> lhs, XprMatrix<E2, Cols1, Cols2> rhs)
 */
template<class E1, int Rows1, int Cols1,
	 class E2, int Cols2>
inline
XprMatrix<
  XprMMProduct<
    XprMatrix<E1, Rows1, Cols1>, Rows1, Cols1,	// M1(Rows1, Cols1)
    XprMatrix<E2, Cols1, Cols2>, Cols2		// M2(Cols1, Cols2)
  >,
  Rows1, Cols2
>
operator*(const XprMatrix<E1, Rows1, Cols1>& lhs, const XprMatrix<E2, Cols1, Cols2>& rhs) {
  return prod(lhs, rhs);
}


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix-vector specific prod( ... ) operators
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/**
 * \fn operator*(const XprMatrix<E1, Rows, Cols>& lhs, const XprVector<E2, Cols>& rhs)
 * \brief Evaluate the product of XprMatrix and XprVector.
 * \ingroup _binary_operator
 * \sa prod(XprMatrix<E1, Rows, Cols> lhs, XprVector<E2, Cols> rhs)
 */
template<class E1, int Rows, int Cols,
	 class E2>
inline
XprVector<
  XprMVProduct<
    XprMatrix<E1, Rows, Cols>, Rows, Cols,
    XprVector<E2, Cols>
  >,
  Rows
>
operator*(const XprMatrix<E1, Rows, Cols>& lhs, const XprVector<E2, Cols>& rhs) {
  return prod(lhs, rhs);
}



/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * global unary operators
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * unary_operator(const XprMatrix<E, Rows, Cols>& m)
 * Note: per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)					\
template <class E, int Rows, int Cols>			\
inline									\
XprMatrix<								\
  XprUnOp<								\
    Fcnl_##NAME<typename E::value_type>,				\
    XprMatrix<E, Rows, Cols>						\
  >,									\
  Rows, Cols								\
>									\
operator OP (const XprMatrix<E, Rows, Cols>& m) {			\
  typedef XprUnOp<							\
    Fcnl_##NAME<typename E::value_type>,				\
    XprMatrix<E, Rows, Cols>						\
  >  							 expr_type;	\
  return XprMatrix<expr_type, Rows, Cols>(expr_type(m));		\
}

TVMET_IMPLEMENT_MACRO(neg, -)

#undef TVMET_IMPLEMENT_MACRO


} // namespace tvmet

#endif // TVMET_XPR_MATRIX_OPERATORS_H

// Local Variables:
// mode:C++
// End:
