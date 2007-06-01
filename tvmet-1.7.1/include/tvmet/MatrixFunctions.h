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
 * $Id: MatrixFunctions.h,v 1.59 2004/11/04 16:21:17 opetzold Exp $
 */

#ifndef TVMET_MATRIX_FUNCTIONS_H
#define TVMET_MATRIX_FUNCTIONS_H

#include <tvmet/Extremum.h>

namespace tvmet {

/* forwards */
template<class T, int Sz> class Vector;
template<class T, int Sz> class VectorConstReference;


/*********************************************************
 * PART I: DECLARATION
 *********************************************************/


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Vector arithmetic functions add, sub, mul and div
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * function(Matrix<T1, Rows, Cols>, Matrix<T2, Rows, Cols>)
 * function(XprMatrix<E, Rows, Cols>, Matrix<T, Rows, Cols>)
 * function(Matrix<T, Rows, Cols>, XprMatrix<E, Rows, Cols>)
 */
#define TVMET_DECLARE_MACRO(NAME)					\
template<class T1, class T2, int Rows, int Cols>	\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<T1, T2>,						\
    MatrixConstReference<T1, Rows, Cols>,				\
    MatrixConstReference<T2, Rows, Cols>				\
  >,									\
  Rows, Cols								\
>									\
NAME (const Matrix<T1, Rows, Cols>& lhs,				\
      const Matrix<T2, Rows, Cols>& rhs) TVMET_CXX_ALWAYS_INLINE;	\
									\
template<class E, class T, int Rows, int Cols>		\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, T>,				\
    XprMatrix<E, Rows, Cols>,						\
    MatrixConstReference<T, Rows, Cols>					\
  >,									\
  Rows, Cols								\
>									\
NAME (const XprMatrix<E, Rows, Cols>& lhs,				\
      const Matrix<T, Rows, Cols>& rhs) TVMET_CXX_ALWAYS_INLINE;	\
									\
template<class T, class E, int Rows, int Cols>		\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, T>,				\
    MatrixConstReference<T, Rows, Cols>,				\
    XprMatrix<E, Rows, Cols>						\
  >,									\
  Rows, Cols								\
>									\
NAME (const Matrix<T, Rows, Cols>& lhs,					\
      const XprMatrix<E, Rows, Cols>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(add)			// per se element wise
TVMET_DECLARE_MACRO(sub)			// per se element wise
namespace element_wise {
  TVMET_DECLARE_MACRO(mul)			// not defined for matrizes
  TVMET_DECLARE_MACRO(div)			// not defined for matrizes
}

#undef TVMET_DECLARE_MACRO


/*
 * function(Matrix<T, Rows, Cols>, POD)
 * function(POD, Matrix<T, Rows, Cols>)
 * Note: - operations +,-,*,/ are per se element wise
 */
#define TVMET_DECLARE_MACRO(NAME, POD)					\
template<class T, int Rows, int Cols>			\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<T, POD >,						\
    MatrixConstReference<T, Rows, Cols>,				\
    XprLiteral<POD >							\
  >,									\
  Rows, Cols								\
>									\
NAME (const Matrix<T, Rows, Cols>& lhs, 				\
      POD rhs) TVMET_CXX_ALWAYS_INLINE;					\
									\
template<class T, int Rows, int Cols>			\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME< POD, T>,						\
    XprLiteral< POD >,							\
    MatrixConstReference<T, Rows, Cols>					\
  >,									\
  Rows, Cols								\
>									\
NAME (POD lhs, 								\
      const Matrix<T, Rows, Cols>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(add, int)
TVMET_DECLARE_MACRO(sub, int)
TVMET_DECLARE_MACRO(mul, int)
TVMET_DECLARE_MACRO(div, int)

#if defined(TVMET_HAVE_LONG_LONG)
TVMET_DECLARE_MACRO(add, long long int)
TVMET_DECLARE_MACRO(sub, long long int)
TVMET_DECLARE_MACRO(mul, long long int)
TVMET_DECLARE_MACRO(div, long long int)
#endif

TVMET_DECLARE_MACRO(add, float)
TVMET_DECLARE_MACRO(sub, float)
TVMET_DECLARE_MACRO(mul, float)
TVMET_DECLARE_MACRO(div, float)

TVMET_DECLARE_MACRO(add, double)
TVMET_DECLARE_MACRO(sub, double)
TVMET_DECLARE_MACRO(mul, double)
TVMET_DECLARE_MACRO(div, double)

#if defined(TVMET_HAVE_LONG_DOUBLE)
TVMET_DECLARE_MACRO(add, long double)
TVMET_DECLARE_MACRO(sub, long double)
TVMET_DECLARE_MACRO(mul, long double)
TVMET_DECLARE_MACRO(div, long double)
#endif

#undef TVMET_DECLARE_MACRO


#if defined(TVMET_HAVE_COMPLEX)
/*
 * function(Matrix<T, Rows, Cols>, complex<T>)
 * function(complex<T>, Matrix<T, Rows, Cols>)
 * Note: - operations +,-,*,/ are per se element wise
 * \todo type promotion
 */
#define TVMET_DECLARE_MACRO(NAME)						\
template<class T, int Rows, int Cols>				\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,				\
    MatrixConstReference< std::complex<T>, Rows, Cols>,				\
    XprLiteral<std::complex<T> >						\
  >,										\
  Rows, Cols									\
>										\
NAME (const Matrix< std::complex<T>, Rows, Cols>& lhs,				\
      const std::complex<T>& rhs) TVMET_CXX_ALWAYS_INLINE;			\
										\
template<class T, int Rows, int Cols>				\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,				\
    XprLiteral< std::complex<T> >,						\
    MatrixConstReference< std::complex<T>, Rows, Cols>				\
  >,										\
  Rows, Cols									\
>										\
NAME (const std::complex<T>& lhs,						\
      const Matrix< std::complex<T>, Rows, Cols>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(add)
TVMET_DECLARE_MACRO(sub)
TVMET_DECLARE_MACRO(mul)
TVMET_DECLARE_MACRO(div)

#undef TVMET_DECLARE_MACRO

#endif // defined(TVMET_HAVE_COMPLEX)


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix specific prod( ... ) functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


template<class T1, int Rows1, int Cols1,
	 class T2, int Cols2>
XprMatrix<
  XprMMProduct<
    MatrixConstReference<T1, Rows1, Cols1>, Rows1, Cols1,	// M1(Rows1, Cols1)
    MatrixConstReference<T2, Cols1, Cols2>, Cols2 		// M2(Cols1, Cols2)
  >,
  Rows1, Cols2							// return Dim
>
prod(const Matrix<T1, Rows1, Cols1>& lhs,
     const Matrix<T2, Cols1, Cols2>& rhs) TVMET_CXX_ALWAYS_INLINE;


template<class E1, int Rows1, int Cols1,
	 class T2, int Cols2>
XprMatrix<
  XprMMProduct<
    XprMatrix<E1, Rows1, Cols1>, Rows1, Cols1,			// M1(Rows1, Cols1)
    MatrixConstReference<T2, Cols1, Cols2>, Cols2		// M2(Cols1, Cols2)
  >,
  Rows1, Cols2							// return Dim
>
prod(const XprMatrix<E1, Rows1, Cols1>& lhs,
     const Matrix<T2, Cols1, Cols2>& rhs) TVMET_CXX_ALWAYS_INLINE;


template<class T1, int Rows1, int Cols1,
	 class E2, int Cols2>
XprMatrix<
  XprMMProduct<
    MatrixConstReference<T1, Rows1, Cols1>, Rows1, Cols1,	// M1(Rows1, Cols1)
    XprMatrix<E2, Cols1, Cols2>, Cols2				// M2(Cols1, Cols2)
  >,
  Rows1, Cols2							// return Dim
>
prod(const Matrix<T1, Rows1, Cols1>& lhs,
     const XprMatrix<E2, Cols1, Cols2>& rhs) TVMET_CXX_ALWAYS_INLINE;


template<class T1, int Rows1, int Cols1,
	 class T2, int Cols2>
XprMatrix<
  XprMMProductTransposed<
    MatrixConstReference<T1, Rows1, Cols1>, Rows1, Cols1,	// M1(Rows1, Cols1)
    MatrixConstReference<T2, Cols1, Cols2>, Cols2		// M2(Cols1, Cols2)
  >,
  Cols2, Rows1							// return Dim
>
trans_prod(const Matrix<T1, Rows1, Cols1>& lhs,
	   const Matrix<T2, Cols1, Cols2>& rhs) TVMET_CXX_ALWAYS_INLINE;


template<class T1, int Rows1, int Cols1,
	 class T2, int Cols2>	// Rows2 = Rows1
XprMatrix<
  XprMtMProduct<
    MatrixConstReference<T1, Rows1, Cols1>, Rows1, Cols1,	// M1(Rows1, Cols1)
    MatrixConstReference<T2, Rows1, Cols2>, Cols2		// M2(Rows1, Cols2)
  >,
  Cols1, Cols2							// return Dim
>
MtM_prod(const Matrix<T1, Rows1, Cols1>& lhs,
	 const Matrix<T2, Rows1, Cols2>& rhs) TVMET_CXX_ALWAYS_INLINE;


template<class T1, int Rows1, int Cols1,
	 class T2, int Rows2>
XprMatrix<
  XprMMtProduct<
    MatrixConstReference<T1, Rows1, Cols1>, Rows1, Cols1,	// M1(Rows1, Cols1)
    MatrixConstReference<T2, Rows2, Cols1>, Cols1 		// M2(Rows2, Cols1)
  >,
  Rows1, Rows2							// return Dim
>
MMt_prod(const Matrix<T1, Rows1, Cols1>& lhs,
	 const Matrix<T2, Rows2, Cols1>& rhs) TVMET_CXX_ALWAYS_INLINE;


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix-vector specific prod( ... ) functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


template<class T1, class T2, int Rows, int Cols>
XprVector<
  XprMVProduct<
    MatrixConstReference<T1, Rows, Cols>, Rows, Cols,	// M(Rows, Cols)
    VectorConstReference<T2, Cols> 			// V
  >,
  Rows
>
prod(const Matrix<T1, Rows, Cols>& lhs,
     const Vector<T2, Cols>& rhs) TVMET_CXX_ALWAYS_INLINE;


template<class T1, class E2, int Rows, int Cols>
XprVector<
  XprMVProduct<
    MatrixConstReference<T1, Rows, Cols>, Rows, Cols,
    XprVector<E2, Cols>
  >,
  Rows
>
prod(const Matrix<T1, Rows, Cols>& lhs,
     const XprVector<E2, Cols>& rhs) TVMET_CXX_ALWAYS_INLINE;


template<class E1, class T2, int Rows, int Cols>
XprVector<
  XprMVProduct<
    XprMatrix<E1, Rows, Cols>, Rows, Cols,		// M(Rows, Cols)
    VectorConstReference<T2, Cols> 			// V
  >,
  Rows
>
prod(const XprMatrix<E1, Rows, Cols>& lhs,
     const Vector<T2, Cols>& rhs) TVMET_CXX_ALWAYS_INLINE;


template<class T1, class T2, int Rows, int Cols>
XprVector<
  XprMtVProduct<
    MatrixConstReference<T1, Rows, Cols>, Rows,	Cols,   // M(Rows, Cols)
    VectorConstReference<T2, Rows> 			// V
  >,
  Cols
>
Mtx_prod(const Matrix<T1, Rows, Cols>& lhs,
	 const Vector<T2, Rows>& rhs) TVMET_CXX_ALWAYS_INLINE;


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix specific functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


template<class T, int Rows, int Cols>
XprMatrix<
  XprMatrixTranspose<
    MatrixConstReference<T, Rows, Cols>
  >,
  Cols, Rows
>
trans(const Matrix<T, Rows, Cols>& rhs) TVMET_CXX_ALWAYS_INLINE;


template<class T, int Sz>
typename NumericTraits<T>::sum_type
trace(const Matrix<T, Sz, Sz>& m) TVMET_CXX_ALWAYS_INLINE;


template<class T, int Rows, int Cols>
XprVector<
  XprMatrixRow<
    MatrixConstReference<T, Rows, Cols>,
    Rows, Cols
  >,
  Cols
>
row(const Matrix<T, Rows, Cols>& m,
    int no) TVMET_CXX_ALWAYS_INLINE;


template<class T, int Rows, int Cols>
XprVector<
  XprMatrixCol<
    MatrixConstReference<T, Rows, Cols>,
    Rows, Cols
  >,
  Rows
>
col(const Matrix<T, Rows, Cols>& m,
    int no) TVMET_CXX_ALWAYS_INLINE;


template<class T, int Sz>
XprVector<
  XprMatrixDiag<
    MatrixConstReference<T, Sz, Sz>,
    Sz
  >,
  Sz
>
diag(const Matrix<T, Sz, Sz>& m) TVMET_CXX_ALWAYS_INLINE;


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * min/max unary functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


template<class E, int Rows, int Cols>
Extremum<typename E::value_type, int, matrix_tag>
maximum(const XprMatrix<E, Rows, Cols>& e); // NOT TVMET_CXX_ALWAYS_INLINE;


template<class T, int Rows, int Cols>
Extremum<T, int, matrix_tag>
maximum(const Matrix<T, Rows, Cols>& m) TVMET_CXX_ALWAYS_INLINE;


template<class E, int Rows, int Cols>
Extremum<typename E::value_type, int, matrix_tag>
minimum(const XprMatrix<E, Rows, Cols>& e); // NOT TVMET_CXX_ALWAYS_INLINE;


template<class T, int Rows, int Cols>
Extremum<T, int, matrix_tag>
minimum(const Matrix<T, Rows, Cols>& m) TVMET_CXX_ALWAYS_INLINE;


template<class E, int Rows, int Cols>
typename E::value_type
max(const XprMatrix<E, Rows, Cols>& e); // NOT TVMET_CXX_ALWAYS_INLINE;


template<class T, int Rows, int Cols>
T max(const Matrix<T, Rows, Cols>& m) TVMET_CXX_ALWAYS_INLINE;


template<class E, int Rows, int Cols>
typename E::value_type
min(const XprMatrix<E, Rows, Cols>& e); // NOT TVMET_CXX_ALWAYS_INLINE;


template<class T, int Rows, int Cols>
T min(const Matrix<T, Rows, Cols>& m) TVMET_CXX_ALWAYS_INLINE;


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * other unary functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


template<class T, int Rows, int Cols>
XprMatrix<
  XprIdentity<T, Rows, Cols>,
  Rows, Cols
>
identity() TVMET_CXX_ALWAYS_INLINE;


template<class M>
XprMatrix<
  XprIdentity<
    typename M::value_type,
    M::Rows, M::Cols>,
  M::Rows, M::Cols
>
identity() TVMET_CXX_ALWAYS_INLINE;


template<class T, int Rows, int Cols>
XprMatrix<
  MatrixConstReference<T, Rows, Cols>,
  Rows, Cols
>
cmatrix_ref(const T* mem) TVMET_CXX_ALWAYS_INLINE;


/*********************************************************
 * PART II: IMPLEMENTATION
 *********************************************************/


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Vector arithmetic functions add, sub, mul and div
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * function(Matrix<T1, Rows, Cols>, Matrix<T2, Rows, Cols>)
 * function(XprMatrix<E, Rows, Cols>, Matrix<T, Rows, Cols>)
 * function(Matrix<T, Rows, Cols>, XprMatrix<E, Rows, Cols>)
 */
#define TVMET_IMPLEMENT_MACRO(NAME)						\
template<class T1, class T2, int Rows, int Cols>		\
inline										\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME<T1, T2>,							\
    MatrixConstReference<T1, Rows, Cols>,					\
    MatrixConstReference<T2, Rows, Cols>					\
  >,										\
  Rows, Cols									\
>										\
NAME (const Matrix<T1, Rows, Cols>& lhs, const Matrix<T2, Rows, Cols>& rhs) {	\
  typedef XprBinOp <								\
    Fcnl_##NAME<T1, T2>,							\
    MatrixConstReference<T1, Rows, Cols>,					\
    MatrixConstReference<T2, Rows, Cols>					\
  >							expr_type;		\
  return XprMatrix<expr_type, Rows, Cols>(					\
    expr_type(lhs.const_ref(), rhs.const_ref()));				\
}										\
										\
template<class E, class T, int Rows, int Cols>			\
inline										\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME<typename E::value_type, T>,					\
    XprMatrix<E, Rows, Cols>,							\
    MatrixConstReference<T, Rows, Cols>						\
  >,										\
  Rows, Cols									\
>										\
NAME (const XprMatrix<E, Rows, Cols>& lhs, const Matrix<T, Rows, Cols>& rhs) {	\
  typedef XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, T>,					\
    XprMatrix<E, Rows, Cols>,							\
    MatrixConstReference<T, Rows, Cols>						\
  > 							 expr_type;		\
  return XprMatrix<expr_type, Rows, Cols>(					\
    expr_type(lhs, rhs.const_ref()));						\
}										\
										\
template<class T, class E, int Rows, int Cols>			\
inline										\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME<typename E::value_type, T>,					\
    MatrixConstReference<T, Rows, Cols>,					\
    XprMatrix<E, Rows, Cols>							\
  >,										\
  Rows, Cols									\
>										\
NAME (const Matrix<T, Rows, Cols>& lhs, const XprMatrix<E, Rows, Cols>& rhs) {	\
  typedef XprBinOp<								\
    Fcnl_##NAME<T, typename E::value_type>,					\
    MatrixConstReference<T, Rows, Cols>,					\
    XprMatrix<E, Rows, Cols>							\
  >	 						 expr_type;		\
  return XprMatrix<expr_type, Rows, Cols>(					\
    expr_type(lhs.const_ref(), rhs));						\
}

TVMET_IMPLEMENT_MACRO(add)			// per se element wise
TVMET_IMPLEMENT_MACRO(sub)			// per se element wise
namespace element_wise {
  TVMET_IMPLEMENT_MACRO(mul)			// not defined for matrizes
  TVMET_IMPLEMENT_MACRO(div)			// not defined for matrizes
}

#undef TVMET_IMPLEMENT_MACRO


/*
 * function(Matrix<T, Rows, Cols>, POD)
 * function(POD, Matrix<T, Rows, Cols>)
 * Note: - operations +,-,*,/ are per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, POD)				\
template<class T, int Rows, int Cols>			\
inline									\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<T, POD >,						\
    MatrixConstReference<T, Rows, Cols>,				\
    XprLiteral<POD >							\
  >,									\
  Rows, Cols								\
>									\
NAME (const Matrix<T, Rows, Cols>& lhs, POD rhs) {			\
  typedef XprBinOp<							\
    Fcnl_##NAME<T, POD >,						\
    MatrixConstReference<T, Rows, Cols>,				\
    XprLiteral< POD >							\
  >							expr_type;	\
  return XprMatrix<expr_type, Rows, Cols>(				\
    expr_type(lhs.const_ref(), XprLiteral< POD >(rhs)));		\
}									\
									\
template<class T, int Rows, int Cols>			\
inline									\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME< POD, T>,						\
    XprLiteral< POD >,							\
    MatrixConstReference<T, Rows, Cols>					\
  >,									\
  Rows, Cols								\
>									\
NAME (POD lhs, const Matrix<T, Rows, Cols>& rhs) {			\
  typedef XprBinOp<							\
    Fcnl_##NAME< POD, T>,						\
    XprLiteral< POD >,							\
    MatrixConstReference<T, Rows, Cols>					\
  >							expr_type;	\
  return XprMatrix<expr_type, Rows, Cols>(				\
    expr_type(XprLiteral< POD >(lhs), rhs.const_ref()));		\
}

TVMET_IMPLEMENT_MACRO(add, int)
TVMET_IMPLEMENT_MACRO(sub, int)
TVMET_IMPLEMENT_MACRO(mul, int)
TVMET_IMPLEMENT_MACRO(div, int)

#if defined(TVMET_HAVE_LONG_LONG)
TVMET_IMPLEMENT_MACRO(add, long long int)
TVMET_IMPLEMENT_MACRO(sub, long long int)
TVMET_IMPLEMENT_MACRO(mul, long long int)
TVMET_IMPLEMENT_MACRO(div, long long int)
#endif

TVMET_IMPLEMENT_MACRO(add, float)
TVMET_IMPLEMENT_MACRO(sub, float)
TVMET_IMPLEMENT_MACRO(mul, float)
TVMET_IMPLEMENT_MACRO(div, float)

TVMET_IMPLEMENT_MACRO(add, double)
TVMET_IMPLEMENT_MACRO(sub, double)
TVMET_IMPLEMENT_MACRO(mul, double)
TVMET_IMPLEMENT_MACRO(div, double)

#if defined(TVMET_HAVE_LONG_DOUBLE)
TVMET_IMPLEMENT_MACRO(add, long double)
TVMET_IMPLEMENT_MACRO(sub, long double)
TVMET_IMPLEMENT_MACRO(mul, long double)
TVMET_IMPLEMENT_MACRO(div, long double)
#endif

#undef TVMET_IMPLEMENT_MACRO


#if defined(TVMET_HAVE_COMPLEX)
/*
 * function(Matrix<T, Rows, Cols>, complex<T>)
 * function(complex<T>, Matrix<T, Rows, Cols>)
 * Note: - operations +,-,*,/ are per se element wise
 * \todo type promotion
 */
#define TVMET_IMPLEMENT_MACRO(NAME)					\
template<class T, int Rows, int Cols>			\
inline									\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,			\
    MatrixConstReference< std::complex<T>, Rows, Cols>,			\
    XprLiteral<std::complex<T> >					\
  >,									\
  Rows, Cols								\
>									\
NAME (const Matrix< std::complex<T>, Rows, Cols>& lhs,			\
      const std::complex<T>& rhs) {					\
  typedef XprBinOp<							\
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,			\
    MatrixConstReference< std::complex<T>, Rows, Cols>,			\
    XprLiteral< std::complex<T> >					\
  >							expr_type;	\
  return XprMatrix<expr_type, Rows, Cols>(				\
    expr_type(lhs.const_ref(), XprLiteral< std::complex<T> >(rhs)));	\
}									\
									\
template<class T, int Rows, int Cols>			\
inline									\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,			\
    XprLiteral< std::complex<T> >,					\
    MatrixConstReference< std::complex<T>, Rows, Cols>			\
  >,									\
  Rows, Cols								\
>									\
NAME (const std::complex<T>& lhs,					\
      const Matrix< std::complex<T>, Rows, Cols>& rhs) {		\
  typedef XprBinOp<							\
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,			\
    XprLiteral< std::complex<T> >,					\
    MatrixConstReference<T, Rows, Cols>					\
  >							expr_type;	\
  return XprMatrix<expr_type, Rows, Cols>(				\
    expr_type(XprLiteral< std::complex<T> >(lhs), rhs.const_ref()));	\
}

TVMET_IMPLEMENT_MACRO(add)
TVMET_IMPLEMENT_MACRO(sub)
TVMET_IMPLEMENT_MACRO(mul)
TVMET_IMPLEMENT_MACRO(div)

#undef TVMET_IMPLEMENT_MACRO

#endif // defined(TVMET_HAVE_COMPLEX)


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix specific prod( ... ) functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/**
 * \fn prod(const Matrix<T1, Rows1, Cols1>& lhs, const Matrix<T2, Cols1, Cols2>& rhs)
 * \brief Function for the matrix-matrix-product.
 * \ingroup _binary_function
 * \note The rows2 has to be equal to cols1.
 */
template<class T1, int Rows1, int Cols1,
	 class T2, int Cols2>
inline
XprMatrix<
  XprMMProduct<
    MatrixConstReference<T1, Rows1, Cols1>, Rows1, Cols1,	// M1(Rows1, Cols1)
    MatrixConstReference<T2, Cols1, Cols2>, Cols2 		// M2(Cols1, Cols2)
  >,
  Rows1, Cols2							// return Dim
>
prod(const Matrix<T1, Rows1, Cols1>& lhs, const Matrix<T2, Cols1, Cols2>& rhs) {
  typedef XprMMProduct<
    MatrixConstReference<T1, Rows1, Cols1>, Rows1, Cols1,
    MatrixConstReference<T2, Cols1, Cols2>, Cols2
  >							expr_type;
  return XprMatrix<expr_type, Rows1, Cols2>(
    expr_type(lhs.const_ref(), rhs.const_ref()));
}


/**
 * \fn prod(const XprMatrix<E1, Rows1, Cols1>& lhs, const Matrix<T2, Cols1, Cols2>& rhs)
 * \brief Evaluate the product of XprMatrix and Matrix.
 * \ingroup _binary_function
 */
template<class E1, int Rows1, int Cols1,
	 class T2, int Cols2>
inline
XprMatrix<
  XprMMProduct<
    XprMatrix<E1, Rows1, Cols1>, Rows1, Cols1,			// M1(Rows1, Cols1)
    MatrixConstReference<T2, Cols1, Cols2>, Cols2		// M2(Cols1, Cols2)
  >,
  Rows1, Cols2							// return Dim
>
prod(const XprMatrix<E1, Rows1, Cols1>& lhs, const Matrix<T2, Cols1, Cols2>& rhs) {
  typedef XprMMProduct<
    XprMatrix<E1, Rows1, Cols1>, Rows1, Cols1,
    MatrixConstReference<T2, Cols1, Cols2>, Cols2
  >							expr_type;
  return XprMatrix<expr_type, Rows1, Cols2>(
    expr_type(lhs, rhs.const_ref()));
}


/**
 * \fn prod(const Matrix<T1, Rows1, Cols1>& lhs, const XprMatrix<E2, Cols1, Cols2>& rhs)
 * \brief Evaluate the product of Matrix and XprMatrix.
 * \ingroup _binary_function
 */
template<class T1, int Rows1, int Cols1,
	 class E2, int Cols2>
inline
XprMatrix<
  XprMMProduct<
    MatrixConstReference<T1, Rows1, Cols1>, Rows1, Cols1,	// M1(Rows1, Cols1)
    XprMatrix<E2, Cols1, Cols2>, Cols2				// M2(Cols1, Cols2)
  >,
  Rows1, Cols2							// return Dim
>
prod(const Matrix<T1, Rows1, Cols1>& lhs, const XprMatrix<E2, Cols1, Cols2>& rhs) {
  typedef XprMMProduct<
    MatrixConstReference<T1, Rows1, Cols1>, Rows1, Cols1,
    XprMatrix<E2, Cols1, Cols2>, Cols2
  >							expr_type;
  return XprMatrix<expr_type, Rows1, Cols2>(
    expr_type(lhs.const_ref(), rhs));
}


/**
 * \fn trans_prod(const Matrix<T1, Rows1, Cols1>& lhs, const Matrix<T2, Cols1, Cols2>& rhs)
 * \brief Function for the trans(matrix-matrix-product)
 * \ingroup _binary_function
 * Perform on given Matrix M1 and M2:
 * \f[
 * (M_1\,M_2)^T
 * \f]
 */
template<class T1, int Rows1, int Cols1,
	 class T2, int Cols2>
inline
XprMatrix<
  XprMMProductTransposed<
    MatrixConstReference<T1, Rows1, Cols1>, Rows1, Cols1,	// M1(Rows1, Cols1)
    MatrixConstReference<T2, Cols1, Cols2>, Cols2		// M2(Cols1, Cols2)
  >,
  Cols2, Rows1							// return Dim
>
trans_prod(const Matrix<T1, Rows1, Cols1>& lhs, const Matrix<T2, Cols1, Cols2>& rhs) {
  typedef XprMMProductTransposed<
    MatrixConstReference<T1, Rows1, Cols1>, Rows1, Cols1,
    MatrixConstReference<T2, Cols1, Cols2>, Cols2
  >							expr_type;
  return XprMatrix<expr_type, Cols2, Rows1>(
    expr_type(lhs.const_ref(), rhs.const_ref()));
}


/**
 * \fn MtM_prod(const Matrix<T1, Rows1, Cols1>& lhs, const Matrix<T2, Rows1, Cols2>& rhs)
 * \brief Function for the trans(matrix)-matrix-product.
 * \ingroup _binary_function
 *        using formula
 *        \f[
 *        M_1^{T}\,M_2
 *        \f]
 * \note The number of cols of matrix 2 have to be equal to number of rows of
 *       matrix 1, since matrix 1 is trans - the result is a (Cols1 x Cols2)
 *       matrix.
 */
template<class T1, int Rows1, int Cols1,
	 class T2, int Cols2>	// Rows2 = Rows1
inline
XprMatrix<
  XprMtMProduct<
    MatrixConstReference<T1, Rows1, Cols1>, Rows1, Cols1,	// M1(Rows1, Cols1)
    MatrixConstReference<T2, Rows1, Cols2>, Cols2		// M2(Rows1, Cols2)
  >,
  Cols1, Cols2							// return Dim
>
MtM_prod(const Matrix<T1, Rows1, Cols1>& lhs, const Matrix<T2, Rows1, Cols2>& rhs) {
  typedef XprMtMProduct<
    MatrixConstReference<T1, Rows1, Cols1>, Rows1, Cols1,
    MatrixConstReference<T2, Rows1, Cols2>, Cols2
  >							expr_type;
  return XprMatrix<expr_type, Cols1, Cols2>(
    expr_type(lhs.const_ref(), rhs.const_ref()));
}


/**
 * \fn MMt_prod(const Matrix<T1, Rows1, Cols1>& lhs, const Matrix<T2, Rows2, Cols1>& rhs)
 * \brief Function for the matrix-trans(matrix)-product.
 * \ingroup _binary_function
 * \note The Cols2 has to be equal to Cols1.
 */
template<class T1, int Rows1, int Cols1,
	 class T2, int Rows2>
inline
XprMatrix<
  XprMMtProduct<
    MatrixConstReference<T1, Rows1, Cols1>, Rows1, Cols1,	// M1(Rows1, Cols1)
    MatrixConstReference<T2, Rows2, Cols1>, Cols1 		// M2(Rows2, Cols1)
  >,
  Rows1, Rows2							// return Dim
>
MMt_prod(const Matrix<T1, Rows1, Cols1>& lhs, const Matrix<T2, Rows2, Cols1>& rhs) {
  typedef XprMMtProduct<
    MatrixConstReference<T1, Rows1, Cols1>, Rows1, Cols1,
    MatrixConstReference<T2, Rows2, Cols1>, Cols1
  >							expr_type;
  return XprMatrix<expr_type, Rows1, Rows2>(
    expr_type(lhs.const_ref(), rhs.const_ref()));
}


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix-vector specific prod( ... ) functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/**
 * \fn prod(const Matrix<T1, Rows, Cols>& lhs, const Vector<T2, Cols>& rhs)
 * \brief Function for the matrix-vector-product
 * \ingroup _binary_function
 */
template<class T1, class T2, int Rows, int Cols>
inline
XprVector<
  XprMVProduct<
    MatrixConstReference<T1, Rows, Cols>, Rows, Cols,	// M(Rows, Cols)
    VectorConstReference<T2, Cols> 			// V
  >,
  Rows
>
prod(const Matrix<T1, Rows, Cols>& lhs, const Vector<T2, Cols>& rhs) {
  typedef XprMVProduct<
    MatrixConstReference<T1, Rows, Cols>, Rows, Cols,
    VectorConstReference<T2, Cols>
  > 							expr_type;
  return XprVector<expr_type, Rows>(
    expr_type(lhs.const_ref(), rhs.const_ref()));
}


/**
 * \fn prod(const Matrix<T1, Rows, Cols>& lhs, const XprVector<E2, Cols>& rhs)
 * \brief Function for the matrix-vector-product
 * \ingroup _binary_function
 */
template<class T1, class E2, int Rows, int Cols>
inline
XprVector<
  XprMVProduct<
    MatrixConstReference<T1, Rows, Cols>, Rows, Cols,
    XprVector<E2, Cols>
  >,
  Rows
>
prod(const Matrix<T1, Rows, Cols>& lhs, const XprVector<E2, Cols>& rhs) {
  typedef XprMVProduct<
    MatrixConstReference<T1, Rows, Cols>, Rows, Cols,
    XprVector<E2, Cols>
  > 							expr_type;
  return XprVector<expr_type, Rows>(
    expr_type(lhs.const_ref(), rhs));
}


/*
 * \fn prod(const XprMatrix<E, Rows, Cols>& lhs, const Vector<T, Cols>& rhs)
 * \brief Compute the product of an XprMatrix with a Vector.
 * \ingroup _binary_function
 */
template<class E1, class T2, int Rows, int Cols>
inline
XprVector<
  XprMVProduct<
    XprMatrix<E1, Rows, Cols>, Rows, Cols,		// M(Rows, Cols)
    VectorConstReference<T2, Cols> 			// V
  >,
  Rows
>
prod(const XprMatrix<E1, Rows, Cols>& lhs, const Vector<T2, Cols>& rhs) {
  typedef XprMVProduct<
    XprMatrix<E1, Rows, Cols>, Rows, Cols,
    VectorConstReference<T2, Cols>
  > 							expr_type;
  return XprVector<expr_type, Rows>(
    expr_type(lhs, rhs.const_ref()));
}


/**
 * \fn Mtx_prod(const Matrix<T1, Rows, Cols>& matrix, const Vector<T2, Rows>& vector)
 * \brief Function for the trans(matrix)-vector-product
 * \ingroup _binary_function
 * Perform on given Matrix M and vector x:
 * \f[
 * M^T\, x
 * \f]
 */
template<class T1, class T2, int Rows, int Cols>
inline
XprVector<
  XprMtVProduct<
    MatrixConstReference<T1, Rows, Cols>, Rows,	Cols,   // M(Rows, Cols)
    VectorConstReference<T2, Rows> 			// V
  >,
  Cols
>
Mtx_prod(const Matrix<T1, Rows, Cols>& lhs, const Vector<T2, Rows>& rhs) {
  typedef XprMtVProduct<
    MatrixConstReference<T1, Rows, Cols>, Rows, Cols,
    VectorConstReference<T2, Rows>
  > 							expr_type;
  return XprVector<expr_type, Cols>(
    expr_type(lhs.const_ref(), rhs.const_ref()));
}


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix specific functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/**
 * \fn trans(const Matrix<T, Rows, Cols>& rhs)
 * \brief Transpose the matrix
 * \ingroup _unary_function
 */
template<class T, int Rows, int Cols>
inline
XprMatrix<
  XprMatrixTranspose<
    MatrixConstReference<T, Rows, Cols>
  >,
  Cols, Rows
>
trans(const Matrix<T, Rows, Cols>& rhs) {
  typedef XprMatrixTranspose<
    MatrixConstReference<T, Rows, Cols>
  >							expr_type;
  return XprMatrix<expr_type, Cols, Rows>(
    expr_type(rhs.const_ref()));
}


/*
 * \fn trace(const Matrix<T, Sz, Sz>& m)
 * \brief Compute the trace of a square matrix.
 * \ingroup _unary_function
 *
 * Simply compute the trace of the given matrix as:
 * \f[
 *  \sum_{k = 0}^{Sz-1} m(k, k)
 * \f]
 */
template<class T, int Sz>
inline
typename NumericTraits<T>::sum_type
trace(const Matrix<T, Sz, Sz>& m) {
  return meta::Matrix<Sz, Sz, 0, 0>::trace(m);
}


/**
 * \fn row(const Matrix<T, Rows, Cols>& m, int no)
 * \brief Returns a row vector of the given matrix.
 * \ingroup _binary_function
 */
template<class T, int Rows, int Cols>
inline
XprVector<
  XprMatrixRow<
    MatrixConstReference<T, Rows, Cols>,
    Rows, Cols
  >,
  Cols
>
row(const Matrix<T, Rows, Cols>& m, int no) {
  typedef XprMatrixRow<
    MatrixConstReference<T, Rows, Cols>,
    Rows, Cols
  >							expr_type;
  return XprVector<expr_type, Cols>(expr_type(m.const_ref(), no));
}


/**
 * \fn col(const Matrix<T, Rows, Cols>& m, int no)
 * \brief Returns a column vector of the given matrix.
 * \ingroup _binary_function
 */
template<class T, int Rows, int Cols>
inline
XprVector<
  XprMatrixCol<
    MatrixConstReference<T, Rows, Cols>,
    Rows, Cols
  >,
  Rows
>
col(const Matrix<T, Rows, Cols>& m, int no) {
  typedef XprMatrixCol<
    MatrixConstReference<T, Rows, Cols>,
    Rows, Cols
  >							expr_type;
  return XprVector<expr_type, Rows>(expr_type(m.const_ref(), no));
}


/**
 * \fn diag(const Matrix<T, Sz, Sz>& m)
 * \brief Returns the diagonal vector of the given square matrix.
 * \ingroup _unary_function
 */
template<class T, int Sz>
inline
XprVector<
  XprMatrixDiag<
    MatrixConstReference<T, Sz, Sz>,
    Sz
  >,
  Sz
>
diag(const Matrix<T, Sz, Sz>& m) {
  typedef XprMatrixDiag<
    MatrixConstReference<T, Sz, Sz>,
    Sz
  >							expr_type;
  return XprVector<expr_type, Sz>(expr_type(m.const_ref()));
}


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * min/max unary functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/**
 * \fn maximum(const XprMatrix<E, Rows, Cols>& e)
 * \brief Find the maximum of a matrix expression
 * \ingroup _unary_function
 */
template<class E, int Rows, int Cols>
inline
Extremum<typename E::value_type, int, matrix_tag>
maximum(const XprMatrix<E, Rows, Cols>& e) {
  typedef typename E::value_type 			value_type;

  value_type 						temp(e(0, 0));
  int 						row_no(0), col_no(0);

  for(int i = 0; i != Rows; ++i) {
    for(int j = 0; j != Cols; ++j) {
      if(e(i, j) > temp) {
	temp = e(i, j);
	row_no = i;
	col_no = j;
      }
    }
  }

  return Extremum<value_type, int, matrix_tag>(temp, row_no, col_no);
}


/**
 * \fn maximum(const Matrix<T, Rows, Cols>& m)
 * \brief Find the maximum of a matrix
 * \ingroup _unary_function
 */
template<class T, int Rows, int Cols>
inline
Extremum<T, int, matrix_tag>
maximum(const Matrix<T, Rows, Cols>& m) { return maximum(m.as_expr()); }


/**
 * \fn minimum(const XprMatrix<E, Rows, Cols>& e)
 * \brief Find the minimum of a matrix expression
 * \ingroup _unary_function
 */
template<class E, int Rows, int Cols>
inline
Extremum<typename E::value_type, int, matrix_tag>
minimum(const XprMatrix<E, Rows, Cols>& e) {
  typedef typename E::value_type 			value_type;

  value_type 						temp(e(0, 0));
  int 						row_no(0), col_no(0);

  for(int i = 0; i != Rows; ++i) {
    for(int j = 0; j != Cols; ++j) {
      if(e(i, j) < temp) {
	temp = e(i, j);
	row_no = i;
	col_no = j;
      }
    }
  }

  return Extremum<value_type, int, matrix_tag>(temp, row_no, col_no);
}


/**
 * \fn minimum(const Matrix<T, Rows, Cols>& m)
 * \brief Find the minimum of a matrix
 * \ingroup _unary_function
 */
template<class T, int Rows, int Cols>
inline
Extremum<T, int, matrix_tag>
minimum(const Matrix<T, Rows, Cols>& m) { return minimum(m.as_expr()); }


/**
 * \fn max(const XprMatrix<E, Rows, Cols>& e)
 * \brief Find the maximum of a matrix expression
 * \ingroup _unary_function
 */
template<class E, int Rows, int Cols>
inline
typename E::value_type
max(const XprMatrix<E, Rows, Cols>& e) {
  typedef typename E::value_type 			value_type;

  value_type 						temp(e(0, 0));

  for(int i = 0; i != Rows; ++i)
    for(int j = 0; j != Cols; ++j)
      if(e(i, j) > temp)
	temp = e(i, j);

  return temp;
}


/**
 * \fn max(const Matrix<T, Rows, Cols>& m)
 * \brief Find the maximum of a matrix
 * \ingroup _unary_function
 */
template<class T, int Rows, int Cols>
inline
T max(const Matrix<T, Rows, Cols>& m) {
  typedef T			 			value_type;
  typedef typename Matrix<
   T, Rows, Cols
  >::const_iterator					const_iterator;

  const_iterator					iter(m.begin());
  const_iterator					last(m.end());
  value_type 						temp(*iter);

  for( ; iter != last; ++iter)
    if(*iter > temp)
      temp = *iter;

  return temp;
}


/**
 * \fn min(const XprMatrix<E, Rows, Cols>& e)
 * \brief Find the minimum of a matrix expression
 * \ingroup _unary_function
 */
template<class E, int Rows, int Cols>
inline
typename E::value_type
min(const XprMatrix<E, Rows, Cols>& e) {
  typedef typename E::value_type			value_type;

  value_type 						temp(e(0, 0));

  for(int i = 0; i != Rows; ++i)
    for(int j = 0; j != Cols; ++j)
      if(e(i, j) < temp)
	temp = e(i, j);

  return temp;
}


/**
 * \fn min(const Matrix<T, Rows, Cols>& m)
 * \brief Find the minimum of a matrix
 * \ingroup _unary_function
 */
template<class T, int Rows, int Cols>
inline
T min(const Matrix<T, Rows, Cols>& m) {
  typedef T			 			value_type;
  typedef typename Matrix<
   T, Rows, Cols
  >::const_iterator					const_iterator;

  const_iterator					iter(m.begin());
  const_iterator					last(m.end());
  value_type 						temp(*iter);

  for( ; iter != last; ++iter)
    if(*iter < temp)
      temp = *iter;

  return temp;
}


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * other unary functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/**
 * \fn XprMatrix<XprIdentity<typename M::value_type, M::Rows, M::Cols>, M::Rows, M::Cols>identity()
 * \brief Fill a matrix to an identity matrix.
 * \ingroup _unary_function
 *
 * \note The matrix doesn't need to be square. Only the elements
 *       where the current number of rows are equal to columns
 *       will be set to 1, else to 0.
 *
 * \par Usage:
 * \code
 * typedef Matrix<double,3,3>		matrix_type;
 * ...
 * matrix_type E( identity<double, 3, 3>() );
 * \endcode
 *
 * Note, we have to specify the type, number of rows and columns
 * since ADL can't work here.
 *
 *
 *
 * \since release 1.6.0
 */
template<class T, int Rows, int Cols>
inline
XprMatrix<
  XprIdentity<T, Rows, Cols>,
  Rows, Cols
>
identity() {
  typedef XprIdentity<T, Rows, Cols>		expr_type;

  return XprMatrix<expr_type, Rows, Cols>(expr_type());
}

/**
 * \fn XprMatrix<XprIdentity<typename M::value_type, M::Rows, M::Cols>, M::Rows, M::Cols>identity()
 * \brief Fill a matrix to an identity matrix (convenience wrapper
 *        for matrix typedefs).
 * \ingroup _unary_function
 *
 * \note The matrix doesn't need to be square. Only the elements
 *       where the current number of rows are equal to columns
 *       will be set to 1, else to 0.
 *
 * \par Usage:
 * \code
 * typedef Matrix<double,3,3>		matrix_type;
 * ...
 * matrix_type E( identity<matrix_type>() );
 * \endcode
 *
 * Note, we have to specify the matrix type, since ADL can't work here.
 *
 * \since release 1.6.0
 */
template<class M>
inline
XprMatrix<
  XprIdentity<
    typename M::value_type,
    M::Rows, M::Cols>,
  M::Rows, M::Cols
>
identity() {
  return identity<typename M::value_type, M::Rows, M::Cols>();
}


/**
 * \fn cmatrix_ref(const T* mem)
 * \brief Creates an expression wrapper for a C like matrices.
 * \ingroup _unary_function
 *
 * This is like creating a matrix of external data, as described
 * at \ref construct. With this function you wrap an expression
 * around a C style matrix and you can operate directly with it
 * as usual.
 *
 * \par Example:
 * \code
 * static float lhs[3][3] = {
 *   {-1,  0,  1}, { 1,  0,  1}, {-1,  0, -1}
 * };
 * static float rhs[3][3] = {
 *   { 0,  1,  1}, { 0,  1, -1}, { 0, -1,  1}
 * };
 * ...
 *
 * typedef Matrix<float, 3, 3>			matrix_type;
 *
 * matrix_type M( cmatrix_ref<float, 3, 3>(&lhs[0][0])
 *                *  cmatrix_ref<float, 3, 3>(&rhs[0][0]) );
 * \endcode
 *
 * \since release 1.6.0
 */
template<class T, int Rows, int Cols>
inline
XprMatrix<
  MatrixConstReference<T, Rows, Cols>,
  Rows, Cols
>
cmatrix_ref(const T* mem) {
  typedef MatrixConstReference<T, Rows, Cols>	expr_type;

  return XprMatrix<expr_type, Rows, Cols>(expr_type(mem));
};


} // namespace tvmet

#endif // TVMET_MATRIX_FUNCTIONS_H

// Local Variables:
// mode:C++
// End:
