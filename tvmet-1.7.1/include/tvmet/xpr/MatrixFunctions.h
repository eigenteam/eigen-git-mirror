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
 * $Id: MatrixFunctions.h,v 1.39 2004/07/06 05:49:22 opetzold Exp $
 */

#ifndef TVMET_XPR_MATRIX_FUNCTIONS_H
#define TVMET_XPR_MATRIX_FUNCTIONS_H

namespace tvmet {


/* forwards */
template<class T, int Rows, int Cols> class Matrix;
template<class T, int Sz> class Vector;
template<class E, int Sz> class XprVector;
template<class E> class XprMatrixTranspose;
template<class E, int Sz> class XprMatrixDiag;
template<class E, int Rows, int Cols> class XprMatrixRow;
template<class E, int Rows, int Cols> class XprMatrixCol;


/*********************************************************
 * PART I: DECLARATION
 *********************************************************/


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Matrix arithmetic functions add, sub, mul and div
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * function(XprMatrix<E1, Rows, Cols>, XprMatrix<E2, Rows, Cols>)
 */
#define TVMET_DECLARE_MACRO(NAME)					\
template<class E1, class E2, int Rows, int Cols>	\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E1::value_type, typename E2::value_type>,	\
    XprMatrix<E1, Rows, Cols>,						\
    XprMatrix<E2, Rows, Cols>						\
  >,									\
  Rows, Cols								\
>									\
NAME (const XprMatrix<E1, Rows, Cols>& lhs,				\
      const XprMatrix<E2, Rows, Cols>& rhs) _tvmet_always_inline;

TVMET_DECLARE_MACRO(add)			// per se element wise
TVMET_DECLARE_MACRO(sub)			// per se element wise
namespace element_wise {
  TVMET_DECLARE_MACRO(mul)			// not defined for matrizes
  TVMET_DECLARE_MACRO(div)			// not defined for matrizes
}

#undef TVMET_DECLARE_MACRO


/*
 * function(XprMatrix<E, Rows, Cols>, POD)
 * function(POD, XprMatrix<E, Rows, Cols>)
 * Note: - operations +,-,*,/ are per se element wise
 */
#define TVMET_DECLARE_MACRO(NAME, POD)					\
template<class E, int Rows, int Cols>			\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, POD >,				\
    XprMatrix<E, Rows, Cols>,						\
    XprLiteral< POD >							\
  >,									\
  Rows, Cols								\
>									\
NAME (const XprMatrix<E, Rows, Cols>& lhs, 				\
      POD rhs) _tvmet_always_inline;					\
									\
template<class E, int Rows, int Cols>			\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME< POD, typename E::value_type>,				\
    XprLiteral< POD >,							\
    XprMatrix<E, Rows, Cols>						\
  >,									\
  Rows, Cols								\
>									\
NAME (POD lhs, 								\
      const XprMatrix<E, Rows, Cols>& rhs) _tvmet_always_inline;

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
#define TVMET_DECLARE_MACRO(NAME)					\
template<class E, class T, int Rows, int Cols>		\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, std::complex<T> >,		\
    XprMatrix<E, Rows, Cols>,						\
    XprLiteral< std::complex<T> >					\
  >,									\
  Rows, Cols								\
>									\
NAME (const XprMatrix<E, Rows, Cols>& lhs,				\
      const std::complex<T>& rhs) _tvmet_always_inline;		\
									\
template<class T, class E, int Rows, int Cols>		\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME< std::complex<T>, typename E::value_type>,		\
    XprLiteral< std::complex<T> >,					\
    XprMatrix<E, Rows, Cols>						\
  >,									\
  Rows, Cols								\
>									\
NAME (const std::complex<T>& lhs,					\
      const XprMatrix<E, Rows, Cols>& rhs) _tvmet_always_inline;

TVMET_DECLARE_MACRO(add)
TVMET_DECLARE_MACRO(sub)
TVMET_DECLARE_MACRO(mul)
TVMET_DECLARE_MACRO(div)

#undef TVMET_DECLARE_MACRO

#endif // defined(EIGEN_USE_COMPLEX)


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix prod( ... ) functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


template<class E1, int Rows1, int Cols1,
	 class E2, int Cols2>
XprMatrix<
  XprMMProduct<
    XprMatrix<E1, Rows1, Cols1>, Rows1, Cols1,	// M1(Rows1, Cols1)
    XprMatrix<E2, Cols1, Cols2>, Cols2
  >,
  Rows1, Cols2					// return Dim
>
prod(const XprMatrix<E1, Rows1, Cols1>& lhs,
     const XprMatrix<E2, Cols1, Cols2>& rhs) _tvmet_always_inline;


template<class E1, int Rows1, int Cols1,
	 class E2, int Cols2>
XprMatrix<
  XprMMProductTransposed<
    XprMatrix<E1, Rows1, Cols1>, Rows1, Cols1,	// M1(Rows1, Cols1)
    XprMatrix<E2, Cols1, Cols2>, Cols2		// M2(Cols1, Cols2)
  >,
  Cols2, Rows1					// return Dim
>
trans_prod(const XprMatrix<E1, Rows1, Cols1>& lhs,
	   const XprMatrix<E2, Cols1, Cols2>& rhs) _tvmet_always_inline;


template<class E1, int Rows1, int Cols1,
	 class E2, int Cols2>	// Rows2 = Rows1
XprMatrix<
  XprMtMProduct<
    XprMatrix<E1, Rows1, Cols1>, Rows1, Cols1,	// M1(Rows1, Cols1)
    XprMatrix<E2, Rows1, Cols2>, Cols2		// M2(Rows1, Cols2)
  >,
  Cols1, Cols2					// return Dim
>
MtM_prod(const XprMatrix<E1, Rows1, Cols1>& lhs,
	 const XprMatrix<E2, Rows1, Cols2>& rhs) _tvmet_always_inline;


template<class E1, int Rows1, int Cols1,
	 class E2, int Rows2> 		// Cols2 = Cols1
XprMatrix<
  XprMMtProduct<
    XprMatrix<E1, Rows1, Cols1>, Rows1, Cols1,	// M1(Rows1, Cols1)
    XprMatrix<E2, Rows2, Cols1>, Cols1 		// M2(Rows2, Cols1)
  >,
  Rows1, Rows2					// return Dim
>
MMt_prod(const XprMatrix<E1, Rows1, Cols1>& lhs,
	 const XprMatrix<E2, Rows2, Cols1>& rhs) _tvmet_always_inline;


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix-vector specific prod( ... ) functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


template<class E1, int Rows, int Cols,
	 class E2>
XprVector<
  XprMVProduct<
    XprMatrix<E1, Rows, Cols>, Rows, Cols,
    XprVector<E2, Cols>
  >,
  Rows
>
prod(const XprMatrix<E1, Rows, Cols>& lhs,
     const XprVector<E2, Cols>& rhs) _tvmet_always_inline;


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix specific functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


template<class E, int Rows, int Cols>
XprMatrix<
  XprMatrixTranspose<
    XprMatrix<E, Rows, Cols>
  >,
  Cols, Rows
>
trans(const XprMatrix<E, Rows, Cols>& rhs) _tvmet_always_inline;


#if 0 // XXX needs declaration of meta::Matrix<Sz, Sz, 0, 0>::trace
template<class E, int Sz>
typename Traits<typename E::value_type>::sum_type
trace(const XprMatrix<E, Sz, Sz>& m)_tvmet_always_inline;
#endif


template<class E, int Rows, int Cols>
XprVector<
  XprMatrixRow<
    XprMatrix<E, Rows, Cols>,
    Rows, Cols
  >,
  Cols
>
row(const XprMatrix<E, Rows, Cols>& m,
    int no) _tvmet_always_inline;


template<class E, int Rows, int Cols>
XprVector<
  XprMatrixCol<
    XprMatrix<E, Rows, Cols>,
    Rows, Cols
  >,
  Rows
>
col(const XprMatrix<E, Rows, Cols>& m, int no) _tvmet_always_inline;


template<class E, int Sz>
XprVector<
  XprMatrixDiag<
    XprMatrix<E, Sz, Sz>,
    Sz
  >,
  Sz
>
diag(const XprMatrix<E, Sz, Sz>& m) _tvmet_always_inline;


/*********************************************************
 * PART II: IMPLEMENTATION
 *********************************************************/


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Matrix arithmetic functions add, sub, mul and div
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * function(XprMatrix<E1, Rows, Cols>, XprMatrix<E2, Rows, Cols>)
 */
#define TVMET_IMPLEMENT_MACRO(NAME)					\
template<class E1, class E2, int Rows, int Cols>	\
inline									\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E1::value_type, typename E2::value_type>,	\
    XprMatrix<E1, Rows, Cols>,						\
    XprMatrix<E2, Rows, Cols>						\
  >,									\
  Rows, Cols								\
>									\
NAME (const XprMatrix<E1, Rows, Cols>& lhs, 				\
      const XprMatrix<E2, Rows, Cols>& rhs) {				\
  typedef XprBinOp<							\
    Fcnl_##NAME<typename E1::value_type, typename E2::value_type>,	\
    XprMatrix<E1, Rows, Cols>,						\
    XprMatrix<E2, Rows, Cols>						\
  > 							 expr_type;	\
  return XprMatrix<expr_type, Rows, Cols>(expr_type(lhs, rhs));		\
}

TVMET_IMPLEMENT_MACRO(add)			// per se element wise
TVMET_IMPLEMENT_MACRO(sub)			// per se element wise
namespace element_wise {
  TVMET_IMPLEMENT_MACRO(mul)			// not defined for matrizes
  TVMET_IMPLEMENT_MACRO(div)			// not defined for matrizes
}

#undef TVMET_IMPLEMENT_MACRO


/*
 * function(XprMatrix<E, Rows, Cols>, POD)
 * function(POD, XprMatrix<E, Rows, Cols>)
 * Note: - operations +,-,*,/ are per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, POD)				\
template<class E, int Rows, int Cols>			\
inline									\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, POD >,				\
    XprMatrix<E, Rows, Cols>,						\
    XprLiteral< POD >							\
  >,									\
  Rows, Cols								\
>									\
NAME (const XprMatrix<E, Rows, Cols>& lhs, POD rhs) {			\
  typedef XprBinOp<							\
    Fcnl_##NAME<typename E::value_type, POD >,				\
    XprMatrix<E, Rows, Cols>,						\
    XprLiteral< POD >							\
  >							expr_type;	\
  return XprMatrix<expr_type, Rows, Cols>(				\
    expr_type(lhs, XprLiteral< POD >(rhs)));				\
}									\
									\
template<class E, int Rows, int Cols>			\
inline									\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME< POD, typename E::value_type>,				\
    XprLiteral< POD >,							\
    XprMatrix<E, Rows, Cols>						\
  >,									\
  Rows, Cols								\
>									\
NAME (POD lhs, const XprMatrix<E, Rows, Cols>& rhs) {			\
  typedef XprBinOp<							\
    Fcnl_##NAME< POD, typename E::value_type>,				\
    XprLiteral< POD >,							\
    XprMatrix<E, Rows, Cols>						\
  >							expr_type;	\
  return XprMatrix<expr_type, Rows, Cols>(				\
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
#define TVMET_IMPLEMENT_MACRO(NAME)					\
template<class E, class T, int Rows, int Cols>		\
inline									\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<typename E::value_type, std::complex<T> >,		\
    XprMatrix<E, Rows, Cols>,						\
    XprLiteral< std::complex<T> >					\
  >,									\
  Rows, Cols								\
>									\
NAME (const XprMatrix<E, Rows, Cols>& lhs, 				\
      const std::complex<T>& rhs) {					\
  typedef XprBinOp<							\
    Fcnl_##NAME<typename E::value_type, std::complex<T> >,		\
    XprMatrix<E, Rows, Cols>,						\
    XprLiteral< std::complex<T> >					\
  >							expr_type;	\
  return XprMatrix<expr_type, Rows, Cols>(				\
    expr_type(lhs, XprLiteral< std::complex<T> >(rhs)));		\
}									\
									\
template<class T, class E, int Rows, int Cols>		\
inline									\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME< std::complex<T>, typename E::value_type>,		\
    XprLiteral< std::complex<T> >,					\
    XprMatrix<E, Rows, Cols>						\
  >,									\
  Rows, Cols								\
>									\
NAME (const std::complex<T>& lhs, 					\
      const XprMatrix<E, Rows, Cols>& rhs) {				\
  typedef XprBinOp<							\
    Fcnl_##NAME< std::complex<T>, typename E::value_type>,		\
    XprLiteral< std::complex<T> >,					\
    XprMatrix<E, Rows, Cols>						\
  >							expr_type;	\
  return XprMatrix<expr_type, Rows, Cols>(				\
    expr_type(XprLiteral< std::complex<T> >(lhs), rhs));		\
}

TVMET_IMPLEMENT_MACRO(add)
TVMET_IMPLEMENT_MACRO(sub)
TVMET_IMPLEMENT_MACRO(mul)
TVMET_IMPLEMENT_MACRO(div)

#undef TVMET_IMPLEMENT_MACRO

#endif // defined(EIGEN_USE_COMPLEX)


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix prod( ... ) functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/**
 * \fn prod(const XprMatrix<E1, Rows1, Cols1>& lhs, const XprMatrix<E2, Cols1, Cols2>& rhs)
 * \brief Evaluate the product of two XprMatrix.
 * Perform on given Matrix M1 and M2:
 * \f[
 * M_1\,M_2
 * \f]
 * \note The numer of Rows2 has to be equal to Cols1.
 * \ingroup _binary_function
 */
template<class E1, int Rows1, int Cols1,
	 class E2, int Cols2>
inline
XprMatrix<
  XprMMProduct<
    XprMatrix<E1, Rows1, Cols1>, Rows1, Cols1,	// M1(Rows1, Cols1)
    XprMatrix<E2, Cols1, Cols2>, Cols2
  >,
  Rows1, Cols2					// return Dim
>
prod(const XprMatrix<E1, Rows1, Cols1>& lhs, const XprMatrix<E2, Cols1, Cols2>& rhs) {
  typedef XprMMProduct<
    XprMatrix<E1, Rows1, Cols1>, Rows1, Cols1,
    XprMatrix<E2, Cols1, Cols2>, Cols2
  >							expr_type;
  return XprMatrix<expr_type, Rows1, Cols2>(expr_type(lhs, rhs));
}


/**
 * \fn trans_prod(const XprMatrix<E1, Rows1, Cols1>& lhs, const XprMatrix<E2, Cols1, Cols2>& rhs)
 * \brief Function for the trans(matrix-matrix-product)
 * Perform on given Matrix M1 and M2:
 * \f[
 * (M_1\,M_2)^T
 * \f]
 * \note The numer of Rows2 has to be equal to Cols1.
 * \ingroup _binary_function
 */
template<class E1, int Rows1, int Cols1,
	 class E2, int Cols2>
inline
XprMatrix<
  XprMMProductTransposed<
    XprMatrix<E1, Rows1, Cols1>, Rows1, Cols1,	// M1(Rows1, Cols1)
    XprMatrix<E2, Cols1, Cols2>, Cols2		// M2(Cols1, Cols2)
  >,
  Cols2, Rows1					// return Dim
>
trans_prod(const XprMatrix<E1, Rows1, Cols1>& lhs, const XprMatrix<E2, Cols1, Cols2>& rhs) {
  typedef XprMMProductTransposed<
    XprMatrix<E1, Rows1, Cols1>, Rows1, Cols1,
    XprMatrix<E2, Cols1, Cols2>, Cols2
  >							expr_type;
  return XprMatrix<expr_type, Cols2, Rows1>(expr_type(lhs, rhs));
}


/**
 * \fn MtM_prod(const XprMatrix<E1, Rows1, Cols1>& lhs, const XprMatrix<E2, Rows1, Cols2>& rhs)
 * \brief Function for the trans(matrix)-matrix-product.
 *        using formula
 *        \f[
 *        M_1^{T}\,M_2
 *        \f]
 * \note The number of cols of matrix 2 have to be equal to number of rows of
 *       matrix 1, since matrix 1 is trans - the result is a (Cols1 x Cols2)
 *       matrix.
 * \ingroup _binary_function
 */
template<class E1, int Rows1, int Cols1,
	 class E2, int Cols2>	// Rows2 = Rows1
inline
XprMatrix<
  XprMtMProduct<
    XprMatrix<E1, Rows1, Cols1>, Rows1, Cols1,	// M1(Rows1, Cols1)
    XprMatrix<E2, Rows1, Cols2>, Cols2		// M2(Rows1, Cols2)
  >,
  Cols1, Cols2					// return Dim
>
MtM_prod(const XprMatrix<E1, Rows1, Cols1>& lhs, const XprMatrix<E2, Rows1, Cols2>& rhs) {
  typedef XprMtMProduct<
    XprMatrix<E1, Rows1, Cols1>, Rows1, Cols1,
    XprMatrix<E2, Rows1, Cols2>, Cols2
  >							expr_type;
  return XprMatrix<expr_type, Cols1, Cols2>(expr_type(lhs, rhs));
}


/**
 * \fn MMt_prod(const XprMatrix<E1, Rows1, Cols1>& lhs, const XprMatrix<E2, Rows2, Cols1>& rhs)
 * \brief Function for the matrix-trans(matrix)-product.
 * \ingroup _binary_function
 * \note The cols2 has to be equal to cols1.
 */
template<class E1, int Rows1, int Cols1,
	 class E2, int Rows2> // Cols2 = Cols1
inline
XprMatrix<
  XprMMtProduct<
    XprMatrix<E1, Rows1, Cols1>, Rows1, Cols1,	// M1(Rows1, Cols1)
    XprMatrix<E2, Rows2, Cols1>, Cols1	 	// M2(Rows2, Cols1)
  >,
  Rows1, Rows2					// return Dim
>
MMt_prod(const XprMatrix<E1, Rows1, Cols1>& lhs, const XprMatrix<E2, Rows2, Cols1>& rhs) {
  typedef XprMMtProduct<
    XprMatrix<E1, Rows1, Cols1>, Rows1, Cols1,
    XprMatrix<E2, Rows2, Cols1>, Cols1
  >							expr_type;
  return XprMatrix<expr_type, Rows1, Rows2>(expr_type(lhs, rhs));
}


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix-vector specific prod( ... ) functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/**
 * \fn prod(const XprMatrix<E1, Rows, Cols>& lhs, const XprVector<E2, Cols>& rhs)
 * \brief Evaluate the product of XprMatrix and XprVector.
 * \ingroup _binary_function
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
prod(const XprMatrix<E1, Rows, Cols>& lhs, const XprVector<E2, Cols>& rhs) {
  typedef XprMVProduct<
    XprMatrix<E1, Rows, Cols>, Rows, Cols,
    XprVector<E2, Cols>
  >							expr_type;
  return XprVector<expr_type, Rows>(expr_type(lhs, rhs));
}


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix specific functions
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/**
 * \fn trans(const XprMatrix<E, Rows, Cols>& rhs)
 * \brief Transpose an expression matrix.
 * \ingroup _unary_function
 */
template<class E, int Rows, int Cols>
inline
XprMatrix<
  XprMatrixTranspose<
    XprMatrix<E, Rows, Cols>
  >,
  Cols, Rows
>
trans(const XprMatrix<E, Rows, Cols>& rhs) {
  typedef XprMatrixTranspose<
    XprMatrix<E, Rows, Cols>
  >							expr_type;
  return XprMatrix<expr_type, Cols, Rows>(expr_type(rhs));
}


#if 0 // XXX needs declaration of meta::Matrix<Sz, Sz, 0, 0>::trace
/*
 * \fn trace(const XprMatrix<E, Sz, Sz>& m)
 * \brief Compute the trace of a square matrix.
 * \ingroup _unary_function
 *
 * Simply compute the trace of the given matrix as:
 * \f[
 *  \sum_{k = 0}^{Sz-1} m(k, k)
 * \f]
 */
template<class E, int Sz>
inline
typename Traits<typename E::value_type>::sum_type
trace(const XprMatrix<E, Sz, Sz>& m) {
  return meta::Matrix<Sz, Sz, 0, 0>::trace(m);
}
#endif


/**
 * \fn row(const XprMatrix<E, Rows, Cols>& m, int no)
 * \brief Returns a row vector of the given matrix.
 * \ingroup _binary_function
 */
template<class E, int Rows, int Cols>
inline
XprVector<
  XprMatrixRow<
    XprMatrix<E, Rows, Cols>,
    Rows, Cols
  >,
  Cols
>
row(const XprMatrix<E, Rows, Cols>& m, int no) {
  typedef XprMatrixRow<
    XprMatrix<E, Rows, Cols>,
    Rows, Cols
  >							expr_type;

  return XprVector<expr_type, Cols>(expr_type(m, no));
}


/**
 * \fn col(const XprMatrix<E, Rows, Cols>& m, int no)
 * \brief Returns a column vector of the given matrix.
 * \ingroup _binary_function
 */
template<class E, int Rows, int Cols>
inline
XprVector<
  XprMatrixCol<
    XprMatrix<E, Rows, Cols>,
    Rows, Cols
  >,
  Rows
>
col(const XprMatrix<E, Rows, Cols>& m, int no) {
  typedef XprMatrixCol<
    XprMatrix<E, Rows, Cols>,
    Rows, Cols
  >							expr_type;

  return XprVector<expr_type, Cols>(expr_type(m, no));
}


/**
 * \fn diag(const XprMatrix<E, Sz, Sz>& m)
 * \brief Returns the diagonal vector of the given square matrix.
 * \ingroup _unary_function
 */
template<class E, int Sz>
inline
XprVector<
  XprMatrixDiag<
    XprMatrix<E, Sz, Sz>,
    Sz
  >,
  Sz
>
diag(const XprMatrix<E, Sz, Sz>& m) {
  typedef XprMatrixDiag<
    XprMatrix<E, Sz, Sz>,
  Sz> 						expr_type;

  return XprVector<expr_type, Sz>(expr_type(m));
}


} // namespace tvmet

#endif // TVMET_XPR_MATRIX_FUNCTIONS_H

// Local Variables:
// mode:C++
// End:
