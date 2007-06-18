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
 * $Id: MatrixOperators.h,v 1.33 2004/06/17 15:53:12 opetzold Exp $
 */

#ifndef TVMET_MATRIX_OPERATORS_H
#define TVMET_MATRIX_OPERATORS_H

namespace tvmet {


/*********************************************************
 * PART I: DECLARATION
 *********************************************************/


template<class T, int Rows, int Cols>
std::ostream& operator<<(std::ostream& os,
			 const Matrix<T, Rows, Cols>& rhs) TVMET_CXX_ALWAYS_INLINE;


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Member operators (arithmetic and bit ops)
 *++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * update_operator(Matrix<T1, Rows, Cols>, Matrix<T2, Rows, Cols>)
 * update_operator(Matrix<T1, Rows, Cols>, XprMatrix<E, Rows, Cols> rhs)
 * Note: per se element wise
 * \todo: the operator*= can have element wise mul oder product, decide!
 */
#define TVMET_DECLARE_MACRO(NAME, OP)						\
template<class T1, class T2, int Rows, int Cols>		\
Matrix<T1, Rows, Cols>&								\
operator OP (Matrix<T1, Rows, Cols>& lhs, 					\
	     const Matrix<T2, Rows, Cols>& rhs) TVMET_CXX_ALWAYS_INLINE;	\
										\
template<class T, class E, int Rows,  int Cols>			\
Matrix<T, Rows, Cols>&								\
operator OP (Matrix<T, Rows, Cols>& lhs, 					\
	     const XprMatrix<E, Rows, Cols>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(add_eq, +=)		// per se element wise
TVMET_DECLARE_MACRO(sub_eq, -=)		// per se element wise
namespace element_wise {
  TVMET_DECLARE_MACRO(mul_eq, *=)		// see note
  TVMET_DECLARE_MACRO(div_eq, /=)		// not defined for vectors
}

#undef TVMET_DECLARE_MACRO


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Matrix arithmetic operators implemented by functions
 * add, sub, mul and div
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * operator(Matrix<T1, Rows, Cols>, Matrix<T2, Rows, Cols>)
 * operator(XprMatrix<E, Rows, Cols>, Matrix<T, Rows, Cols>)
 * operator(Matrix<T, Rows, Cols>, XprMatrix<E, Rows, Cols>)
 * Note: per se element wise
 */
#define TVMET_DECLARE_MACRO(NAME, OP)						\
template<class T1, class T2, int Rows, int Cols>		\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME<T1, T2>,							\
    MatrixConstReference<T1, Rows, Cols>,					\
    MatrixConstReference<T2, Rows, Cols>					\
  >,										\
  Rows, Cols									\
>										\
operator OP (const Matrix<T1, Rows, Cols>& lhs,					\
	     const Matrix<T2, Rows, Cols>& rhs) TVMET_CXX_ALWAYS_INLINE;	\
										\
template<class E, class T, int Rows, int Cols>			\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME<typename E::value_type, T>,					\
    XprMatrix<E, Rows, Cols>,							\
    MatrixConstReference<T, Rows, Cols>						\
  >,										\
  Rows, Cols									\
>										\
operator OP (const XprMatrix<E, Rows, Cols>& lhs, 				\
	     const Matrix<T, Rows, Cols>& rhs) TVMET_CXX_ALWAYS_INLINE;		\
										\
template<class T, class E, int Rows, int Cols>			\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME<typename E::value_type, T>,					\
    MatrixConstReference<T, Rows, Cols>,					\
    XprMatrix<E, Rows, Cols>							\
  >,										\
  Rows, Cols									\
>										\
operator OP (const Matrix<T, Rows, Cols>& lhs, 					\
	     const XprMatrix<E, Rows, Cols>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(add, +)			// per se element wise
TVMET_DECLARE_MACRO(sub, -)			// per se element wise
namespace element_wise {
  TVMET_DECLARE_MACRO(mul, *)			// see as prod()
  TVMET_DECLARE_MACRO(div, /)			// not defined for matrizes
}
#undef TVMET_DECLARE_MACRO


/*
 * operator(Matrix<T, Rows, Cols>, POD)
 * operator(POD, Matrix<T, Rows, Cols>)
 * Note: operations +,-,*,/ are per se element wise
 */
#define TVMET_DECLARE_MACRO(NAME, OP, POD)				\
template<class T, int Rows, int Cols>			\
XprMatrix<								\
  XprBinOp<								\
    Fcnl_##NAME<T, POD >,						\
    MatrixConstReference<T, Rows, Cols>,				\
    XprLiteral<POD >							\
  >,									\
  Rows, Cols								\
>									\
operator OP (const Matrix<T, Rows, Cols>& lhs, 				\
	     POD rhs) TVMET_CXX_ALWAYS_INLINE;				\
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
operator OP (POD lhs, 							\
	     const Matrix<T, Rows, Cols>& rhs) TVMET_CXX_ALWAYS_INLINE;

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


#if defined(TVMET_HAVE_COMPLEX)
/*
 * operator(Matrix<T, Rows, Cols>, complex<T>)
 * operator(complex<T>, Matrix<T, Rows, Cols>)
 * Note: operations +,-,*,/ are per se element wise
 * \todo type promotion
 */
#define TVMET_DECLARE_MACRO(NAME, OP)							\
template<class T, int Rows, int Cols>					\
XprMatrix<										\
  XprBinOp<										\
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,					\
    MatrixConstReference< std::complex<T>, Rows, Cols>,					\
    XprLiteral<std::complex<T> >							\
  >,											\
  Rows, Cols										\
>											\
operator OP (const Matrix< std::complex<T>, Rows, Cols>& lhs,				\
	     const std::complex<T>& rhs) TVMET_CXX_ALWAYS_INLINE;			\
											\
template<class T, int Rows, int Cols>					\
XprMatrix<										\
  XprBinOp<										\
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,					\
    XprLiteral< std::complex<T> >,							\
    MatrixConstReference< std::complex<T>, Rows, Cols>					\
  >,											\
  Rows, Cols										\
>											\
operator OP (const std::complex<T>& lhs,						\
	     const Matrix< std::complex<T>, Rows, Cols>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(add, +)
TVMET_DECLARE_MACRO(sub, -)
TVMET_DECLARE_MACRO(mul, *)
TVMET_DECLARE_MACRO(div, /)

#undef TVMET_DECLARE_MACRO

#endif // defined(TVMET_HAVE_COMPLEX)


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix specific operator*() = prod() operations
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


template<class T1, int Rows1, int Cols1,
	 class T2, int Cols2>
XprMatrix<
  XprMMProduct<
    MatrixConstReference<T1, Rows1, Cols1>, Rows1, Cols1,
    MatrixConstReference<T2, Cols1, Cols2>, Cols2
  >,
  Rows1, Cols2
>
operator*(const Matrix<T1, Rows1, Cols1>& lhs,
	  const Matrix<T2, Cols1, Cols2>& rhs) TVMET_CXX_ALWAYS_INLINE;


template<class E1, int Rows1, int Cols1,
	 class T2, int Cols2>
XprMatrix<
  XprMMProduct<
    XprMatrix<E1, Rows1, Cols1>, Rows1, Cols1,
    MatrixConstReference<T2, Cols1, Cols2>, Cols2
  >,
  Rows1, Cols2
>
operator*(const XprMatrix<E1, Rows1, Cols1>& lhs,
	  const Matrix<T2, Cols1, Cols2>& rhs) TVMET_CXX_ALWAYS_INLINE;


template<class T1, int Rows1, int Cols1,
	 class E2, int Cols2>
XprMatrix<
  XprMMProduct<
    MatrixConstReference<T1, Rows1, Cols1>, Rows1, Cols1,
    XprMatrix<E2, Cols1, Cols2>, Cols2
  >,
  Rows1, Cols2
>
operator*(const Matrix<T1, Rows1, Cols1>& lhs,
	  const XprMatrix<E2, Cols1, Cols2>& rhs) TVMET_CXX_ALWAYS_INLINE;


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix-vector specific prod( ... ) operators
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


template<class T1, int Rows, int Cols, class T2>
XprVector<
  XprMVProduct<
    MatrixConstReference<T1, Rows, Cols>, Rows, Cols,
    VectorConstReference<T2, Cols>
  >,
  Rows
>
operator*(const Matrix<T1, Rows, Cols>& lhs,
	  const Vector<T2, Cols>& rhs) TVMET_CXX_ALWAYS_INLINE;


template<class T1, class E2, int Rows, int Cols>
XprVector<
  XprMVProduct<
    MatrixConstReference<T1, Rows, Cols>, Rows, Cols,
    XprVector<E2, Cols>
  >,
  Rows
>
operator*(const Matrix<T1, Rows, Cols>& lhs,
	  const XprVector<E2, Cols>& rhs) TVMET_CXX_ALWAYS_INLINE;


template<class E1, class T2, int Rows, int Cols>
XprVector<
  XprMVProduct<
    XprMatrix<E1, Rows, Cols>, Rows, Cols,
    VectorConstReference<T2, Cols>
  >,
  Rows
>
operator*(const XprMatrix<E1, Rows, Cols>& lhs,
	  const Vector<T2, Cols>& rhs) TVMET_CXX_ALWAYS_INLINE;


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * global unary operators
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * unary_operator(Matrix<T, Rows, Cols>)
 * Note: per se element wise
 */
#define TVMET_DECLARE_MACRO(NAME, OP)					\
template <class T, int Rows, int Cols>			\
XprMatrix<								\
  XprUnOp<								\
    Fcnl_##NAME<T>,							\
    MatrixConstReference<T, Rows, Cols>					\
  >,									\
  Rows, Cols								\
>									\
operator OP (const Matrix<T, Rows, Cols>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(neg, -)
#undef TVMET_DECLARE_MACRO


/*********************************************************
 * PART II: IMPLEMENTATION
 *********************************************************/


/**
 * \fn operator<<(std::ostream& os, const Matrix<T, Rows, Cols>& rhs)
 * \brief Overload operator for i/o
 * \ingroup _binary_operator
 */
template<class T, int Rows, int Cols>
inline
std::ostream& operator<<(std::ostream& os, const Matrix<T, Rows, Cols>& rhs) {
  return rhs.print_on(os);
}


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Member operators (arithmetic and bit ops)
 *++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * update_operator(Matrix<T1, Rows, Cols>, Matrix<T2, Rows, Cols>)
 * update_operator(Matrix<T1, Rows, Cols>, XprMatrix<E, Rows, Cols> rhs)
 * Note: per se element wise
 * \todo: the operator*= can have element wise mul oder product, decide!
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)						\
template<class T1, class T2, int Rows, int Cols>		\
inline 										\
Matrix<T1, Rows, Cols>&								\
operator OP (Matrix<T1, Rows, Cols>& lhs, const Matrix<T2, Rows, Cols>& rhs) {	\
  return lhs.M_##NAME(rhs);							\
}										\
										\
template<class T, class E, int Rows,  int Cols>			\
inline 										\
Matrix<T, Rows, Cols>&								\
operator OP (Matrix<T, Rows, Cols>& lhs, const XprMatrix<E, Rows, Cols>& rhs) {	\
  return lhs.M_##NAME(rhs);							\
}

TVMET_IMPLEMENT_MACRO(add_eq, +=)		// per se element wise
TVMET_IMPLEMENT_MACRO(sub_eq, -=)		// per se element wise
namespace element_wise {
  TVMET_IMPLEMENT_MACRO(mul_eq, *=)		// see note
  TVMET_IMPLEMENT_MACRO(div_eq, /=)		// not defined for vectors
}

#undef TVMET_IMPLEMENT_MACRO


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Matrix arithmetic operators implemented by functions
 * add, sub, mul and div
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * operator(Matrix<T1, Rows, Cols>, Matrix<T2, Rows, Cols>)
 * operator(XprMatrix<E, Rows, Cols>, Matrix<T, Rows, Cols>)
 * operator(Matrix<T, Rows, Cols>, XprMatrix<E, Rows, Cols>)
 * Note: per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)						      \
template<class T1, class T2, int Rows, int Cols>		      \
inline										      \
XprMatrix<									      \
  XprBinOp<									      \
    Fcnl_##NAME<T1, T2>,							      \
    MatrixConstReference<T1, Rows, Cols>,					      \
    MatrixConstReference<T2, Rows, Cols>					      \
  >,										      \
  Rows, Cols									      \
>										      \
operator OP (const Matrix<T1, Rows, Cols>& lhs,	const Matrix<T2, Rows, Cols>& rhs) {  \
  return NAME(lhs, rhs);							      \
}										      \
										      \
template<class E, class T, int Rows, int Cols>			      \
inline										      \
XprMatrix<									      \
  XprBinOp<									      \
    Fcnl_##NAME<typename E::value_type, T>,					      \
    XprMatrix<E, Rows, Cols>,							      \
    MatrixConstReference<T, Rows, Cols>						      \
  >,										      \
  Rows, Cols									      \
>										      \
operator OP (const XprMatrix<E, Rows, Cols>& lhs, const Matrix<T, Rows, Cols>& rhs) { \
  return NAME(lhs, rhs);							      \
}										      \
										      \
template<class T, class E, int Rows, int Cols>			      \
inline										      \
XprMatrix<									      \
  XprBinOp<									      \
    Fcnl_##NAME<typename E::value_type, T>,					      \
    MatrixConstReference<T, Rows, Cols>,					      \
    XprMatrix<E, Rows, Cols>							      \
  >,										      \
  Rows, Cols									      \
>										      \
operator OP (const Matrix<T, Rows, Cols>& lhs, const XprMatrix<E, Rows, Cols>& rhs) { \
  return NAME(lhs, rhs);							      \
}

TVMET_IMPLEMENT_MACRO(add, +)			// per se element wise
TVMET_IMPLEMENT_MACRO(sub, -)			// per se element wise
namespace element_wise {
  TVMET_IMPLEMENT_MACRO(mul, *)			// see as prod()
  TVMET_IMPLEMENT_MACRO(div, /)			// not defined for matrizes
}
#undef TVMET_IMPLEMENT_MACRO


/*
 * operator(Matrix<T, Rows, Cols>, POD)
 * operator(POD, Matrix<T, Rows, Cols>)
 * Note: operations +,-,*,/ are per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP, POD)			\
template<class T, int Rows, int Cols>		\
inline								\
XprMatrix<							\
  XprBinOp<							\
    Fcnl_##NAME<T, POD >,					\
    MatrixConstReference<T, Rows, Cols>,			\
    XprLiteral<POD >						\
  >,								\
  Rows, Cols							\
>								\
operator OP (const Matrix<T, Rows, Cols>& lhs, POD rhs) {	\
  return NAME (lhs, rhs);					\
}								\
								\
template<class T, int Rows, int Cols>		\
inline								\
XprMatrix<							\
  XprBinOp<							\
    Fcnl_##NAME< POD, T>,					\
    XprLiteral< POD >,						\
    MatrixConstReference<T, Rows, Cols>				\
  >,								\
  Rows, Cols							\
>								\
operator OP (POD lhs, const Matrix<T, Rows, Cols>& rhs) {	\
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


#if defined(TVMET_HAVE_COMPLEX)
/*
 * operator(Matrix<T, Rows, Cols>, complex<T>)
 * operator(complex<T>, Matrix<T, Rows, Cols>)
 * Note: operations +,-,*,/ are per se element wise
 * \todo type promotion
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)				\
template<class T, int Rows, int Cols>		\
inline								\
XprMatrix<							\
  XprBinOp<							\
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,		\
    MatrixConstReference< std::complex<T>, Rows, Cols>,		\
    XprLiteral<std::complex<T> >				\
  >,								\
  Rows, Cols							\
>								\
operator OP (const Matrix< std::complex<T>, Rows, Cols>& lhs,	\
	     const std::complex<T>& rhs) {			\
  return NAME (lhs, rhs);					\
}								\
								\
template<class T, int Rows, int Cols>		\
inline								\
XprMatrix<							\
  XprBinOp<							\
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,		\
    XprLiteral< std::complex<T> >,				\
    MatrixConstReference< std::complex<T>, Rows, Cols>		\
  >,								\
  Rows, Cols							\
>								\
operator OP (const std::complex<T>& lhs,			\
	     const Matrix< std::complex<T>, Rows, Cols>& rhs) {	\
  return NAME (lhs, rhs);					\
}

TVMET_IMPLEMENT_MACRO(add, +)
TVMET_IMPLEMENT_MACRO(sub, -)
TVMET_IMPLEMENT_MACRO(mul, *)
TVMET_IMPLEMENT_MACRO(div, /)

#undef TVMET_IMPLEMENT_MACRO

#endif // defined(TVMET_HAVE_COMPLEX)


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix specific operator*() = prod() operations
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/**
 * \fn operator*(const Matrix<T1, Rows1, Cols1>& lhs, const Matrix<T2, Cols1, Cols2>& rhs)
 * \brief multiply two Matrices.
 * \ingroup _binary_operator
 * \note The rows2 has to be equal to cols1.
 * \sa prod(const Matrix<T1, Rows1, Cols1>& lhs, const Matrix<T2, Cols1, Cols2>& rhs)
 */
template<class T1, int Rows1, int Cols1,
	 class T2, int Cols2>
inline
XprMatrix<
  XprMMProduct<
    MatrixConstReference<T1, Rows1, Cols1>, Rows1, Cols1,
    MatrixConstReference<T2, Cols1, Cols2>, Cols2
  >,
  Rows1, Cols2
>
operator*(const Matrix<T1, Rows1, Cols1>& lhs, const Matrix<T2, Cols1, Cols2>& rhs) {
  return prod(lhs, rhs);
}


/**
 * \fn operator*(const XprMatrix<E1, Rows1, Cols1>& lhs, const Matrix<T2, Cols1, Cols2>& rhs)
 * \brief Evaluate the product of XprMatrix and Matrix.
 * \ingroup _binary_operator
 * \sa prod(const XprMatrix<E1, Rows1, Cols1>& lhs, const Matrix<T2, Cols1, Cols2>& rhs)
 */
template<class E1, int Rows1, int Cols1,
	 class T2, int Cols2>
inline
XprMatrix<
  XprMMProduct<
    XprMatrix<E1, Rows1, Cols1>, Rows1, Cols1,
    MatrixConstReference<T2, Cols1, Cols2>, Cols2
  >,
  Rows1, Cols2
>
operator*(const XprMatrix<E1, Rows1, Cols1>& lhs, const Matrix<T2, Cols1, Cols2>& rhs) {
  return prod(lhs, rhs);
}


/**
 * \fn operator*(const Matrix<T1, Rows1, Cols1>& lhs, const XprMatrix<E2, Cols1, Cols2>& rhs)
 * \brief Evaluate the product of Matrix and XprMatrix.
 * \ingroup _binary_operator
 * \sa prod(const Matrix<T, Rows1, Cols1>& lhs, const XprMatrix<E, Cols1, Cols2>& rhs)
 */
template<class T1, int Rows1, int Cols1,
	 class E2, int Cols2>
inline
XprMatrix<
  XprMMProduct<
    MatrixConstReference<T1, Rows1, Cols1>, Rows1, Cols1,
    XprMatrix<E2, Cols1, Cols2>, Cols2
  >,
  Rows1, Cols2
>
operator*(const Matrix<T1, Rows1, Cols1>& lhs, const XprMatrix<E2, Cols1, Cols2>& rhs) {
  return prod(lhs, rhs);
}


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * matrix-vector specific prod( ... ) operators
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/**
 * \fn operator*(const Matrix<T1, Rows, Cols>& lhs, const Vector<T2, Cols>& rhs)
 * \brief multiply a Matrix with a Vector.
 * \ingroup _binary_operator
 * \note The length of the Vector has to be equal to the number of Columns.
 * \sa prod(const Matrix<T1, Rows, Cols>& m, const Vector<T2, Cols>& v)
 */
template<class T1, int Rows, int Cols, class T2>
inline
XprVector<
  XprMVProduct<
    MatrixConstReference<T1, Rows, Cols>, Rows, Cols,
    VectorConstReference<T2, Cols>
  >,
  Rows
>
operator*(const Matrix<T1, Rows, Cols>& lhs, const Vector<T2, Cols>& rhs) {
  return prod(lhs, rhs);
}


/**
 * \fn operator*(const Matrix<T1, Rows, Cols>& lhs, const XprVector<E2, Cols>& rhs)
 * \brief Function for the matrix-vector-product
 * \ingroup _binary_operator
 * \sa prod(const Matrix<T, Rows, Cols>& lhs, const XprVector<E, Cols>& rhs)
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
operator*(const Matrix<T1, Rows, Cols>& lhs, const XprVector<E2, Cols>& rhs) {
  return prod(lhs, rhs);
}


/**
 * \fn operator*(const XprMatrix<E1, Rows, Cols>& lhs, const Vector<T2, Cols>& rhs)
 * \brief Compute the product of an XprMatrix with a Vector.
 * \ingroup _binary_operator
 * \sa prod(const XprMatrix<E, Rows, Cols>& lhs, const Vector<T, Cols>& rhs)
 */
template<class E1, class T2, int Rows, int Cols>
inline
XprVector<
  XprMVProduct<
    XprMatrix<E1, Rows, Cols>, Rows, Cols,
    VectorConstReference<T2, Cols>
  >,
  Rows
>
operator*(const XprMatrix<E1, Rows, Cols>& lhs, const Vector<T2, Cols>& rhs) {
  return prod(lhs, rhs);
}


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * global unary operators
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * unary_operator(Matrix<T, Rows, Cols>)
 * Note: per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)				       \
template <class T, int Rows, int Cols>		       \
inline								       \
XprMatrix<							       \
  XprUnOp<							       \
    Fcnl_##NAME<T>,						       \
    MatrixConstReference<T, Rows, Cols>				       \
  >,								       \
  Rows, Cols							       \
>								       \
operator OP (const Matrix<T, Rows, Cols>& rhs) {		       \
  typedef XprUnOp<						       \
    Fcnl_##NAME<T>,						       \
    MatrixConstReference<T, Rows, Cols>				       \
  >  							 expr_type;    \
  return XprMatrix<expr_type, Rows, Cols>(expr_type(rhs.const_ref())); \
}

TVMET_IMPLEMENT_MACRO(neg, -)
#undef TVMET_IMPLEMENT_MACRO


} // namespace tvmet

#endif // TVMET_MATRIX_OPERATORS_H

// Local Variables:
// mode:C++
// End:
