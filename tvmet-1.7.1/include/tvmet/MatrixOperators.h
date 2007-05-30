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


template<class T, std::size_t Rows, std::size_t Cols>
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
template<class T1, class T2, std::size_t Rows, std::size_t Cols>		\
Matrix<T1, Rows, Cols>&								\
operator OP (Matrix<T1, Rows, Cols>& lhs, 					\
	     const Matrix<T2, Rows, Cols>& rhs) TVMET_CXX_ALWAYS_INLINE;	\
										\
template<class T, class E, std::size_t Rows,  std::size_t Cols>			\
Matrix<T, Rows, Cols>&								\
operator OP (Matrix<T, Rows, Cols>& lhs, 					\
	     const XprMatrix<E, Rows, Cols>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(add_eq, +=)		// per se element wise
TVMET_DECLARE_MACRO(sub_eq, -=)		// per se element wise
namespace element_wise {
  TVMET_DECLARE_MACRO(mul_eq, *=)		// see note
  TVMET_DECLARE_MACRO(div_eq, /=)		// not defined for vectors
}

// integer operators only, e.g used on double you wil get an error
namespace element_wise {
  TVMET_DECLARE_MACRO(mod_eq, %=)
  TVMET_DECLARE_MACRO(xor_eq, ^=)
  TVMET_DECLARE_MACRO(and_eq, &=)
  TVMET_DECLARE_MACRO(or_eq, |=)
  TVMET_DECLARE_MACRO(shl_eq, <<=)
  TVMET_DECLARE_MACRO(shr_eq, >>=)
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
template<class T1, class T2, std::size_t Rows, std::size_t Cols>		\
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
template<class E, class T, std::size_t Rows, std::size_t Cols>			\
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
template<class T, class E, std::size_t Rows, std::size_t Cols>			\
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
template<class T, std::size_t Rows, std::size_t Cols>			\
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
template<class T, std::size_t Rows, std::size_t Cols>			\
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

#if defined(TVMET_HAVE_LONG_LONG)
TVMET_DECLARE_MACRO(add, +, long long int)
TVMET_DECLARE_MACRO(sub, -, long long int)
TVMET_DECLARE_MACRO(mul, *, long long int)
TVMET_DECLARE_MACRO(div, /, long long int)
#endif // defined(TVMET_HAVE_LONG_LONG)

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
#endif // defined(TVMET_HAVE_LONG_DOUBLE)

#undef TVMET_DECLARE_MACRO


#if defined(TVMET_HAVE_COMPLEX)
/*
 * operator(Matrix<T, Rows, Cols>, complex<T>)
 * operator(complex<T>, Matrix<T, Rows, Cols>)
 * Note: operations +,-,*,/ are per se element wise
 * \todo type promotion
 */
#define TVMET_DECLARE_MACRO(NAME, OP)							\
template<class T, std::size_t Rows, std::size_t Cols>					\
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
template<class T, std::size_t Rows, std::size_t Cols>					\
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


template<class T1, std::size_t Rows1, std::size_t Cols1,
	 class T2, std::size_t Cols2>
XprMatrix<
  XprMMProduct<
    MatrixConstReference<T1, Rows1, Cols1>, Rows1, Cols1,
    MatrixConstReference<T2, Cols1, Cols2>, Cols2
  >,
  Rows1, Cols2
>
operator*(const Matrix<T1, Rows1, Cols1>& lhs,
	  const Matrix<T2, Cols1, Cols2>& rhs) TVMET_CXX_ALWAYS_INLINE;


template<class E1, std::size_t Rows1, std::size_t Cols1,
	 class T2, std::size_t Cols2>
XprMatrix<
  XprMMProduct<
    XprMatrix<E1, Rows1, Cols1>, Rows1, Cols1,
    MatrixConstReference<T2, Cols1, Cols2>, Cols2
  >,
  Rows1, Cols2
>
operator*(const XprMatrix<E1, Rows1, Cols1>& lhs,
	  const Matrix<T2, Cols1, Cols2>& rhs) TVMET_CXX_ALWAYS_INLINE;


template<class T1, std::size_t Rows1, std::size_t Cols1,
	 class E2, std::size_t Cols2>
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


template<class T1, std::size_t Rows, std::size_t Cols, class T2>
XprVector<
  XprMVProduct<
    MatrixConstReference<T1, Rows, Cols>, Rows, Cols,
    VectorConstReference<T2, Cols>
  >,
  Rows
>
operator*(const Matrix<T1, Rows, Cols>& lhs,
	  const Vector<T2, Cols>& rhs) TVMET_CXX_ALWAYS_INLINE;


template<class T1, class E2, std::size_t Rows, std::size_t Cols>
XprVector<
  XprMVProduct<
    MatrixConstReference<T1, Rows, Cols>, Rows, Cols,
    XprVector<E2, Cols>
  >,
  Rows
>
operator*(const Matrix<T1, Rows, Cols>& lhs,
	  const XprVector<E2, Cols>& rhs) TVMET_CXX_ALWAYS_INLINE;


template<class E1, class T2, std::size_t Rows, std::size_t Cols>
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
 * Matrix integer and compare operators
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * operator(Matrix<T1, Rows, Cols>, Matrix<T2, Rows, Cols>)
 * operator(XprMatrix<E>, Matrix<T, Rows, Cols>)
 * operator(Matrix<T, Rows, Cols>, XprMatrix<E>)
 * Note: operations are per se element wise
 */
#define TVMET_DECLARE_MACRO(NAME, OP)						\
template<class T1, std::size_t Rows, std::size_t Cols,				\
	 class T2>								\
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
template<class E,								\
	 class T, std::size_t Rows, std::size_t Cols>				\
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
template<class T, std::size_t Rows, std::size_t Cols,				\
	 class E>								\
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


#if defined(TVMET_HAVE_COMPLEX)
/*
 * operator(Matrix<T, Rows, Cols>, complex<T>)
 * operator(complex<T>, Matrix<T, Rows, Cols>)
 * Note: - per se element wise
 *       - bit ops on complex<int> doesn't make sense, stay away
 * \todo type promotion
 */
#define TVMET_DECLARE_MACRO(NAME, OP)							\
template<class T, std::size_t Rows, std::size_t Cols>					\
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
template<class T, std::size_t Rows, std::size_t Cols>					\
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


/*
 * operator(Matrix<T, Rows, Cols>, POD)
 * operator(POD, Matrix<T, Rows, Cols>)
 * Note: operations are per se element wise
 */
#define TVMET_DECLARE_MACRO(NAME, OP, TP)					\
template<class T, std::size_t Rows, std::size_t Cols>				\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME<T, TP >,							\
    MatrixConstReference<T, Rows, Cols>,					\
    XprLiteral<TP >								\
  >,										\
  Rows, Cols									\
>										\
operator OP (const Matrix<T, Rows, Cols>& lhs, TP rhs) TVMET_CXX_ALWAYS_INLINE;	\
										\
template<class T, std::size_t Rows, std::size_t Cols>				\
XprMatrix<									\
  XprBinOp<									\
    Fcnl_##NAME< TP, T>,							\
    XprLiteral< TP >,								\
    MatrixConstReference<T, Rows, Cols>						\
  >,										\
  Rows, Cols									\
>										\
operator OP (TP lhs, const Matrix<T, Rows, Cols>& rhs) TVMET_CXX_ALWAYS_INLINE;

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
TVMET_DECLARE_MACRO(and, &&, float)
TVMET_DECLARE_MACRO(or, ||, float)

// necessary operators for eval functions
TVMET_DECLARE_MACRO(greater, >, double)
TVMET_DECLARE_MACRO(less, <, double)
TVMET_DECLARE_MACRO(greater_eq, >=, double)
TVMET_DECLARE_MACRO(less_eq, <=, double)
TVMET_DECLARE_MACRO(eq, ==, double)
TVMET_DECLARE_MACRO(not_eq, !=, double)
TVMET_DECLARE_MACRO(and, &&, double)
TVMET_DECLARE_MACRO(or, ||, double)

#if defined(TVMET_HAVE_LONG_DOUBLE)
// necessary operators for eval functions
TVMET_DECLARE_MACRO(greater, >, long double)
TVMET_DECLARE_MACRO(less, <, long double)
TVMET_DECLARE_MACRO(greater_eq, >=, long double)
TVMET_DECLARE_MACRO(less_eq, <=, long double)
TVMET_DECLARE_MACRO(eq, ==, long double)
TVMET_DECLARE_MACRO(not_eq, !=, long double)
TVMET_DECLARE_MACRO(and, &&, long double)
TVMET_DECLARE_MACRO(or, ||, long double)
#endif // defined(TVMET_HAVE_LONG_DOUBLE)

#undef TVMET_DECLARE_MACRO




/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * global unary operators
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * unary_operator(Matrix<T, Rows, Cols>)
 * Note: per se element wise
 */
#define TVMET_DECLARE_MACRO(NAME, OP)					\
template <class T, std::size_t Rows, std::size_t Cols>			\
XprMatrix<								\
  XprUnOp<								\
    Fcnl_##NAME<T>,							\
    MatrixConstReference<T, Rows, Cols>					\
  >,									\
  Rows, Cols								\
>									\
operator OP (const Matrix<T, Rows, Cols>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(not, !)
TVMET_DECLARE_MACRO(compl, ~)
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
template<class T, std::size_t Rows, std::size_t Cols>
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
template<class T1, class T2, std::size_t Rows, std::size_t Cols>		\
inline 										\
Matrix<T1, Rows, Cols>&								\
operator OP (Matrix<T1, Rows, Cols>& lhs, const Matrix<T2, Rows, Cols>& rhs) {	\
  return lhs.M_##NAME(rhs);							\
}										\
										\
template<class T, class E, std::size_t Rows,  std::size_t Cols>			\
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

// integer operators only, e.g used on double you wil get an error
namespace element_wise {
  TVMET_IMPLEMENT_MACRO(mod_eq, %=)
  TVMET_IMPLEMENT_MACRO(xor_eq, ^=)
  TVMET_IMPLEMENT_MACRO(and_eq, &=)
  TVMET_IMPLEMENT_MACRO(or_eq, |=)
  TVMET_IMPLEMENT_MACRO(shl_eq, <<=)
  TVMET_IMPLEMENT_MACRO(shr_eq, >>=)
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
template<class T1, class T2, std::size_t Rows, std::size_t Cols>		      \
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
template<class E, class T, std::size_t Rows, std::size_t Cols>			      \
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
template<class T, class E, std::size_t Rows, std::size_t Cols>			      \
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
template<class T, std::size_t Rows, std::size_t Cols>		\
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
template<class T, std::size_t Rows, std::size_t Cols>		\
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

#if defined(TVMET_HAVE_LONG_LONG)
TVMET_IMPLEMENT_MACRO(add, +, long long int)
TVMET_IMPLEMENT_MACRO(sub, -, long long int)
TVMET_IMPLEMENT_MACRO(mul, *, long long int)
TVMET_IMPLEMENT_MACRO(div, /, long long int)
#endif // defined(TVMET_HAVE_LONG_LONG)

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
#endif // defined(TVMET_HAVE_LONG_DOUBLE)

#undef TVMET_IMPLEMENT_MACRO


#if defined(TVMET_HAVE_COMPLEX)
/*
 * operator(Matrix<T, Rows, Cols>, complex<T>)
 * operator(complex<T>, Matrix<T, Rows, Cols>)
 * Note: operations +,-,*,/ are per se element wise
 * \todo type promotion
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)				\
template<class T, std::size_t Rows, std::size_t Cols>		\
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
template<class T, std::size_t Rows, std::size_t Cols>		\
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
template<class T1, std::size_t Rows1, std::size_t Cols1,
	 class T2, std::size_t Cols2>
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
template<class E1, std::size_t Rows1, std::size_t Cols1,
	 class T2, std::size_t Cols2>
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
template<class T1, std::size_t Rows1, std::size_t Cols1,
	 class E2, std::size_t Cols2>
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
template<class T1, std::size_t Rows, std::size_t Cols, class T2>
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
template<class T1, class E2, std::size_t Rows, std::size_t Cols>
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
template<class E1, class T2, std::size_t Rows, std::size_t Cols>
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
 * Matrix integer and compare operators
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * operator(Matrix<T1, Rows, Cols>, Matrix<T2, Rows, Cols>)
 * operator(XprMatrix<E>, Matrix<T, Rows, Cols>)
 * operator(Matrix<T, Rows, Cols>, XprMatrix<E>)
 * Note: operations are per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)							\
template<class T1, std::size_t Rows, std::size_t Cols,					\
	 class T2>									\
inline											\
XprMatrix<										\
  XprBinOp<										\
    Fcnl_##NAME<T1, T2>,								\
    MatrixConstReference<T1, Rows, Cols>,						\
    MatrixConstReference<T2, Rows, Cols>						\
  >,											\
  Rows, Cols										\
>											\
operator OP (const Matrix<T1, Rows, Cols>& lhs,						\
	     const Matrix<T2, Rows, Cols>& rhs) {					\
  typedef XprBinOp <									\
    Fcnl_##NAME<T1, T2>,								\
    MatrixConstReference<T1, Rows, Cols>,						\
    MatrixConstReference<T2, Rows, Cols>						\
  >							expr_type;			\
  return XprMatrix<expr_type, Rows, Cols>(expr_type(lhs.const_ref(), rhs.const_ref()));	\
}											\
											\
template<class E,									\
	 class T, std::size_t Rows, std::size_t Cols>					\
inline											\
XprMatrix<										\
  XprBinOp<										\
    Fcnl_##NAME<typename E::value_type, T>,						\
    XprMatrix<E, Rows, Cols>,								\
    MatrixConstReference<T, Rows, Cols>							\
  >,											\
  Rows, Cols										\
>											\
operator OP (const XprMatrix<E, Rows, Cols>& lhs, const Matrix<T, Rows, Cols>& rhs) {	\
  typedef XprBinOp<									\
    Fcnl_##NAME<typename E::value_type, T>,						\
    XprMatrix<E, Rows, Cols>,								\
    MatrixConstReference<T, Rows, Cols>							\
  > 							 expr_type;			\
  return XprMatrix<expr_type, Rows, Cols>(expr_type(lhs, rhs.const_ref()));		\
}											\
											\
template<class T, std::size_t Rows, std::size_t Cols,					\
	 class E>									\
inline											\
XprMatrix<										\
  XprBinOp<										\
    Fcnl_##NAME<typename E::value_type, T>,						\
    MatrixConstReference<T, Rows, Cols>,						\
    XprMatrix<E, Rows, Cols>								\
  >,											\
  Rows, Cols										\
>											\
operator OP (const Matrix<T, Rows, Cols>& lhs, const XprMatrix<E, Rows, Cols>& rhs) {	\
  typedef XprBinOp<									\
    Fcnl_##NAME<T, typename E::value_type>,						\
    MatrixConstReference<T, Rows, Cols>,						\
    XprMatrix<E, Rows, Cols>								\
  >	 						 expr_type;			\
  return XprMatrix<expr_type, Rows, Cols>(expr_type(lhs.const_ref(), rhs));		\
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


#if defined(TVMET_HAVE_COMPLEX)
/*
 * operator(Matrix<T, Rows, Cols>, complex<T>)
 * operator(complex<T>, Matrix<T, Rows, Cols>)
 * Note: - per se element wise
 *       - bit ops on complex<int> doesn't make sense, stay away
 * \todo type promotion
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)				     \
template<class T, std::size_t Rows, std::size_t Cols>		     \
inline								     \
XprMatrix<							     \
  XprBinOp<							     \
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,		     \
    MatrixConstReference< std::complex<T>, Rows, Cols>,		     \
    XprLiteral<std::complex<T> >				     \
  >,								     \
  Rows, Cols							     \
>								     \
operator OP (const Matrix< std::complex<T>, Rows, Cols>& lhs,	     \
	     const std::complex<T>& rhs) {			     \
  typedef XprBinOp<						     \
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,		     \
    MatrixConstReference< std::complex<T>, Rows, Cols>,		     \
    XprLiteral< std::complex<T> >				     \
  >							expr_type;   \
  return XprMatrix<expr_type, Rows, Cols>(			     \
    expr_type(lhs.const_ref(), XprLiteral< std::complex<T> >(rhs))); \
}								     \
								     \
template<class T, std::size_t Rows, std::size_t Cols>		     \
inline								     \
XprMatrix<							     \
  XprBinOp<							     \
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,		     \
    XprLiteral< std::complex<T> >,				     \
    MatrixConstReference< std::complex<T>, Rows, Cols>		     \
  >,								     \
  Rows, Cols							     \
>								     \
operator OP (const std::complex<T>& lhs,			     \
	     const Matrix< std::complex<T>, Rows, Cols>& rhs) {	     \
  typedef XprBinOp<						     \
    Fcnl_##NAME< std::complex<T>, std::complex<T> >,		     \
    XprLiteral< std::complex<T> >,				     \
    MatrixConstReference<T, Rows, Cols>				     \
  >							expr_type;   \
  return XprMatrix<expr_type, Rows, Cols>(			     \
    expr_type(XprLiteral< std::complex<T> >(lhs), rhs.const_ref())); \
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


/*
 * operator(Matrix<T, Rows, Cols>, POD)
 * operator(POD, Matrix<T, Rows, Cols>)
 * Note: operations are per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP, TP)			   \
template<class T, std::size_t Rows, std::size_t Cols>		   \
inline								   \
XprMatrix<							   \
  XprBinOp<							   \
    Fcnl_##NAME<T, TP >,					   \
    MatrixConstReference<T, Rows, Cols>,			   \
    XprLiteral<TP >						   \
  >,								   \
  Rows, Cols							   \
>								   \
operator OP (const Matrix<T, Rows, Cols>& lhs, TP rhs) {	   \
  typedef XprBinOp<						   \
    Fcnl_##NAME<T, TP >,					   \
    MatrixConstReference<T, Rows, Cols>,			   \
    XprLiteral< TP >						   \
  >							expr_type; \
  return XprMatrix<expr_type, Rows, Cols>(			   \
    expr_type(lhs.const_ref(), XprLiteral< TP >(rhs)));		   \
}								   \
								   \
template<class T, std::size_t Rows, std::size_t Cols>		   \
inline								   \
XprMatrix<							   \
  XprBinOp<							   \
    Fcnl_##NAME< TP, T>,					   \
    XprLiteral< TP >,						   \
    MatrixConstReference<T, Rows, Cols>				   \
  >,								   \
  Rows, Cols							   \
>								   \
operator OP (TP lhs, const Matrix<T, Rows, Cols>& rhs) {	   \
  typedef XprBinOp<						   \
    Fcnl_##NAME< TP, T>,					   \
    XprLiteral< TP >,						   \
    MatrixConstReference<T, Rows, Cols>				   \
  >							expr_type; \
  return XprMatrix<expr_type, Rows, Cols>(			   \
    expr_type(XprLiteral< TP >(lhs), rhs.const_ref()));		   \
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
TVMET_IMPLEMENT_MACRO(and, &&, float)
TVMET_IMPLEMENT_MACRO(or, ||, float)

// necessary operators for eval functions
TVMET_IMPLEMENT_MACRO(greater, >, double)
TVMET_IMPLEMENT_MACRO(less, <, double)
TVMET_IMPLEMENT_MACRO(greater_eq, >=, double)
TVMET_IMPLEMENT_MACRO(less_eq, <=, double)
TVMET_IMPLEMENT_MACRO(eq, ==, double)
TVMET_IMPLEMENT_MACRO(not_eq, !=, double)
TVMET_IMPLEMENT_MACRO(and, &&, double)
TVMET_IMPLEMENT_MACRO(or, ||, double)

#if defined(TVMET_HAVE_LONG_DOUBLE)
// necessary operators for eval functions
TVMET_IMPLEMENT_MACRO(greater, >, long double)
TVMET_IMPLEMENT_MACRO(less, <, long double)
TVMET_IMPLEMENT_MACRO(greater_eq, >=, long double)
TVMET_IMPLEMENT_MACRO(less_eq, <=, long double)
TVMET_IMPLEMENT_MACRO(eq, ==, long double)
TVMET_IMPLEMENT_MACRO(not_eq, !=, long double)
TVMET_IMPLEMENT_MACRO(and, &&, long double)
TVMET_IMPLEMENT_MACRO(or, ||, long double)
#endif // defined(TVMET_HAVE_LONG_DOUBLE)

#undef TVMET_IMPLEMENT_MACRO




/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * global unary operators
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*
 * unary_operator(Matrix<T, Rows, Cols>)
 * Note: per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)				       \
template <class T, std::size_t Rows, std::size_t Cols>		       \
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

TVMET_IMPLEMENT_MACRO(not, !)
TVMET_IMPLEMENT_MACRO(compl, ~)
TVMET_IMPLEMENT_MACRO(neg, -)
#undef TVMET_IMPLEMENT_MACRO


} // namespace tvmet

#endif // TVMET_MATRIX_OPERATORS_H

// Local Variables:
// mode:C++
// End:
