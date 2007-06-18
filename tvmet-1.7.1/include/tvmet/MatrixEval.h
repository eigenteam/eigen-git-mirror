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
 * $Id: MatrixEval.h,v 1.14 2004/06/10 16:36:55 opetzold Exp $
 */

#ifndef TVMET_MATRIX_EVAL_H
#define TVMET_MATRIX_EVAL_H

namespace tvmet {


/**
 * \fn bool all_elements(const XprMatrix<E, Rows, Cols>& e)
 * \brief check on statements for all elements
 * \ingroup _unary_function
 * This is for use with boolean operators like
 * \par Example:
 * \code
 * all_elements(matrix > 0) {
 *     // true branch
 * } else {
 *     // false branch
 * }
 * \endcode
 * \sa \ref compare
 */
template<class E, int Rows, int Cols>
inline
bool all_elements(const XprMatrix<E, Rows, Cols>& e) {
  return meta::Matrix<Rows, Cols, 0, 0>::all_elements(e);
}


/**
 * \fn bool any_elements(const XprMatrix<E, Rows, Cols>& e)
 * \brief check on statements for any elements
 * \ingroup _unary_function
 * This is for use with boolean operators like
 * \par Example:
 * \code
 * any_elements(matrix > 0) {
 *     // true branch
 * } else {
 *     // false branch
 * }
 * \endcode
 * \sa \ref compare
 */
template<class E, int Rows, int Cols>
inline
bool any_elements(const XprMatrix<E, Rows, Cols>& e) {
  return meta::Matrix<Rows, Cols, 0, 0>::any_elements(e);
}


/*
 * trinary evaluation functions with matrizes and xpr of
 *
 * XprMatrix<E1, Rows, Cols> ? Matrix<T2, Rows, Cols> : Matrix<T3, Rows, Cols>
 * XprMatrix<E1, Rows, Cols> ? Matrix<T2, Rows, Cols> : XprMatrix<E3, Rows, Cols>
 * XprMatrix<E1, Rows, Cols> ? XprMatrix<E2, Rows, Cols> : Matrix<T3, Rows, Cols>
 * XprMatrix<E1, Rows, Cols> ? XprMatrix<E2, Rows, Cols> : XprMatrix<E3, Rows, Cols>
 */

/**
 * \fn eval(const XprMatrix<E1, Rows, Cols>& e1, const Matrix<T2, Rows, Cols>& m2, const Matrix<T3, Rows, Cols>& m3)
 * \brief Evals the matrix expressions.
 * \ingroup _trinary_function
 * This eval is for the a?b:c syntax, since it's not allowed to overload
 * these operators.
 */
template<class E1, class T2, class T3, int Rows, int Cols>
inline
XprMatrix<
  XprEval<
    XprMatrix<E1, Rows, Cols>,
    MatrixConstReference<T2, Rows, Cols>,
    MatrixConstReference<T3, Rows, Cols>
  >,
  Rows, Cols
>
eval(const XprMatrix<E1, Rows, Cols>& e1,
     const Matrix<T2, Rows, Cols>& m2,
     const Matrix<T3, Rows, Cols>& m3) {
  typedef XprEval<
    XprMatrix<E1, Rows, Cols>,
    MatrixConstReference<T2, Rows, Cols>,
    MatrixConstReference<T3, Rows, Cols>
  > 							expr_type;
  return XprMatrix<expr_type, Rows, Cols>(
    expr_type(e1, m2.const_ref(), m3.const_ref()));
}


/**
 * \fn eval(const XprMatrix<E1, Rows, Cols>& e1, const Matrix<T2, Rows, Cols>& m2, const XprMatrix<E3, Rows, Cols>& e3)
 * \brief Evals the matrix expressions.
 * \ingroup _trinary_function
 * This eval is for the a?b:c syntax, since it's not allowed to overload
 * these operators.
 */
template<class E1, class T2, class E3, int Rows, int Cols>
inline
XprMatrix<
  XprEval<
    XprMatrix<E1, Rows, Cols>,
    MatrixConstReference<T2, Rows, Cols>,
    XprMatrix<E3, Rows, Cols>
  >,
  Rows, Cols
>
eval(const XprMatrix<E1, Rows, Cols>& e1,
     const Matrix<T2, Rows, Cols>& m2,
     const XprMatrix<E3, Rows, Cols>& e3) {
  typedef XprEval<
    XprMatrix<E1, Rows, Cols>,
    MatrixConstReference<T2, Rows, Cols>,
    XprMatrix<E3, Rows, Cols>
  > 							expr_type;
  return XprMatrix<expr_type, Rows, Cols>(
    expr_type(e1, m2.const_ref(), e3));
}


/**
 * \fn eval(const XprMatrix<E1, Rows, Cols>& e1, const XprMatrix<E2, Rows, Cols>& e2, const Matrix<T3, Rows, Cols>& m3)
 * \brief Evals the matrix expressions.
 * \ingroup _trinary_function
 * This eval is for the a?b:c syntax, since it's not allowed to overload
 * these operators.
 */
template<class E1, class E2, class T3, int Rows, int Cols>
inline
XprMatrix<
  XprEval<
    XprMatrix<E1, Rows, Cols>,
    XprMatrix<E2, Rows, Cols>,
    MatrixConstReference<T3, Rows, Cols>
  >,
  Rows, Cols
>
eval(const XprMatrix<E1, Rows, Cols>& e1,
    const  XprMatrix<E2, Rows, Cols>& e2,
     const Matrix<T3, Rows, Cols>& m3) {
  typedef XprEval<
    XprMatrix<E1, Rows, Cols>,
    XprMatrix<E2, Rows, Cols>,
    MatrixConstReference<T3, Rows, Cols>
  > 							expr_type;
  return XprMatrix<expr_type, Rows, Cols>(
    expr_type(e1, e2, m3.const_ref()));
}


/**
 * \fn eval(const XprMatrix<E1, Rows, Cols>& e1, const XprMatrix<E2, Rows, Cols>& e2, const XprMatrix<E3, Rows, Cols>& e3)
 * \brief Evals the matrix expressions.
 * \ingroup _trinary_function
 * This eval is for the a?b:c syntax, since it's not allowed to overload
 * these operators.
 */
template<class E1, class E2, class E3, int Rows, int Cols>
inline
XprMatrix<
  XprEval<
    XprMatrix<E1, Rows, Cols>,
    XprMatrix<E2, Rows, Cols>,
    XprMatrix<E3, Rows, Cols>
  >,
  Rows, Cols
>
eval(const XprMatrix<E1, Rows, Cols>& e1,
     const XprMatrix<E2, Rows, Cols>& e2,
     const XprMatrix<E3, Rows, Cols>& e3) {
  typedef XprEval<
    XprMatrix<E1, Rows, Cols>,
    XprMatrix<E2, Rows, Cols>,
    XprMatrix<E3, Rows, Cols>
  > 							expr_type;
  return XprMatrix<expr_type, Rows, Cols>(expr_type(e1, e2, e3));
}


/*
 * trinary evaluation functions with matrizes, xpr of and POD
 *
 * XprMatrix<E, Rows, Cols> ? POD1 : POD2
 * XprMatrix<E1, Rows, Cols> ? POD : XprMatrix<E3, Rows, Cols>
 * XprMatrix<E1, Rows, Cols> ? XprMatrix<E2, Rows, Cols> : POD
 */
#define TVMET_IMPLEMENT_MACRO(POD)               			\
template<class E, int Rows, int Cols>			\
inline               							\
XprMatrix<               						\
  XprEval<               						\
    XprMatrix<E, Rows, Cols>,               			      	\
    XprLiteral< POD >,               					\
    XprLiteral< POD >               					\
  >,                							\
  Rows, Cols								\
>               							\
eval(const XprMatrix<E, Rows, Cols>& e, POD x2, POD x3) {      		\
  typedef XprEval<               					\
    XprMatrix<E, Rows, Cols>,               				\
    XprLiteral< POD >,                					\
    XprLiteral< POD >                					\
  > 							expr_type; 	\
  return XprMatrix<expr_type, Rows, Cols>(				\
    expr_type(e, XprLiteral< POD >(x2), XprLiteral< POD >(x3))); 	\
}               							\
               								\
template<class E1, class E3, int Rows, int Cols> 	\
inline               							\
XprMatrix<               						\
  XprEval<               						\
    XprMatrix<E1, Rows, Cols>,               				\
    XprLiteral< POD >,               					\
    XprMatrix<E3, Rows, Cols>               				\
  >,                							\
  Rows, Cols								\
>               							\
eval(const XprMatrix<E1, Rows, Cols>& e1, POD x2, const XprMatrix<E3, Rows, Cols>& e3) { \
  typedef XprEval<               					\
    XprMatrix<E1, Rows, Cols>,               				\
    XprLiteral< POD >,                					\
    XprMatrix<E3, Rows, Cols>               				\
  > 							expr_type; 	\
  return XprMatrix<expr_type, Rows, Cols>(				\
    expr_type(e1, XprLiteral< POD >(x2), e3)); 				\
}               							\
               								\
template<class E1, class E2, int Rows, int Cols>	\
inline               							\
XprMatrix<               						\
  XprEval<               						\
    XprMatrix<E1, Rows, Cols>,               				\
    XprMatrix<E2, Rows, Cols>,               				\
    XprLiteral< POD >               					\
  >,                							\
  Rows, Cols								\
>               							\
eval(const XprMatrix<E1, Rows, Cols>& e1, const XprMatrix<E2, Rows, Cols>& e2, POD x3) { \
  typedef XprEval<               					\
    XprMatrix<E1, Rows, Cols>,               				\
    XprMatrix<E2, Rows, Cols>,               				\
    XprLiteral< POD >                					\
  > 							expr_type; 	\
  return XprMatrix<expr_type, Rows, Cols>(				\
    expr_type(e1, e2, XprLiteral< POD >(x3))); 				\
}

TVMET_IMPLEMENT_MACRO(int)

TVMET_IMPLEMENT_MACRO(float)
TVMET_IMPLEMENT_MACRO(double)

#undef TVMET_IMPLEMENT_MACRO


/*
 * trinary evaluation functions with matrizes, xpr of and complex<> types
 *
 * XprMatrix<E, Rows, Cols> e, std::complex<T> z2, std::complex<T> z3
 * XprMatrix<E1, Rows, Cols> e1, std::complex<T> z2, XprMatrix<E3, Rows, Cols> e3
 * XprMatrix<E1, Rows, Cols> e1, XprMatrix<E2, Rows, Cols> e2, std::complex<T> z3
 */
#if defined(TVMET_HAVE_COMPLEX)

/**
 * \fn eval(const XprMatrix<E, Rows, Cols>& e, const std::complex<T>& x2, const std::complex<T>& x3)
 * \brief Evals the matrix expressions.
 * \ingroup _trinary_function
 * This eval is for the a?b:c syntax, since it's not allowed to overload
 * these operators.
 */
template<class E, int Rows, int Cols, class T>
inline
XprMatrix<
  XprEval<
    XprMatrix<E, Rows, Cols>,
    XprLiteral< std::complex<T> >,
    XprLiteral< std::complex<T> >
  >,
  Rows, Cols
>
eval(const XprMatrix<E, Rows, Cols>& e, const std::complex<T>& x2, const std::complex<T>& x3) {
  typedef XprEval<
    XprMatrix<E, Rows, Cols>,
    XprLiteral< std::complex<T> >,
    XprLiteral< std::complex<T> >
  > 							expr_type;
  return XprMatrix<expr_type, Rows, Cols>(
    expr_type(e, XprLiteral< std::complex<T> >(x2), XprLiteral< std::complex<T> >(x3)));
}


/**
 * \fn eval(const XprMatrix<E1, Rows, Cols>& e1, const std::complex<T>& x2, const XprMatrix<E3, Rows, Cols>& e3)
 * \brief Evals the matrix expressions.
 * \ingroup _trinary_function
 * This eval is for the a?b:c syntax, since it's not allowed to overload
 * these operators.
 */
template<class E1, class E3, int Rows, int Cols, class T>
inline
XprMatrix<
  XprEval<
    XprMatrix<E1, Rows, Cols>,
    XprLiteral< std::complex<T> >,
    XprMatrix<E3, Rows, Cols>
  >,
  Rows, Cols
>
eval(const XprMatrix<E1, Rows, Cols>& e1, const std::complex<T>& x2, const XprMatrix<E3, Rows, Cols>& e3) {
  typedef XprEval<
    XprMatrix<E1, Rows, Cols>,
    XprLiteral< std::complex<T> >,
    XprMatrix<E3, Rows, Cols>
  > 							expr_type;
  return XprMatrix<expr_type, Rows, Cols>(
    expr_type(e1, XprLiteral< std::complex<T> >(x2), e3));
}


/**
 * \fn eval(const XprMatrix<E1, Rows, Cols>& e1, const XprMatrix<E2, Rows, Cols>& e2, const std::complex<T>& x3)
 * \brief Evals the matrix expressions.
 * \ingroup _trinary_function
 * This eval is for the a?b:c syntax, since it's not allowed to overload
 * these operators.
 */
template<class E1, class E2, int Rows, int Cols, class T>
inline
XprMatrix<
  XprEval<
    XprMatrix<E1, Rows, Cols>,
    XprMatrix<E2, Rows, Cols>,
    XprLiteral< std::complex<T> >
  >,
  Rows, Cols
>
eval(const XprMatrix<E1, Rows, Cols>& e1, const XprMatrix<E2, Rows, Cols>& e2, const std::complex<T>& x3) {
  typedef XprEval<
    XprMatrix<E1, Rows, Cols>,
    XprMatrix<E2, Rows, Cols>,
    XprLiteral< std::complex<T> >
  > 							expr_type;
  return XprMatrix<expr_type, Rows, Cols>(
    expr_type(e1, e2, XprLiteral< std::complex<T> >(x3)));
}
#endif // defined(TVMET_HAVE_COMPLEX)


} // namespace tvmet

#endif // TVMET_MATRIX_EVAL_H

// Local Variables:
// mode:C++
// End:
