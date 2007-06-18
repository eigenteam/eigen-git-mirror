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
 * $Id: VectorEval.h,v 1.14 2003/11/30 08:26:25 opetzold Exp $
 */

#ifndef TVMET_VECTOR_EVAL_H
#define TVMET_VECTOR_EVAL_H

namespace tvmet {


/********************************************************************
 * functions all_elements/any_elements
 ********************************************************************/


/**
 * \fn bool all_elements(const XprVector<E, Sz>& e)
 * \brief check on statements for all elements
 * \ingroup _unary_function
 * This is for use with boolean operators like
 * \par Example:
 * \code
 * all_elements(vector > 0) {
 *     // true branch
 * } else {
 *     // false branch
 * }
 * \endcode
 * \sa \ref compare
 */
template<class E, int Sz>
inline
bool all_elements(const XprVector<E, Sz>& e) {
  return meta::Vector<Sz>::all_elements(e);
}


/**
 * \fn bool any_elements(const XprVector<E, Sz>& e)
 * \brief check on statements for any elements
 * \ingroup _unary_function
 * This is for use with boolean operators like
 * \par Example:
 * \code
 * any_elements(vector > 0) {
 *     // true branch
 * } else {
 *     // false branch
 * }
 * \endcode
 * \sa \ref compare
 */
template<class E, int Sz>
inline
bool any_elements(const XprVector<E, Sz>& e) {
  return meta::Vector<Sz>::any_elements(e);
}


/*
 * trinary evaluation functions with vectors and xpr of
 * XprVector<E1, Sz> ? Vector<T2, Sz> : Vector<T3, Sz>
 * XprVector<E1, Sz> ? Vector<T2, Sz> : XprVector<E3, Sz>
 * XprVector<E1, Sz> ? XprVector<E2, Sz> : Vector<T3, Sz>
 * XprVector<E1, Sz> ? XprVector<E2, Sz> : XprVector<E3, Sz>
 */

/**
 * eval(const XprVector<E1, Sz>& e1, const Vector<T2, Sz>& v2, const Vector<T3, Sz>& v3)
 * \brief Evals the vector expressions.
 * \ingroup _trinary_function
 * This eval is for the a?b:c syntax, since it's not allowed to overload
 * these operators.
 */
template<class E1, class T2, class T3, int Sz>
inline
XprVector<
  XprEval<
    XprVector<E1, Sz>,
    VectorConstReference<T2, Sz>,
    VectorConstReference<T3, Sz>
  >,
  Sz
>
eval(const XprVector<E1, Sz>& e1, const Vector<T2, Sz>& v2, const Vector<T3, Sz>& v3) {
  typedef XprEval<
    XprVector<E1, Sz>,
    VectorConstReference<T2, Sz>,
    VectorConstReference<T3, Sz>
  > 							expr_type;
  return XprVector<expr_type, Sz>(
    expr_type(e1, v2.const_ref(), v3.const_ref()));
}


/**
 * eval(const XprVector<E1, Sz>& e1, const Vector<T2, Sz>& v2, const XprVector<E3, Sz>& e3)
 * \brief Evals the vector expressions.
 * \ingroup _trinary_function
 * This eval is for the a?b:c syntax, since it's not allowed to overload
 * these operators.
 */
template<class E1, class T2, class E3, int Sz>
inline
XprVector<
  XprEval<
    XprVector<E1, Sz>,
    VectorConstReference<T2, Sz>,
    XprVector<E3, Sz>
  >,
  Sz
>
eval(const XprVector<E1, Sz>& e1, const Vector<T2, Sz>& v2, const XprVector<E3, Sz>& e3) {
  typedef XprEval<
    XprVector<E1, Sz>,
    VectorConstReference<T2, Sz>,
    XprVector<E3, Sz>
  > 							expr_type;
  return XprVector<expr_type, Sz>(
    expr_type(e1, v2.const_ref(), e3));
}


/**
 * eval(const XprVector<E1, Sz>& e1, const XprVector<E2, Sz>& e2, const Vector<T3, Sz>& v3)
 * \brief Evals the vector expressions.
 * \ingroup _trinary_function
 * This eval is for the a?b:c syntax, since it's not allowed to overload
 * these operators.
 */
template<class E1, class E2, class T3, int Sz>
inline
XprVector<
  XprEval<
    XprVector<E1, Sz>,
    XprVector<E2, Sz>,
    VectorConstReference<T3, Sz>
  >,
  Sz
>
eval(const XprVector<E1, Sz>& e1, const XprVector<E2, Sz>& e2, const Vector<T3, Sz>& v3) {
  typedef XprEval<
    XprVector<E1, Sz>,
    XprVector<E2, Sz>,
    VectorConstReference<T3, Sz>
  > 							expr_type;
  return XprVector<expr_type, Sz>(
    expr_type(e1, e2, v3.const_ref()));
}


/**
 * eval(const XprVector<E1, Sz>& e1, const XprVector<E2, Sz>& e2, const XprVector<E3, Sz>& e3)
 * \brief Evals the vector expressions.
 * \ingroup _trinary_function
 * This eval is for the a?b:c syntax, since it's not allowed to overload
 * these operators.
 */
template<class E1, class E2, class E3, int Sz>
inline
XprVector<
  XprEval<
    XprVector<E1, Sz>,
    XprVector<E2, Sz>,
    XprVector<E3, Sz>
  >,
  Sz
>
eval(const XprVector<E1, Sz>& e1, const XprVector<E2, Sz>& e2, const XprVector<E3, Sz>& e3) {
  typedef XprEval<
    XprVector<E1, Sz>,
    XprVector<E2, Sz>,
    XprVector<E3, Sz>
  > 							expr_type;
  return XprVector<expr_type, Sz>(expr_type(e1, e2, e3));
}


/*
 * trinary evaluation functions with vectors, xpr of and POD
 *
 * XprVector<E, Sz> ? POD1 : POD2
 * XprVector<E1, Sz> ? POD : XprVector<E3, Sz>
 * XprVector<E1, Sz> ? XprVector<E2, Sz> : POD
 */
#define TVMET_IMPLEMENT_MACRO(POD)         				\
template<class E, int Sz>      					\
inline               							\
XprVector<               						\
  XprEval<               						\
    XprVector<E, Sz>,               					\
    XprLiteral< POD >,               					\
    XprLiteral< POD >               					\
  >,                							\
  Sz									\
>               							\
eval(const XprVector<E, Sz>& e, POD x2, POD x3) {      			\
  typedef XprEval<               					\
    XprVector<E, Sz>,               					\
    XprLiteral< POD >,                					\
    XprLiteral< POD >                					\
  > 							expr_type; 	\
  return XprVector<expr_type, Sz>(					\
    expr_type(e, XprLiteral< POD >(x2), XprLiteral< POD >(x3))); 	\
}               							\
               								\
template<class E1, class E3, int Sz>   				\
inline               							\
XprVector<               						\
  XprEval<               						\
    XprVector<E1, Sz>,               					\
    XprLiteral< POD >,               					\
    XprVector<E3, Sz>               					\
  >,                							\
  Sz									\
>               							\
eval(const XprVector<E1, Sz>& e1, POD x2, const XprVector<E3, Sz>& e3) { \
  typedef XprEval<               					\
    XprVector<E1, Sz>,               					\
    XprLiteral< POD >,                					\
    XprVector<E3, Sz>               					\
  > 							expr_type; 	\
  return XprVector<expr_type, Sz>(					\
    expr_type(e1, XprLiteral< POD >(x2), e3)); 				\
}               							\
               								\
template<class E1, class E2, int Sz>   				\
inline               							\
XprVector<               						\
  XprEval<               						\
    XprVector<E1, Sz>,               					\
    XprVector<E2, Sz>,               					\
    XprLiteral< POD >               					\
  >,                							\
  Sz									\
>               							\
eval(const XprVector<E1, Sz>& e1, const XprVector<E2, Sz>& e2, POD x3) { \
  typedef XprEval<               					\
    XprVector<E1, Sz>,               					\
    XprVector<E2, Sz>,               					\
    XprLiteral< POD >                					\
  > 							expr_type; 	\
  return XprVector<expr_type, Sz>(					\
    expr_type(e1, e2, XprLiteral< POD >(x3))); 				\
}

TVMET_IMPLEMENT_MACRO(int)

TVMET_IMPLEMENT_MACRO(float)
TVMET_IMPLEMENT_MACRO(double)

#undef TVMET_IMPLEMENT_MACRO


/*
 * trinary evaluation functions with vectors, xpr of and complex<> types
 *
 * XprVector<E, Sz> e, std::complex<T> z2, std::complex<T> z3
 * XprVector<E1, Sz> e1, std::complex<T> z2, XprVector<E3, Sz> e3
 * XprVector<E1, Sz> e1, XprVector<E2, Sz> e2, std::complex<T> z3
 */
#if defined(TVMET_HAVE_COMPLEX)


/**
 * eval(const XprVector<E, Sz>& e, std::complex<T> z2, std::complex<T> z3)
 * \brief Evals the vector expressions.
 * \ingroup _trinary_function
 * This eval is for the a?b:c syntax, since it's not allowed to overload
 * these operators.
 */
template<class E, int Sz, class T>
inline
XprVector<
  XprEval<
    XprVector<E, Sz>,
    XprLiteral< std::complex<T> >,
    XprLiteral< std::complex<T> >
  >,
  Sz
>
eval(const XprVector<E, Sz>& e, std::complex<T> z2, std::complex<T> z3) {
  typedef XprEval<
    XprVector<E, Sz>,
    XprLiteral< std::complex<T> >,
    XprLiteral< std::complex<T> >
  > 							expr_type;
  return XprVector<expr_type, Sz>(
    expr_type(e, XprLiteral< std::complex<T> >(z2), XprLiteral< std::complex<T> >(z3)));
}

/**
 * eval(const XprVector<E1, Sz>& e1, std::complex<T> z2, const XprVector<E3, Sz>& e3)
 * \brief Evals the vector expressions.
 * \ingroup _trinary_function
 * This eval is for the a?b:c syntax, since it's not allowed to overload
 * these operators.
 */
template<class E1, class E3, int Sz, class T>
inline
XprVector<
  XprEval<
    XprVector<E1, Sz>,
    XprLiteral< std::complex<T> >,
    XprVector<E3, Sz>
  >,
  Sz
>
eval(const XprVector<E1, Sz>& e1, std::complex<T> z2, const XprVector<E3, Sz>& e3) {
  typedef XprEval<
    XprVector<E1, Sz>,
    XprLiteral< std::complex<T> >,
    XprVector<E3, Sz>
  > 							expr_type;
  return XprVector<expr_type, Sz>(
    expr_type(e1, XprLiteral< std::complex<T> >(z2), e3));
}

/**
 * eval(const XprVector<E1, Sz>& e1, const XprVector<E2, Sz>& e2, std::complex<T> z3)
 * \brief Evals the vector expressions.
 * \ingroup _trinary_function
 * This eval is for the a?b:c syntax, since it's not allowed to overload
 * these operators.
 */
template<class E1, class E2, int Sz, class T>
inline
XprVector<
  XprEval<
    XprVector<E1, Sz>,
    XprVector<E2, Sz>,
    XprLiteral< std::complex<T> >
  >,
  Sz
>
eval(const XprVector<E1, Sz>& e1, const XprVector<E2, Sz>& e2, std::complex<T> z3) {
  typedef XprEval<
    XprVector<E1, Sz>,
    XprVector<E2, Sz>,
    XprLiteral< std::complex<T> >
  > 							expr_type;
  return XprVector<expr_type, Sz>(
    expr_type(e1, e2, XprLiteral< std::complex<T> >(z3)));
}
#endif // defined(TVMET_HAVE_COMPLEX)


} // namespace tvmet

#endif // TVMET_VECTOR_EVAL_H

// Local Variables:
// mode:C++
// End:
