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
 * $Id: VectorUnaryFunctions.h,v 1.7 2004/06/10 16:36:55 opetzold Exp $
 */

#ifndef TVMET_XPR_VECTOR_UNARY_FUNCTIONS_H
#define TVMET_XPR_VECTOR_UNARY_FUNCTIONS_H

namespace tvmet {


/*********************************************************
 * PART I: DECLARATION
 *********************************************************/

/*
 * unary_function(XprVector<E, Sz>)
 */
#define TVMET_DECLARE_MACRO(NAME)				\
template<class E, int Sz>				\
inline								\
XprVector<							\
  XprUnOp<							\
    Fcnl_##NAME<typename E::value_type>,			\
    XprVector<E, Sz>						\
  >,								\
  Sz								\
>								\
NAME(const XprVector<E, Sz>& rhs) TVMET_CXX_ALWAYS_INLINE;

TVMET_DECLARE_MACRO(abs)
TVMET_DECLARE_MACRO(cbrt)
TVMET_DECLARE_MACRO(ceil)
TVMET_DECLARE_MACRO(floor)
TVMET_DECLARE_MACRO(rint)
TVMET_DECLARE_MACRO(sin)
TVMET_DECLARE_MACRO(cos)
TVMET_DECLARE_MACRO(tan)
TVMET_DECLARE_MACRO(sinh)
TVMET_DECLARE_MACRO(cosh)
TVMET_DECLARE_MACRO(tanh)
TVMET_DECLARE_MACRO(asin)
TVMET_DECLARE_MACRO(acos)
TVMET_DECLARE_MACRO(atan)
TVMET_DECLARE_MACRO(exp)
TVMET_DECLARE_MACRO(log)
TVMET_DECLARE_MACRO(log10)
TVMET_DECLARE_MACRO(sqrt)

#if defined(TVMET_HAVE_IEEE_MATH)
TVMET_DECLARE_MACRO(asinh)
TVMET_DECLARE_MACRO(acosh)
TVMET_DECLARE_MACRO(atanh)
TVMET_DECLARE_MACRO(expm1)
TVMET_DECLARE_MACRO(log1p)
TVMET_DECLARE_MACRO(erf)
TVMET_DECLARE_MACRO(erfc)
TVMET_DECLARE_MACRO(j0)
TVMET_DECLARE_MACRO(j1)
TVMET_DECLARE_MACRO(y0)
TVMET_DECLARE_MACRO(y1)
TVMET_DECLARE_MACRO(lgamma)
/** \todo isnan etc. - default return is only an int! */
#if !defined(TVMET_NO_IEEE_MATH_ISNAN)
TVMET_DECLARE_MACRO(isnan)
#endif
#if !defined(TVMET_NO_IEEE_MATH_ISINF)
TVMET_DECLARE_MACRO(isinf)
#endif
TVMET_DECLARE_MACRO(finite)
#endif // defined(TVMET_HAVE_IEEE_MATH)

#undef TVMET_DECLARE_MACRO


/*********************************************************
 * PART II: IMPLEMENTATION
 *********************************************************/


/*
 * unary_function(XprVector<E, Sz>)
 */
#define TVMET_IMPLEMENT_MACRO(NAME)					\
template<class E, int Sz>					\
inline									\
XprVector<								\
  XprUnOp<								\
    Fcnl_##NAME<typename E::value_type>,				\
    XprVector<E, Sz>							\
  >,									\
  Sz									\
>									\
NAME(const XprVector<E, Sz>& rhs) {					\
  typedef XprUnOp<							\
    Fcnl_##NAME<typename E::value_type>,				\
    XprVector<E, Sz>							\
  > 							expr_type;	\
  return XprVector<expr_type, Sz>(expr_type(rhs));			\
}

TVMET_IMPLEMENT_MACRO(abs)
TVMET_IMPLEMENT_MACRO(cbrt)
TVMET_IMPLEMENT_MACRO(ceil)
TVMET_IMPLEMENT_MACRO(floor)
TVMET_IMPLEMENT_MACRO(rint)
TVMET_IMPLEMENT_MACRO(sin)
TVMET_IMPLEMENT_MACRO(cos)
TVMET_IMPLEMENT_MACRO(tan)
TVMET_IMPLEMENT_MACRO(sinh)
TVMET_IMPLEMENT_MACRO(cosh)
TVMET_IMPLEMENT_MACRO(tanh)
TVMET_IMPLEMENT_MACRO(asin)
TVMET_IMPLEMENT_MACRO(acos)
TVMET_IMPLEMENT_MACRO(atan)
TVMET_IMPLEMENT_MACRO(exp)
TVMET_IMPLEMENT_MACRO(log)
TVMET_IMPLEMENT_MACRO(log10)
TVMET_IMPLEMENT_MACRO(sqrt)

#if defined(TVMET_HAVE_IEEE_MATH)
TVMET_IMPLEMENT_MACRO(asinh)
TVMET_IMPLEMENT_MACRO(acosh)
TVMET_IMPLEMENT_MACRO(atanh)
TVMET_IMPLEMENT_MACRO(expm1)
TVMET_IMPLEMENT_MACRO(log1p)
TVMET_IMPLEMENT_MACRO(erf)
TVMET_IMPLEMENT_MACRO(erfc)
TVMET_IMPLEMENT_MACRO(j0)
TVMET_IMPLEMENT_MACRO(j1)
TVMET_IMPLEMENT_MACRO(y0)
TVMET_IMPLEMENT_MACRO(y1)
TVMET_IMPLEMENT_MACRO(lgamma)
/** \todo isnan etc. - default return is only an int! */
#if !defined(TVMET_NO_IEEE_MATH_ISNAN)
TVMET_IMPLEMENT_MACRO(isnan)
#endif
#if !defined(TVMET_NO_IEEE_MATH_ISINF)
TVMET_IMPLEMENT_MACRO(isinf)
#endif
TVMET_IMPLEMENT_MACRO(finite)
#endif // defined(TVMET_HAVE_IEEE_MATH)

#undef TVMET_IMPLEMENT_MACRO



} // namespace tvmet

#endif // TVMET_XPR_VECTOR_FUNCTIONS_H

// Local Variables:
// mode:C++
// End:
