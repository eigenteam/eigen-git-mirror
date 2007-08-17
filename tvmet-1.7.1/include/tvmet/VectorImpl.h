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
 * $Id: VectorImpl.h,v 1.27 2004/06/10 16:36:55 opetzold Exp $
 */

#ifndef TVMET_VECTOR_IMPL_H
#define TVMET_VECTOR_IMPL_H

#include <iomanip>			// setw

#include <tvmet/Functional.h>

namespace tvmet {

/*
 * member operators for i/o
 */
template<class T, int Size>
std::ostream& Vector<T, Size>::print_xpr(std::ostream& os, int l) const
{
  os << IndentLevel(l++) << "Vector[" << ops << "]<"
     << typeid(T).name() << ", " << Size << ">,"
     << IndentLevel(--l)
     << std::endl;

  return os;
}


template<class T, int Size>
std::ostream& Vector<T, Size>::print_on(std::ostream& os) const
{
  os << "[\n  ";
  for(int i = 0; i < (Size - 1); ++i) {
    os << m_array[i] << ", ";
  }
  os << m_array[Size - 1] << "\n]";
  return os;
}


/*
 * member operators with scalars, per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)					\
template<class T, int Size>					\
inline									\
Vector<T, Size>& Vector<T, Size>::operator OP (value_type rhs) {		\
  typedef XprLiteral<value_type> 			expr_type;	\
  this->M_##NAME(XprVector<expr_type, Size>(expr_type(rhs)));		\
  return *this;								\
}

TVMET_IMPLEMENT_MACRO(add_eq, +=)
TVMET_IMPLEMENT_MACRO(sub_eq, -=)
TVMET_IMPLEMENT_MACRO(mul_eq, *=)
TVMET_IMPLEMENT_MACRO(div_eq, /=)
#undef TVMET_IMPLEMENT_MACRO


/*
 * member functions (operators) with vectors, for use with +=,-= ... <<=
 */
#define TVMET_IMPLEMENT_MACRO(NAME)									\
template<class T1, int Size>									\
template <class T2>											\
inline Vector<T1, Size>&											\
Vector<T1, Size>::M_##NAME (const Vector<T2, Size>& rhs) {						\
  this->M_##NAME( XprVector<typename Vector<T2, Size>::ConstRef, Size>(rhs.constRef()) );	\
  return *this;												\
}

TVMET_IMPLEMENT_MACRO(add_eq)
TVMET_IMPLEMENT_MACRO(sub_eq)
TVMET_IMPLEMENT_MACRO(mul_eq)
TVMET_IMPLEMENT_MACRO(div_eq)
#undef TVMET_IMPLEMENT_MACRO


/*
 * member functions (operators) with expressions, for use with +=,-= ... <<=
 */
#define TVMET_IMPLEMENT_MACRO(NAME)					   \
template<class T, int Size>					   \
template <class E>							   \
inline 									   \
Vector<T, Size>&								   \
Vector<T, Size>::M_##NAME (const XprVector<E, Size>& rhs) {		   \
  rhs.assign_to(*this, Fcnl_##NAME<value_type, typename E::value_type>()); \
  return *this;								   \
}

TVMET_IMPLEMENT_MACRO(add_eq)
TVMET_IMPLEMENT_MACRO(sub_eq)
TVMET_IMPLEMENT_MACRO(mul_eq)
TVMET_IMPLEMENT_MACRO(div_eq)
#undef TVMET_IMPLEMENT_MACRO


/*
 * aliased member functions (operators) with vectors,
 * for use with +=,-= ... <<=
 */
#define TVMET_IMPLEMENT_MACRO(NAME)								     \
template<class T1, int Size>								     \
template <class T2>										     \
inline 												     \
Vector<T1, Size>&											     \
Vector<T1, Size>::alias_##NAME (const Vector<T2, Size>& rhs) {					     \
  this->alias_##NAME( XprVector<typename Vector<T2, Size>::ConstRef, Size>(rhs.constRef()) ); \
  return *this;											     \
}

TVMET_IMPLEMENT_MACRO(assign)
TVMET_IMPLEMENT_MACRO(add_eq)
TVMET_IMPLEMENT_MACRO(sub_eq)
TVMET_IMPLEMENT_MACRO(mul_eq)
TVMET_IMPLEMENT_MACRO(div_eq)
#undef TVMET_IMPLEMENT_MACRO


/*
 * aliased member functions (operators) with expressions,
 * for use with +=,-= ... <<=
 */
#define TVMET_IMPLEMENT_MACRO(NAME)						      \
template<class T, int Size>						      \
template <class E>								      \
inline 										      \
Vector<T, Size>&									      \
Vector<T, Size>::alias_##NAME (const XprVector<E, Size>& rhs) {			      \
  typedef Vector<T, Size>					temp_type;		      \
  temp_type(rhs).assign_to(*this, Fcnl_##NAME<value_type, typename E::value_type>()); \
  return *this;									      \
}

TVMET_IMPLEMENT_MACRO(assign)
TVMET_IMPLEMENT_MACRO(add_eq)
TVMET_IMPLEMENT_MACRO(sub_eq)
TVMET_IMPLEMENT_MACRO(mul_eq)
TVMET_IMPLEMENT_MACRO(div_eq)
#undef TVMET_IMPLEMENT_MACRO


} // namespace tvmet

#endif // TVMET_VECTOR_IMPL_H

// Local Variables:
// mode:C++
// End:
