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
 * $Id: MatrixImpl.h,v 1.27 2004/06/10 16:36:55 opetzold Exp $
 */

#ifndef TVMET_MATRIX_IMPL_H
#define TVMET_MATRIX_IMPL_H

#include <iomanip>			// setw

#include <tvmet/Functional.h>

namespace tvmet {

/*
 * member operators for i/o
 */
template<class T, int Rows, int Cols>
std::ostream& Matrix<T, Rows, Cols>::print_xpr(std::ostream& os, int l) const
{
  os << IndentLevel(l++) << "Matrix[" << ops << "]<"
     << typeid(T).name() << ", " << Rows << ", " << Cols << ">,"
     << IndentLevel(--l)
     << std::endl;

  return os;
}


template<class T, int Rows, int Cols>
std::ostream& Matrix<T, Rows, Cols>::print_on(std::ostream& os) const
{
  os << "[\n";
  for(int i = 0; i < Rows; ++i) {
    os << " [";
    for(int j = 0; j < (Cols - 1); ++j) {
      os << this->operator()(i, j) << ", ";
    }
    os << this->operator()(i, Cols - 1)
       << (i != (Rows-1) ? "],\n" : "]\n");
    }
  os << "]";
  return os;
}

/*
 * member operators with scalars, per se element wise
 */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)				    \
template<class T, int Rows, int Cols>		    \
inline 								    \
Matrix<T, Rows, Cols>&					    \
Matrix<T, Rows, Cols>::operator OP (value_type rhs) {		    \
  typedef XprLiteral<value_type> 			expr_type;  \
  this->M_##NAME(XprMatrix<expr_type, Rows, Cols>(expr_type(rhs))); \
  return *this;							    \
}

TVMET_IMPLEMENT_MACRO(add_eq, +=)
TVMET_IMPLEMENT_MACRO(sub_eq, -=)
TVMET_IMPLEMENT_MACRO(mul_eq, *=)
TVMET_IMPLEMENT_MACRO(div_eq, /=)
#undef TVMET_IMPLEMENT_MACRO

/*
 *  member functions (operators) with matrizes, for use with +=,-= ...
 */
#define TVMET_IMPLEMENT_MACRO(NAME)									     \
template<class T1, int Rows, int Cols>						     \
template <class T2>											     \
inline 													     \
Matrix<T1, Rows, Cols>&										     \
Matrix<T1, Rows, Cols>::M_##NAME (const Matrix<T2, Rows, Cols>& rhs) {				     \
  this->M_##NAME( XprMatrix<typename Matrix<T2, Rows, Cols>::ConstRef, Rows, Cols>(rhs.constRef()) ); \
  return *this;												     \
}

TVMET_IMPLEMENT_MACRO(add_eq)
TVMET_IMPLEMENT_MACRO(sub_eq)
TVMET_IMPLEMENT_MACRO(mul_eq)
TVMET_IMPLEMENT_MACRO(div_eq)
#undef TVMET_IMPLEMENT_MACRO


/*
 * member functions (operators) with expressions, for use with +=,-= ...
 */
#define TVMET_IMPLEMENT_MACRO(NAME)					   \
template<class T, int Rows, int Cols>			   \
template<class E>							   \
inline 									   \
Matrix<T, Rows, Cols>&						   \
Matrix<T, Rows, Cols>::M_##NAME (const XprMatrix<E, Rows, Cols>& rhs) {  \
  rhs.assign_to(*this, Fcnl_##NAME<value_type, typename E::value_type>()); \
  return *this;								   \
}

TVMET_IMPLEMENT_MACRO(add_eq)
TVMET_IMPLEMENT_MACRO(sub_eq)
TVMET_IMPLEMENT_MACRO(mul_eq)
TVMET_IMPLEMENT_MACRO(div_eq)
#undef TVMET_IMPLEMENT_MACRO

/*
 * aliased member functions (operators) with matrizes,
 * for use with +=,-= ...
 */
#define TVMET_IMPLEMENT_MACRO(NAME)										 \
template<class T1, int Rows, int Cols>							 \
template <class T2>												 \
inline 														 \
Matrix<T1, Rows, Cols>&											 \
Matrix<T1, Rows, Cols>::alias_##NAME (const Matrix<T2, Rows, Cols>& rhs) {					 \
  this->alias_##NAME( XprMatrix<typename Matrix<T2, Rows, Cols>::ConstRef, Rows, Cols>(rhs.constRef()) ); \
  return *this;													 \
}

TVMET_IMPLEMENT_MACRO(assign)
TVMET_IMPLEMENT_MACRO(add_eq)
TVMET_IMPLEMENT_MACRO(sub_eq)
TVMET_IMPLEMENT_MACRO(mul_eq)
TVMET_IMPLEMENT_MACRO(div_eq)
#undef TVMET_IMPLEMENT_MACRO


/*
 * aliased member functions (operators) with expressions,
 * for use with +=,-= ... and aliased(),
 */
#define TVMET_IMPLEMENT_MACRO(NAME)						      \
template<class T, int Rows, int Cols>				      \
template<class E>								      \
inline 										      \
Matrix<T, Rows, Cols>&							      \
Matrix<T, Rows, Cols>::alias_##NAME (const XprMatrix<E, Rows, Cols>& rhs) {	      \
  typedef Matrix<T, Rows, Cols> 			temp_type;		      \
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

#endif // TVMET_MATRIX_IMPL_H

// Local Variables:
// mode:C++
// End:
