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
 * $Id: UnaryFunctionals.h,v 1.18 2004/10/04 11:44:42 opetzold Exp $
 */

#ifndef TVMET_UNARY_FUNCTIONAL_H
#define TVMET_UNARY_FUNCTIONAL_H

namespace tvmet {

/** \class Fcnl_compl	UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_neg		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
#define TVMET_IMPLEMENT_MACRO(NAME, OP)					\
template <class T>							\
struct Fcnl_##NAME : public UnaryFunctional {				\
  typedef T						value_type;	\
									\
  static inline 							\
  value_type apply_on(value_type rhs) {					\
    return OP rhs;							\
  }									\
  									\
  static 								\
  void print_xpr(std::ostream& os, int l=0) {			\
    os << IndentLevel(l) << "Fcnl_" << #NAME << "<T="			\
       << typeid(T).name() << ">,"					\
       << std::endl;							\
  }									\
};

TVMET_IMPLEMENT_MACRO(neg, -)
#undef TVMET_IMPLEMENT_MACRO

/*
 * complex support
 */


#if defined(EIGEN_USE_COMPLEX)

/**
 * \class Fcnl_conj< std::complex<T> > UnaryFunctionals.h "tvmet/UnaryFunctionals.h"
 * \brief %Functional for conj.
 */
template <class T> struct Fcnl_conj : public UnaryFunctional { };


/** \class Fcnl_conj< std::complex<T> > UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
template <class T>
struct Fcnl_conj< std::complex<T> > : public UnaryFunctional {
  typedef std::complex<T>                               value_type;

  static inline
  value_type apply_on(const std::complex<T>& rhs) {
    return std::conj(rhs);
  }

  static
  void print_xpr(std::ostream& os, int l=0) {
    os << IndentLevel(l) << "Fcnl_conj<T="
       << typeid(std::complex<T>).name() << ">,"
       << std::endl;
  }
};


/** \class Fcnl_real< std::complex<T> > UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_imag< std::complex<T> > UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
#define TVMET_IMPLEMENT_MACRO(NAME)					\
template <class T> struct Fcnl_##NAME;					\
template <class T>							\
struct Fcnl_##NAME< std::complex<T> > : public UnaryFunctional {	\
  typedef T						value_type;	\
									\
  static inline 							\
  value_type apply_on(const std::complex<T>& rhs) {			\
    return TVMET_STD_SCOPE(NAME)(rhs);					\
  }									\
  									\
  static 								\
  void print_xpr(std::ostream& os, int l=0) {			\
    os << IndentLevel(l) << "Fcnl_" << #NAME << "<T="			\
       << typeid(std::complex<T>).name() << ">,"			\
       << std::endl;							\
  }									\
};

TVMET_IMPLEMENT_MACRO(real)
TVMET_IMPLEMENT_MACRO(imag)

#undef TVMET_IMPLEMENT_MACRO

#endif // defined(EIGEN_USE_COMPLEX)

} // namespace tvmet

#endif // TVMET_UNARY_FUNCTIONAL_H

// Local Variables:
// mode:C++
// End:
