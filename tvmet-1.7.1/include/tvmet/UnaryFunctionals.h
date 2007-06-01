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


/**
 * \class Fcnl_not	UnaryFunctionals.h "tvmet/UnaryFunctionals.h"
 * \brief unary functional for logical not.
 */
template <class T>
struct Fcnl_not : public UnaryFunctional {
  static inline
  bool apply_on(T rhs) {
    return !rhs;
  }

  static
  void print_xpr(std::ostream& os, int l=0) {
    os << IndentLevel(l) << "Fcnl_not<T="
       << typeid(T).name() << ">,"
       << std::endl;
  }
};


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

TVMET_IMPLEMENT_MACRO(compl, ~)
TVMET_IMPLEMENT_MACRO(neg, -)
#undef TVMET_IMPLEMENT_MACRO


/** \class Fcnl_abs		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_ceil		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_floor		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_sin		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_cos		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_tan		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_sinh		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_cosh		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_tanh		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_asin		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_acos		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_atan		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_exp		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_log		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_log10		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_sqrt		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
#define TVMET_IMPLEMENT_MACRO(NAME)					\
template <class T>							\
struct Fcnl_##NAME : public UnaryFunctional {				\
  typedef T						value_type;	\
  									\
  static inline 							\
  value_type apply_on(value_type rhs) {					\
    return TVMET_STD_SCOPE(NAME)(rhs);					\
  }									\
  									\
 static 								\
 void print_xpr(std::ostream& os, int l=0) {			\
    os << IndentLevel(l) << "Fcnl_" << #NAME << "<T="			\
       << typeid(value_type).name() << ">,"				\
       << std::endl;							\
  }									\
};

TVMET_IMPLEMENT_MACRO(abs)	// specialized later, see below
TVMET_IMPLEMENT_MACRO(ceil)
TVMET_IMPLEMENT_MACRO(floor)
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

#undef TVMET_IMPLEMENT_MACRO


/** \class Fcnl_cbrt		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_rint		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
#define TVMET_IMPLEMENT_MACRO(NAME)					\
template <class T>							\
struct Fcnl_##NAME : public UnaryFunctional {				\
  typedef T						value_type;	\
									\
  static inline								\
  value_type apply_on(value_type rhs) {					\
    return TVMET_GLOBAL_SCOPE(NAME)(rhs);				\
  }									\
									\
 static									\
 void print_xpr(std::ostream& os, int l=0) {			\
    os << IndentLevel(l) << "Fcnl_" << #NAME << "<T="			\
       << typeid(value_type).name() << ">,"				\
       << std::endl;							\
  }									\
};

TVMET_IMPLEMENT_MACRO(cbrt)
TVMET_IMPLEMENT_MACRO(rint)

#undef TVMET_IMPLEMENT_MACRO


#if defined(TVMET_HAVE_IEEE_MATH)

/** \class Fcnl_asinh		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_acosh		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_atanh		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_expm1		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_log1p		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_erf		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_erfc		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_j0		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_j1		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_y0		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_y1		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_lgamma		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
#define TVMET_IMPLEMENT_MACRO(NAME)					\
template <class T>							\
struct Fcnl_##NAME : public UnaryFunctional {				\
  typedef T						value_type;	\
									\
  static inline 							\
  value_type apply_on(value_type rhs) {					\
    return TVMET_GLOBAL_SCOPE(NAME)(rhs);				\
  }									\
  									\
  static 								\
  void print_xpr(std::ostream& os, int l=0) {			\
    os << IndentLevel(l) << "Fcnl_" << #NAME << "<T="			\
       << typeid(value_type).name() << ">,"				\
       << std::endl;							\
  }									\
};

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

#undef TVMET_IMPLEMENT_MACRO

#endif // defined(TVMET_HAVE_IEEE_MATH)


/** \class Fcnl_abs<long int>		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_abs<long long int>	UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_abs<float>		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_abs<double>		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_abs<long double> 	UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
#define TVMET_IMPLEMENT_MACRO(NAME, POD)				\
template <class T> struct Fcnl_##NAME;					\
template <>								\
struct Fcnl_##NAME< POD > : public UnaryFunctional {			\
  typedef POD						value_type;	\
									\
  static inline 							\
  value_type apply_on(value_type rhs) {					\
    return TVMET_STD_SCOPE(NAME)(rhs);					\
  }									\
  									\
  static 								\
  void print_xpr(std::ostream& os, int l=0) {			\
    os << IndentLevel(l) << "Fcnl_" << #NAME << "<T="			\
       << typeid(value_type).name() << ">,"				\
       << std::endl;							\
  }									\
};

TVMET_IMPLEMENT_MACRO(labs, long int)

#if defined(TVMET_HAVE_LONG_LONG)
TVMET_IMPLEMENT_MACRO(labs, long long int)
#endif

TVMET_IMPLEMENT_MACRO(fabs, float)
TVMET_IMPLEMENT_MACRO(fabs, double)

#if defined(TVMET_HAVE_LONG_DOUBLE)
TVMET_IMPLEMENT_MACRO(fabs, long double)
#endif

#undef TVMET_IMPLEMENT_MACRO


/*
 * complex support
 */

#if defined(TVMET_HAVE_COMPLEX)
/**
 * \class Fcnl_abs< std::complex<T> > UnaryFunctionals.h "tvmet/UnaryFunctionals.h"
 */
template <class T>
struct Fcnl_abs< std::complex<T> > : public UnaryFunctional {
  typedef T						value_type;

  static inline
  value_type apply_on(const std::complex<T>& rhs) {
    return std::abs(rhs);
  }

  static
  void print_xpr(std::ostream& os, int l=0) {
    os << IndentLevel(l) << "Fcnl_abs<T="
       << typeid(std::complex<T>).name() << ">,"
       << std::endl;
  }
};


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
/** \class Fcnl_arg< std::complex<T> > UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_norm< std::complex<T> > UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
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
TVMET_IMPLEMENT_MACRO(arg)
TVMET_IMPLEMENT_MACRO(norm)

#undef TVMET_IMPLEMENT_MACRO

#endif // defined(TVMET_HAVE_COMPLEX)


/** \class Fcnl_isnan		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_isinf		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
/** \class Fcnl_finite		UnaryFunctionals.h "tvmet/UnaryFunctionals.h" */
#define TVMET_IMPLEMENT_MACRO(NAME, POD)				\
template <class T>							\
struct Fcnl_##NAME : public UnaryFunctional {				\
  typedef T						value_type;	\
									\
  static inline 							\
  POD apply_on(T rhs) {							\
    return TVMET_GLOBAL_SCOPE(NAME)(rhs);				\
  }									\
  									\
  static 								\
  void print_xpr(std::ostream& os, int l=0) {			\
    os << IndentLevel(l) << "Fcnl_" << #NAME << "<T="			\
       << typeid(POD).name() << ">,"					\
       << std::endl;							\
  }									\
};

TVMET_IMPLEMENT_MACRO(isnan, int)
TVMET_IMPLEMENT_MACRO(isinf, int)
TVMET_IMPLEMENT_MACRO(finite, int)

#undef TVMET_IMPLEMENT_MACRO


} // namespace tvmet

#endif // TVMET_UNARY_FUNCTIONAL_H

// Local Variables:
// mode:C++
// End:
