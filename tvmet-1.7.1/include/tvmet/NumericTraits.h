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
 * $Id: NumericTraits.h,v 1.11 2004/11/04 18:10:35 opetzold Exp $
 */

#ifndef TVMET_NUMERIC_TRAITS_H
#define TVMET_NUMERIC_TRAITS_H

#if defined(TVMET_HAVE_COMPLEX)
#  include <complex>
#endif
#include <cmath>
#include <limits>

#include <tvmet/CompileTimeError.h>


namespace tvmet {


/**
 * \class NumericTraits NumericTraits.h "tvmet/NumericTraits.h"
 * \brief Traits for integral types for operations.
 *
 * For each type we have to specialize this traits.
 *
 * \note Keep in mind that the long types long long and long double doesn't
 *       have traits. This is due to the sum_type. We can't give a guarantee
 *       that there is a type of holding the sum. Therefore using this traits
 *       is only safe if you have long long resp. long double types by
 *       working on long ints and doubles. Otherwise you will get not expected
 *       result for some circumstances. Anyway, you can use big integer/float
 *       libraries and specialize the traits by your own.
 *
 * \todo The abs function of complex<non_float_type> can have an
 *       overrun due to numeric computation. Solve it (someone
 *       using value_type=long here?)
 */
template<class T>
struct NumericTraits {
  typedef T					base_type;
  typedef T					value_type;
  typedef value_type				sum_type;
  typedef value_type				diff_type;
  typedef value_type				float_type;
  typedef value_type				signed_type;

  typedef NumericTraits<value_type>		traits_type;
  typedef const value_type&			argument_type;

  static inline
  base_type real(argument_type x);

  static inline
  base_type imag(argument_type x);

  static inline
  value_type conj(argument_type x);

  static inline
  base_type abs(argument_type x);

  static inline
  value_type sqrt(argument_type x);

  static inline
  base_type norm_1(argument_type x) {
    return NumericTraits<base_type>::abs(traits_type::real(x))
         + NumericTraits<base_type>::abs(traits_type::imag(x));
  }

  static inline
  base_type norm_2(argument_type x) { return traits_type::abs(x); }

  static inline
  base_type norm_inf(argument_type x) {
    return std::max(NumericTraits<base_type>::abs(traits_type::real(x)),
		    NumericTraits<base_type>::abs(traits_type::imag(x)));
   }

  static inline
  bool equals(argument_type lhs, argument_type rhs) {
    static base_type sqrt_epsilon(
      NumericTraits<base_type>::sqrt(
        std::numeric_limits<base_type>::epsilon()));

    return traits_type::norm_inf(lhs - rhs) < sqrt_epsilon *
      std::max(std::max(traits_type::norm_inf(lhs),
			traits_type::norm_inf(rhs)),
	       std::numeric_limits<base_type>::min());
  }
};


/*
 * numeric traits for standard types
 */


/**
 * \class NumericTraits<char> NumericTraits.h "tvmet/NumericTraits.h"
 * \brief Traits specialized for char.
 */
template<>
struct NumericTraits<char> {
  typedef char					value_type;
  typedef value_type				base_type;
  typedef long					sum_type;
  typedef int					diff_type;
  typedef float					float_type;
  typedef char					signed_type;

  typedef NumericTraits<value_type>		traits_type;
  typedef value_type				argument_type;

  static inline
  base_type real(argument_type x) { return x; }

  static inline
  base_type imag(argument_type x) { TVMET_UNUSED(x); return 0; }

  static inline
  value_type conj(argument_type x) { return x; }

  static inline
  base_type abs(argument_type x) { return std::abs(x); }

  static inline
  value_type sqrt(argument_type x) {
    return static_cast<value_type>(std::sqrt(static_cast<float_type>(x)));
  }

  static inline
  base_type norm_1(argument_type x) { return traits_type::abs(x); }

  static inline
  base_type norm_2(argument_type x) { return traits_type::abs(x); }

  static inline
  base_type norm_inf(argument_type x) { return traits_type::abs(x); }

  static inline
  bool equals(argument_type lhs, argument_type rhs) { return lhs == rhs; }

  enum { is_complex = false };

  /** Complexity on operations. */
  enum {
    ops_plus = 1,	/**< Complexity on plus/minus ops. */
    ops_muls = 1	/**< Complexity on multiplications. */
  };
};


/**
 * \class NumericTraits<unsigned char> NumericTraits.h "tvmet/NumericTraits.h"
 * \brief Traits specialized for unsigned char.
 *
 * \note Normally it doesn't make sense to call <tt>conj</tt>
 *       for an unsigned type! An unary minus operator
 *       applied to unsigned type will result unsigned. Therefore
 *       this function is missing here.
 */
template<>
struct NumericTraits<unsigned char> {
  typedef unsigned char 			value_type;
  typedef value_type				base_type;
  typedef unsigned long	 			sum_type;
  typedef int		 			diff_type;
  typedef float		 			float_type;
  typedef int					signed_type;

  typedef NumericTraits<value_type>		traits_type;
  typedef value_type				argument_type;

  static inline
  base_type real(argument_type x) { return x; }

  static inline
  base_type imag(argument_type x) { TVMET_UNUSED(x); return 0; }

  static inline
  base_type abs(argument_type x) { return std::abs(x); }

  static inline
  value_type sqrt(argument_type x) {
    return static_cast<value_type>(std::sqrt(static_cast<float_type>(x)));
  }

  static inline
  base_type norm_1(argument_type x) { return traits_type::abs(x); }

  static inline
  base_type norm_2(argument_type x) { return traits_type::abs(x); }

  static inline
  base_type norm_inf(argument_type x) { return traits_type::abs(x); }

  static inline
  bool equals(argument_type lhs, argument_type rhs) { return lhs == rhs; }

  enum { is_complex = false };

  /** Complexity on operations. */
  enum {
    ops_plus = 1,	/**< Complexity on plus/minus ops. */
    ops_muls = 1	/**< Complexity on multiplications. */
  };
};


/**
 * \class NumericTraits<short int> NumericTraits.h "tvmet/NumericTraits.h"
 * \brief Traits specialized for short int.
 */
template<>
struct NumericTraits<short int> {
  typedef short int 				value_type;
  typedef value_type				base_type;
#if defined(TVMET_HAVE_LONG_LONG)
  typedef long long		 		sum_type;
#else
  typedef long 					sum_type;
#endif
  typedef int			 		diff_type;
  typedef float 				float_type;
  typedef short int				signed_type;

  typedef NumericTraits<value_type>		traits_type;
  typedef value_type				argument_type;

  static inline
  base_type real(argument_type x) { return x; }

  static inline
  base_type imag(argument_type x) { TVMET_UNUSED(x); return 0; }

  static inline
  value_type conj(argument_type x) { return x; }

  static inline
  base_type abs(argument_type x) { return std::abs(x); }

  static inline
  value_type sqrt(argument_type x) {
    return static_cast<value_type>(std::sqrt(static_cast<float_type>(x)));
  }

  static inline
  base_type norm_1(argument_type x) { return traits_type::abs(x); }

  static inline
  base_type norm_2(argument_type x) { return traits_type::abs(x); }

  static inline
  base_type norm_inf(argument_type x) { return traits_type::abs(x); }

  static inline
  bool equals(argument_type lhs, argument_type rhs) { return lhs == rhs; }

  enum { is_complex = false };

  /** Complexity on operations. */
  enum {
    ops_plus = 1,	/**< Complexity on plus/minus ops. */
    ops_muls = 1	/**< Complexity on multiplications. */
  };
};


/**
 * \class NumericTraits<short unsigned int> NumericTraits.h "tvmet/NumericTraits.h"
 * \brief Traits specialized for short unsigned int.
 *
 * \note Normally it doesn't make sense to call <tt>conj</tt>
 *       for an unsigned type! An unary minus operator
 *       applied to unsigned type will result unsigned. Therefore
 *       this function is missing here.
 */
template<>
struct NumericTraits<short unsigned int> {
  typedef short unsigned int			value_type;
  typedef value_type				base_type;
#if defined(TVMET_HAVE_LONG_LONG)
  typedef unsigned long long			sum_type;
#else
  typedef unsigned long 			sum_type;
#endif
  typedef int 					diff_type;
  typedef float 				float_type;
  typedef int					signed_type;

  typedef NumericTraits<value_type>		traits_type;
  typedef value_type				argument_type;

  static inline
  base_type real(argument_type x) { return x; }

  static inline
  base_type imag(argument_type x) { TVMET_UNUSED(x); return 0; }

  static inline
  base_type abs(argument_type x) { return std::abs(x); }

  static inline
  value_type sqrt(argument_type x) {
    return static_cast<value_type>(std::sqrt(static_cast<float_type>(x)));
  }

  static inline
  base_type norm_1(argument_type x) { return traits_type::abs(x); }

  static inline
  base_type norm_2(argument_type x) { return traits_type::abs(x); }

  static inline
  base_type norm_inf(argument_type x) { return traits_type::abs(x); }

  static inline
  bool equals(argument_type lhs, argument_type rhs) { return lhs == rhs; }

  enum { is_complex = false };

  /** Complexity on operations. */
  enum {
    ops_plus = 1,	/**< Complexity on plus/minus ops. */
    ops_muls = 1	/**< Complexity on multiplications. */
  };
};


/**
 * \class NumericTraits<int> NumericTraits.h "tvmet/NumericTraits.h"
 * \brief Traits specialized for int.
 */
template<>
struct NumericTraits<int> {
  typedef int 					value_type;
  typedef value_type				base_type;
#if defined(TVMET_HAVE_LONG_LONG)
  typedef long long		 		sum_type;
#else
  typedef long 					sum_type;
#endif
  typedef int			 		diff_type;
  typedef double			 	float_type;
  typedef int					signed_type;

  typedef NumericTraits<value_type>		traits_type;
  typedef value_type				argument_type;

  static inline
  base_type real(argument_type x) { return x; }

  static inline
  base_type imag(argument_type x) { TVMET_UNUSED(x); return 0; }

  static inline
  value_type conj(argument_type x) { return x; }

  static inline
  base_type abs(argument_type x) { return std::abs(x); }

  static inline
  value_type sqrt(argument_type x) {
    return static_cast<value_type>(std::sqrt(static_cast<float_type>(x)));
  }

  static inline
  base_type norm_1(argument_type x) { return traits_type::abs(x); }

  static inline
  base_type norm_2(argument_type x) { return traits_type::abs(x); }

  static inline
  base_type norm_inf(argument_type x) { return traits_type::abs(x); }

  static inline
  bool equals(argument_type lhs, argument_type rhs) { return lhs == rhs; }

  enum { is_complex = false };

  /** Complexity on operations. */
  enum {
    ops_plus = 1,	/**< Complexity on plus/minus ops. */
    ops_muls = 1	/**< Complexity on multiplications. */
  };
};


/**
 * \class NumericTraits<unsigned int> NumericTraits.h "tvmet/NumericTraits.h"
 * \brief Traits specialized for unsigned int.
 *
 * \note Normally it doesn't make sense to call <tt>conj</tt>
 *       for an unsigned type! An unary minus operator
 *       applied to unsigned type will result unsigned. Therefore
 *       this function is missing here.
 */
template<>
struct NumericTraits<unsigned int> {
  typedef unsigned int 		 		value_type;
  typedef value_type				base_type;
#if defined(TVMET_HAVE_LONG_LONG)
  typedef unsigned long long			sum_type;
#else
  typedef unsigned long 			sum_type;
#endif
  typedef int 			 		diff_type;
  typedef double 				float_type;
  typedef long			 		signed_type;

  typedef NumericTraits<value_type>		traits_type;
  typedef value_type				argument_type;

  static inline
  base_type real(argument_type x) { return x; }

  static inline
  base_type imag(argument_type x) { TVMET_UNUSED(x); return 0; }

  static inline
  base_type abs(argument_type x) { return x; }

  static inline
  value_type sqrt(argument_type x) {
    return static_cast<value_type>(std::sqrt(static_cast<float_type>(x)));
  }

  static inline
  base_type norm_1(argument_type x) { return traits_type::abs(x); }

  static inline
  base_type norm_2(argument_type x) { return traits_type::abs(x); }

  static inline
  base_type norm_inf(argument_type x) { return traits_type::abs(x); }

  static inline
  bool equals(argument_type lhs, argument_type rhs) { return lhs == rhs; }

  enum { is_complex = false };

  /** Complexity on operations. */
  enum {
    ops_plus = 1,	/**< Complexity on plus/minus ops. */
    ops_muls = 1	/**< Complexity on multiplications. */
  };
};


/**
 * \class NumericTraits<long> NumericTraits.h "tvmet/NumericTraits.h"
 * \brief Traits specialized for long.
 */
template<>
struct NumericTraits<long> {
  typedef long 			 		value_type;
  typedef value_type				base_type;
#if defined(TVMET_HAVE_LONG_LONG)
  typedef long long		 		sum_type;
#else
  typedef long			  		sum_type;
#endif
  typedef long 		 			diff_type;
  typedef double 				float_type;
  typedef long		 	 		signed_type;

  typedef NumericTraits<value_type>		traits_type;
  typedef value_type				argument_type;

  static inline
  base_type real(argument_type x) { return x; }

  static inline
  base_type imag(argument_type x) { TVMET_UNUSED(x); return 0; }

  static inline
  value_type conj(argument_type x) { return x; }

  static inline
  base_type abs(argument_type x) { return std::abs(x); }

  static inline
  value_type sqrt(argument_type x) {
    return static_cast<value_type>(std::sqrt(static_cast<float_type>(x)));
  }

  static inline
  base_type norm_1(argument_type x) { return traits_type::abs(x); }

  static inline
  base_type norm_2(argument_type x) { return traits_type::abs(x); }

  static inline
  base_type norm_inf(argument_type x) { return traits_type::abs(x); }

  static inline
  bool equals(argument_type lhs, argument_type rhs) { return lhs == rhs; }

  enum { is_complex = false };

  /** Complexity on operations. */
  enum {
    ops_plus = 1,	/**< Complexity on plus/minus ops. */
    ops_muls = 1	/**< Complexity on multiplications. */
  };
};


/**
 * \class NumericTraits<unsigned long> NumericTraits.h "tvmet/NumericTraits.h"
 * \brief Traits specialized for unsigned long.
 *
 * \note Normally it doesn't make sense to call <tt>conj</tt>
 *       for an unsigned type! An unary minus operator
 *       applied to unsigned type will result unsigned. Therefore
 *       this function is missing here.
 */
template<>
struct NumericTraits<unsigned long> {
  typedef unsigned long 			value_type;
  typedef value_type				base_type;
#if defined(TVMET_HAVE_LONG_LONG)
  typedef unsigned long long 			sum_type;
#else
  typedef unsigned long 			sum_type;
#endif
  typedef unsigned long 			diff_type;
  typedef double 				float_type;
  typedef long					signed_type;

  typedef NumericTraits<value_type>		traits_type;
  typedef value_type				argument_type;

  static inline
  base_type real(argument_type x) { return x; }

  static inline
  base_type imag(argument_type x) { TVMET_UNUSED(x); return 0; }

  static inline
  base_type abs(argument_type x) { return x; }

  static inline
  value_type sqrt(argument_type x) {
    return static_cast<value_type>(std::sqrt(static_cast<float_type>(x)));
  }

  static inline
  base_type norm_1(argument_type x) { return traits_type::abs(x); }

  static inline
  base_type norm_2(argument_type x) { return traits_type::abs(x); }

  static inline
  base_type norm_inf(argument_type x) { return traits_type::abs(x); }

  static inline
  bool equals(argument_type lhs, argument_type rhs) { return lhs == rhs; }

  enum { is_complex = false };

  /** Complexity on operations. */
  enum {
    ops_plus = 1,	/**< Complexity on plus/minus ops. */
    ops_muls = 1	/**< Complexity on multiplications. */
  };
};


/**
 * \class NumericTraits<float> NumericTraits.h "tvmet/NumericTraits.h"
 * \brief Traits specialized for float.
 */
template<>
struct NumericTraits<float> {
  typedef float					value_type;
  typedef value_type				base_type;
  typedef double 				sum_type;
  typedef float 				diff_type;
  typedef float 				float_type;
  typedef float					signed_type;

  typedef NumericTraits<value_type>		traits_type;
  typedef value_type				argument_type;

  static inline
  base_type real(argument_type x) { return x; }

  static inline
  base_type imag(argument_type x) { TVMET_UNUSED(x); return 0; }

  static inline
  value_type conj(argument_type x) { return x; }

  static inline
  base_type abs(argument_type x) { return std::abs(x); }

  static inline
  value_type sqrt(argument_type x) { return std::sqrt(x); }

  static inline
  base_type norm_1(argument_type x) { return traits_type::abs(x); }

  static inline
  base_type norm_2(argument_type x) { return traits_type::abs(x); }

  static inline
  base_type norm_inf(argument_type x) { return traits_type::abs(x); }

  static inline
  bool equals(argument_type lhs, argument_type rhs) {
    static base_type sqrt_epsilon(
      NumericTraits<base_type>::sqrt(
        std::numeric_limits<base_type>::epsilon()));

    return traits_type::norm_inf(lhs - rhs) < sqrt_epsilon *
      std::max(std::max(traits_type::norm_inf(lhs),
			traits_type::norm_inf(rhs)),
	       std::numeric_limits<base_type>::min());
  }

  enum { is_complex = false };

  /** Complexity on operations. */
  enum {
    ops_plus = 1,	/**< Complexity on plus/minus ops. */
    ops_muls = 1	/**< Complexity on multiplications. */
  };
};


/**
 * \class NumericTraits<double> NumericTraits.h "tvmet/NumericTraits.h"
 * \brief Traits specialized for double.
 */
template<>
struct NumericTraits<double> {
  typedef double 				value_type;
  typedef value_type				base_type;
#if defined(TVMET_HAVE_LONG_DOUBLE)
  typedef long double		 		sum_type;
#else
  typedef double 				sum_type;
#endif
  typedef double				diff_type;
  typedef double 				float_type;
  typedef double				signed_type;

  typedef NumericTraits<value_type>		traits_type;
  typedef value_type				argument_type;

  static inline
  base_type real(argument_type x) { return x; }

  static inline
  base_type imag(argument_type x) { TVMET_UNUSED(x); return 0; }

  static inline
  value_type conj(argument_type x) { return x; }

  static inline
  base_type abs(argument_type x) { return std::abs(x); }

  static inline
  value_type sqrt(argument_type x) { return std::sqrt(x); }

  static inline
  base_type norm_1(argument_type x) { return traits_type::abs(x); }

  static inline
  base_type norm_2(argument_type x) { return traits_type::abs(x); }

  static inline
  base_type norm_inf(argument_type x) { return traits_type::abs(x); }

  static inline
  bool equals(argument_type lhs, argument_type rhs) {
    static base_type sqrt_epsilon(
      NumericTraits<base_type>::sqrt(
        std::numeric_limits<base_type>::epsilon()));

    return traits_type::norm_inf(lhs - rhs) < sqrt_epsilon *
      std::max(std::max(traits_type::norm_inf(lhs),
			traits_type::norm_inf(rhs)),
	       std::numeric_limits<base_type>::min());
  }

  enum { is_complex = false };

  /** Complexity on operations. */
  enum {
    ops_plus = 1,	/**< Complexity on plus/minus ops. */
    ops_muls = 1	/**< Complexity on multiplications. */
  };
};


#if defined(TVMET_HAVE_LONG_DOUBLE)
/**
 * \class NumericTraits<long double> NumericTraits.h "tvmet/NumericTraits.h"
 * \brief Traits specialized for long double.
 */
template<>
struct NumericTraits<long double> {
  typedef long double 				value_type;
  typedef value_type				base_type;
  typedef long double		 		sum_type;
  typedef long double				diff_type;
  typedef long double 				float_type;
  typedef long double				signed_type;

  typedef NumericTraits<value_type>		traits_type;
  typedef value_type				argument_type;

  static inline
  base_type real(argument_type x) { return x; }

  static inline
  base_type imag(argument_type x) { TVMET_UNUSED(x); return 0; }

  static inline
  value_type conj(argument_type x) { return x; }

  static inline
  base_type abs(argument_type x) { return std::abs(x); }

  static inline
  value_type sqrt(argument_type x) { return std::sqrt(x); }

  static inline
  base_type norm_1(argument_type x) { return traits_type::abs(x); }

  static inline
  base_type norm_2(argument_type x) { return traits_type::abs(x); }

  static inline
  base_type norm_inf(argument_type x) { return traits_type::abs(x); }

  static inline
  bool equals(argument_type lhs, argument_type rhs) {
    static base_type sqrt_epsilon(
      NumericTraits<base_type>::sqrt(
        std::numeric_limits<base_type>::epsilon()));

    return traits_type::norm_inf(lhs - rhs) < sqrt_epsilon *
      std::max(std::max(traits_type::norm_inf(lhs),
			traits_type::norm_inf(rhs)),
	       std::numeric_limits<base_type>::min());
  }

  enum { is_complex = false };

  /** Complexity on operations. */
  enum {
    ops_plus = 1,	/**< Complexity on plus/minus ops. */
    ops_muls = 1	/**< Complexity on multiplications. */
  };
};
#endif // TVMET_HAVE_LONG_DOUBLE


/*
 * numeric traits for complex types
 */
#if defined(TVMET_HAVE_COMPLEX)

/**
 * \class NumericTraits< std::complex<int> > NumericTraits.h "tvmet/NumericTraits.h"
 * \brief Traits specialized for std::complex<int>.
 */
template<>
struct NumericTraits< std::complex<int> > {
  typedef int					base_type;
  typedef std::complex<int>			value_type;
  typedef std::complex<long> 			sum_type;
  typedef std::complex<int>			diff_type;
  typedef std::complex<float>			float_type;
  typedef std::complex<int>			signed_type;

  typedef NumericTraits<value_type>		traits_type;
  typedef const value_type&			argument_type;

  static inline
  base_type real(argument_type z) { return std::real(z); }

  static inline
  base_type imag(argument_type z) { return std::imag(z); }

  static inline
  value_type conj(argument_type z) { return std::conj(z); }

  static inline
  base_type abs(argument_type z) {
    base_type x = z.real();
    base_type y = z.imag();

    // XXX probably case of overrun; header complex uses scaling
    return static_cast<base_type>(NumericTraits<base_type>::sqrt(x * x + y * y));
  }

  static /* inline */
  value_type sqrt(argument_type z) {
    // borrowed and adapted from header complex
    base_type x = z.real();
    base_type y = z.imag();

    if(x == base_type()) {
	base_type t = NumericTraits<base_type>::sqrt(
                        NumericTraits<base_type>::abs(y) / 2);
	return value_type(t, y < base_type() ? -t : t);
    }
    else {
      base_type t = NumericTraits<base_type>::sqrt(
		      2 * (traits_type::abs(z)
		            + NumericTraits<base_type>::abs(x)));
      base_type u = t / 2;
      return x > base_type()
	? value_type(u, y / t)
	: value_type(NumericTraits<base_type>::abs(y) / t, y < base_type() ? -u : u);
    }
  }

  static inline
  base_type norm_1(argument_type z) {
    return NumericTraits<base_type>::abs((traits_type::real(z)))
         + NumericTraits<base_type>::abs((traits_type::imag(z)));
  }

  static inline
  base_type norm_2(argument_type z) { return traits_type::abs(z); }

  static inline
  base_type norm_inf(argument_type z) {
    return std::max(NumericTraits<base_type>::abs(traits_type::real(z)),
		    NumericTraits<base_type>::abs(traits_type::imag(z)));
  }

  static inline
  bool equals(argument_type lhs, argument_type rhs) {
    return (traits_type::real(lhs) == traits_type::real(rhs))
        && (traits_type::imag(lhs) == traits_type::imag(rhs));
  }

  enum { is_complex = true };

  /** Complexity on operations. */
  enum {
    ops_plus = 2,	/**< Complexity on plus/minus ops. */
    ops_muls = 6	/**< Complexity on multiplications. */
  };
};


/**
 * \class NumericTraits< std::complex<unsigned int> > NumericTraits.h "tvmet/NumericTraits.h"
 * \brief Traits specialized for std::complex<unsigned int>.
 *
 * \note Normally it doesn't make sense to call <tt>conj</tt>
 *       for an unsigned type! An unary minus operator
 *       applied to unsigned type will result unsigned. Therefore
 *       this function is missing here.
 */
template<>
struct NumericTraits< std::complex<unsigned int> > {
  typedef unsigned int				base_type;
  typedef std::complex<unsigned int> 		value_type;
  typedef std::complex<unsigned long> 		sum_type;
  typedef std::complex<int>			diff_type;
  typedef std::complex<float>			float_type;
  typedef std::complex<int>			signed_type;

  typedef NumericTraits<value_type>		traits_type;
  typedef const value_type&			argument_type;

  static inline
  base_type real(argument_type z) { return std::real(z); }

  static inline
  base_type imag(argument_type z) { return std::imag(z); }

  static inline
  base_type abs(argument_type z) {
    base_type x = z.real();
    base_type y = z.imag();

    // XXX probably case of overrun; header complex uses scaling
    return static_cast<base_type>(NumericTraits<base_type>::sqrt(x * x + y * y));
  }

  static /* inline */
  value_type sqrt(argument_type z) {
    // borrowed and adapted from header complex
    base_type x = z.real();
    base_type y = z.imag();

    if(x == base_type()) {
	base_type t = NumericTraits<base_type>::sqrt(
                        NumericTraits<base_type>::abs(y) / 2);
	return value_type(t, t);
    }
    else {
      base_type t = NumericTraits<base_type>::sqrt(
		      2 * (traits_type::abs(z)
		            + NumericTraits<base_type>::abs(x)));
      return value_type(t / 2, y / t);
    }
  }

  static inline
  base_type norm_1(argument_type z) {
    return NumericTraits<base_type>::abs((traits_type::real(z)))
         + NumericTraits<base_type>::abs((traits_type::imag(z)));
  }

  static inline
  base_type norm_2(argument_type z) { return traits_type::abs(z); }

  static inline
  base_type norm_inf(argument_type z) {
    return std::max(NumericTraits<base_type>::abs(traits_type::real(z)),
		    NumericTraits<base_type>::abs(traits_type::imag(z)));
  }

  static inline
  bool equals(argument_type lhs, argument_type rhs) {
    return (traits_type::real(lhs) == traits_type::real(rhs))
        && (traits_type::imag(lhs) == traits_type::imag(rhs));
  }

  enum { is_complex = true };

  /** Complexity on operations. */
  enum {
    ops_plus = 2,	/**< Complexity on plus/minus ops. */
    ops_muls = 6	/**< Complexity on multiplications. */
  };
};


/**
 * \class NumericTraits< std::complex<long> > NumericTraits.h "tvmet/NumericTraits.h"
 * \brief Traits specialized for std::complex<long>.
 */
template<>
struct NumericTraits< std::complex<long> > {
  typedef long					base_type;
  typedef std::complex<long>			value_type;
#if defined(TVMET_HAVE_LONG_LONG)
  typedef std::complex<long long>		sum_type;
#else
  typedef std::complex<long>			sum_type;
#endif
  typedef std::complex<int>			diff_type;
  typedef std::complex<float>			float_type;
  typedef std::complex<int>			signed_type;

  typedef NumericTraits<value_type>		traits_type;
  typedef const value_type&			argument_type;

  static inline
  base_type real(argument_type z) { return std::real(z); }

  static inline
  base_type imag(argument_type z) { return std::imag(z); }

  static inline
  value_type conj(argument_type z) { return std::conj(z); }

  static inline
  base_type abs(argument_type z) {
    base_type x = z.real();
    base_type y = z.imag();

    // XXX probably case of overrun; header complex uses scaling
    return static_cast<base_type>(NumericTraits<base_type>::sqrt(x * x + y * y));
  }

  static /* inline */
  value_type sqrt(argument_type z) {
    // borrowed and adapted from header complex
    base_type x = z.real();
    base_type y = z.imag();

    if(x == base_type()) {
	base_type t = NumericTraits<base_type>::sqrt(
                        NumericTraits<base_type>::abs(y) / 2);
	return value_type(t, y < base_type() ? -t : t);
    }
    else {
      base_type t = NumericTraits<base_type>::sqrt(
		      2 * (traits_type::abs(z)
		            + NumericTraits<base_type>::abs(x)));
      base_type u = t / 2;
      return x > base_type()
	? value_type(u, y / t)
	: value_type(NumericTraits<base_type>::abs(y) / t, y < base_type() ? -u : u);
    }
  }

  static inline
  base_type norm_1(argument_type z) {
    return NumericTraits<base_type>::abs((traits_type::real(z)))
         + NumericTraits<base_type>::abs((traits_type::imag(z)));
  }

  static inline
  base_type norm_2(argument_type z) { return traits_type::abs(z); }

  static inline
  base_type norm_inf(argument_type z) {
    return std::max(NumericTraits<base_type>::abs(traits_type::real(z)),
		    NumericTraits<base_type>::abs(traits_type::imag(z)));
  }

  static inline
  bool equals(argument_type lhs, argument_type rhs) {
    return (traits_type::real(lhs) == traits_type::real(rhs))
        && (traits_type::imag(lhs) == traits_type::imag(rhs));
  }

  enum { is_complex = true };

  /** Complexity on operations. */
  enum {
    ops_plus = 2,	/**< Complexity on plus/minus ops. */
    ops_muls = 6	/**< Complexity on multiplications. */
  };
};


/**
 * \class NumericTraits< std::complex<unsigned long> > NumericTraits.h "tvmet/NumericTraits.h"
 * \brief Traits specialized for std::complex<unsigned long>.
 *
 * \note Normally it doesn't make sense to call <tt>conj</tt>
 *       for an unsigned type! An unary minus operator
 *       applied to unsigned type will result unsigned. Therefore
 *       this function is missing here.
 */
template<>
struct NumericTraits< std::complex<unsigned long> > {
  typedef unsigned long				base_type;
  typedef std::complex<unsigned long>		value_type;
#if defined(TVMET_HAVE_LONG_LONG)
  typedef std::complex<unsigned long long>	sum_type;
#else
  typedef std::complex<unsigned long>		sum_type;
#endif
  typedef std::complex<long>			diff_type;
  typedef std::complex<float>			float_type;
  typedef std::complex<long>			signed_type;

  typedef NumericTraits<value_type>		traits_type;
  typedef const value_type&			argument_type;

  static inline
  base_type real(argument_type z) { return std::real(z); }

  static inline
  base_type imag(argument_type z) { return std::imag(z); }

  static inline
  base_type abs(argument_type z) {
    base_type x = z.real();
    base_type y = z.imag();

    // XXX probably case of overrun; header complex uses scaling
    return static_cast<base_type>(NumericTraits<base_type>::sqrt(x * x + y * y));
  }

  static /* inline */
  value_type sqrt(argument_type z) {
    // borrowed and adapted from header complex
    base_type x = z.real();
    base_type y = z.imag();

    if(x == base_type()) {
	base_type t = NumericTraits<base_type>::sqrt(
                        NumericTraits<base_type>::abs(y) / 2);
	return value_type(t, t);
    }
    else {
      base_type t = NumericTraits<base_type>::sqrt(
		      2 * (traits_type::abs(z)
		            + NumericTraits<base_type>::abs(x)));
      return value_type(t / 2, y / t);
    }
  }

  static inline
  base_type norm_1(argument_type z) {
    return NumericTraits<base_type>::abs((traits_type::real(z)))
         + NumericTraits<base_type>::abs((traits_type::imag(z)));
  }

  static inline
  base_type norm_2(argument_type z) { return traits_type::abs(z); }

  static inline
  base_type norm_inf(argument_type z) {
    return std::max(NumericTraits<base_type>::abs(traits_type::real(z)),
		    NumericTraits<base_type>::abs(traits_type::imag(z)));
  }

  static inline
  bool equals(argument_type lhs, argument_type rhs) {
    return (traits_type::real(lhs) == traits_type::real(rhs))
        && (traits_type::imag(lhs) == traits_type::imag(rhs));
  }

  enum { is_complex = true };

  /** Complexity  on operations.*/
  enum {
    ops_plus = 2,	/**< Complexity on plus/minus ops. */
    ops_muls = 6	/**< Complexity on multiplications. */
  };
};


/**
 * \class NumericTraits< std::complex<float> > NumericTraits.h "tvmet/NumericTraits.h"
 * \brief Traits specialized for std::complex<float>.
 */
template<>
struct NumericTraits< std::complex<float> > {
  typedef float					base_type;
  typedef std::complex<float>			value_type;
  typedef std::complex<double>			sum_type;
  typedef std::complex<float>			diff_type;
  typedef std::complex<float>			float_type;
  typedef std::complex<float>			signed_type;

  typedef NumericTraits<value_type>		traits_type;
  typedef const value_type&			argument_type;

  static inline
  base_type real(argument_type z) { return std::real(z); }

  static inline
  base_type imag(argument_type z) { return std::imag(z); }

  static inline
  value_type conj(argument_type z) { return std::conj(z); }

  static inline
  base_type abs(argument_type z) { return std::abs(z); }

  static inline
  value_type sqrt(argument_type z) { return std::sqrt(z); }

  static inline
  base_type norm_1(argument_type z) {
    return NumericTraits<base_type>::abs((traits_type::real(z)))
         + NumericTraits<base_type>::abs((traits_type::imag(z)));
  }

  static inline
  base_type norm_2(argument_type z) { return traits_type::abs(z); }

  static inline
  base_type norm_inf(argument_type z) {
    return std::max(NumericTraits<base_type>::abs(traits_type::real(z)),
	            NumericTraits<base_type>::abs(traits_type::imag(z)));
  }

 static inline
  bool equals(argument_type lhs, argument_type rhs) {
    static base_type sqrt_epsilon(
      NumericTraits<base_type>::sqrt(
        std::numeric_limits<base_type>::epsilon()));

    return traits_type::norm_inf(lhs - rhs) < sqrt_epsilon *
      std::max(std::max(traits_type::norm_inf(lhs),
			traits_type::norm_inf(rhs)),
	       std::numeric_limits<base_type>::min());
  }

  enum { is_complex = true };

  /** Complexity on operations. */
  enum {
    ops_plus = 2,	/**< Complexity on plus/minus ops. */
    ops_muls = 6	/**< Complexity on multiplications. */
  };
};


/**
 * \class NumericTraits< std::complex<double> > NumericTraits.h "tvmet/NumericTraits.h"
 * \brief Traits specialized for std::complex<double>.
 */
template<>
struct NumericTraits< std::complex<double> > {
  typedef double				base_type;
  typedef std::complex<double>			value_type;
#if defined(TVMET_HAVE_LONG_DOUBLE)
  typedef std::complex<long double> 		sum_type;
#else
  typedef std::complex<double>			sum_type;
#endif
  typedef std::complex<double>			diff_type;
  typedef std::complex<double>			float_type;
  typedef std::complex<double>			signed_type;

  typedef NumericTraits<value_type>		traits_type;
  typedef const value_type&			argument_type;

  static inline
  base_type real(argument_type z) { return std::real(z); }

  static inline
  base_type imag(argument_type z) { return std::imag(z); }

  static inline
  value_type conj(argument_type z) { return std::conj(z); }

  static inline
  base_type abs(argument_type z) { return std::abs(z); }

  static inline
  value_type sqrt(argument_type z) { return std::sqrt(z); }

  static inline
  base_type norm_1(argument_type z) {
    return NumericTraits<base_type>::abs((traits_type::real(z)))
         + NumericTraits<base_type>::abs((traits_type::imag(z)));
  }

  static inline
  base_type norm_2(argument_type z) { return traits_type::abs(z); }

  static inline
  base_type norm_inf(argument_type z) {
    return std::max(NumericTraits<base_type>::abs(traits_type::real(z)),
	            NumericTraits<base_type>::abs(traits_type::imag(z)));
  }

 static inline
  bool equals(argument_type lhs, argument_type rhs) {
    static base_type sqrt_epsilon(
      NumericTraits<base_type>::sqrt(
        std::numeric_limits<base_type>::epsilon()));

    return traits_type::norm_inf(lhs - rhs) < sqrt_epsilon *
      std::max(std::max(traits_type::norm_inf(lhs),
			traits_type::norm_inf(rhs)),
	       std::numeric_limits<base_type>::min());
  }

  enum { is_complex = true };

  /** Complexity on operations. */
  enum {
    ops_plus = 2,	/**< Complexity on plus/minus ops. */
    ops_muls = 6	/**< Complexity on multiplications. */
  };
};


#if defined(TVMET_HAVE_LONG_DOUBLE)
/**
 * \class NumericTraits< std::complex<long double> > NumericTraits.h "tvmet/NumericTraits.h"
 * \brief Traits specialized for std::complex<double>.
 */
template<>
struct NumericTraits< std::complex<long double> > {
  typedef long double				base_type;
  typedef std::complex<long double>		value_type;
  typedef std::complex<long double> 		sum_type;
  typedef std::complex<long double>		diff_type;
  typedef std::complex<long double>		float_type;
  typedef std::complex<long double>		signed_type;

  typedef NumericTraits<value_type>		traits_type;
  typedef const value_type&			argument_type;

  static inline
  base_type real(argument_type z) { return std::real(z); }

  static inline
  base_type imag(argument_type z) { return std::imag(z); }

  static inline
  value_type conj(argument_type z) { return std::conj(z); }

  static inline
  base_type abs(argument_type z) { return std::abs(z); }

  static inline
  value_type sqrt(argument_type z) { return std::sqrt(z); }

  static inline
  base_type norm_1(argument_type z) {
    return NumericTraits<base_type>::abs((traits_type::real(z)))
         + NumericTraits<base_type>::abs((traits_type::imag(z)));
  }

  static inline
  base_type norm_2(argument_type z) { return traits_type::abs(z); }

  static inline
  base_type norm_inf(argument_type z) {
    return std::max(NumericTraits<base_type>::abs(traits_type::real(z)),
	            NumericTraits<base_type>::abs(traits_type::imag(z)));
  }

  static inline
  bool equals(argument_type lhs, argument_type rhs) {
    static base_type sqrt_epsilon(
      NumericTraits<base_type>::sqrt(
        std::numeric_limits<base_type>::epsilon()));

    return traits_type::norm_inf(lhs - rhs) < sqrt_epsilon *
      std::max(std::max(traits_type::norm_inf(lhs),
			traits_type::norm_inf(rhs)),
	       std::numeric_limits<base_type>::min());
  }

  enum { is_complex = true };

  /** Complexity on operations. */
  enum {
    ops_plus = 2,	/**< Complexity on plus/minus ops. */
    ops_muls = 6	/**< Complexity on multiplications. */
  };
};
#endif // defined(TVMET_HAVE_LONG_DOUBLE)


#endif // defined(TVMET_HAVE_COMPLEX)


} // namespace tvmet


#endif //  TVMET_NUMERIC_TRAITS_H


// Local Variables:
// mode:C++
// End:
