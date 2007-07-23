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

#if defined(EIGEN_USE_COMPLEX)
#  include <complex>
#endif

#include <cmath>
#include <limits>

#include <tvmet/CompileTimeError.h>
#include <tvmet/util/Random.h>

namespace tvmet {

/**
 * \class NumericTraits NumericTraits.h "tvmet/NumericTraits.h"
 * \brief Traits for integral types for operations.
 *
 * For each type we have to specialize this traits.
 */
template<class T>
struct NumericTraits {
  typedef T real_type;
  typedef T value_type;
  typedef T float_type;
  typedef const T & argument_type;

  static inline
  real_type real(argument_type x);

  static inline
  real_type imag(argument_type x);

  static inline
  value_type conj(argument_type x);

  static inline
  real_type abs(argument_type x);

  static inline
  value_type sqrt(argument_type x);

  enum{ is_complex = false };
};

/*
 * numeric traits for built-in types
 */

/**
 * \class NumericTraits<int> NumericTraits.h "tvmet/NumericTraits.h"
 * \brief Traits specialized for int.
 */
template<>
struct NumericTraits<int> {
  typedef int value_type;
  typedef value_type real_type;
  typedef double float_type;
  typedef value_type argument_type;

  static inline
  real_type real(argument_type x) { return x; }

  static inline
  real_type imag(argument_type x) { TVMET_UNUSED(x); return 0; }

  static inline
  value_type conj(argument_type x) { return x; }

  static inline
  value_type sqrt(argument_type x) {
    return static_cast<value_type>(std::sqrt(static_cast<float_type>(x)));
  }
  
  static inline
  value_type abs(argument_type x) {
    return std::abs(x);
  }

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
  typedef float value_type;
  typedef value_type real_type;
  typedef value_type float_type;
  typedef value_type argument_type;

  static inline
  real_type real(argument_type x) { return x; }

  static inline
  real_type imag(argument_type x) { TVMET_UNUSED(x); return 0; }

  static inline
  value_type conj(argument_type x) { return x; }

  static inline
  value_type sqrt(argument_type x) {
    return std::sqrt(x);
  }

  static inline
  value_type abs(argument_type x) {
    return std::abs(x);
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
  typedef double value_type;
  typedef value_type real_type;
  typedef value_type float_type;
  typedef value_type argument_type;

  static inline
  real_type real(argument_type x) { return x; }

  static inline
  real_type imag(argument_type x) { TVMET_UNUSED(x); return 0; }

  static inline
  value_type conj(argument_type x) { return x; }

  static inline
  value_type sqrt(argument_type x) {
    return std::sqrt(x);
  }

  static inline
  value_type abs(argument_type x) {
    return std::abs(x);
  }

  enum { is_complex = false };

  /** Complexity on operations. */
  enum {
    ops_plus = 1,	/**< Complexity on plus/minus ops. */
    ops_muls = 1	/**< Complexity on multiplications. */
  };
};


/*
 * numeric traits for complex types
 */
#if defined(EIGEN_USE_COMPLEX)

/**
 * \class NumericTraits< std::complex<int> > NumericTraits.h "tvmet/NumericTraits.h"
 * \brief Traits specialized for std::complex<int>.
 */
template<>
struct NumericTraits< std::complex<int> > {
  typedef std::complex<int> value_type;
  typedef int real_type;
  typedef std::complex<float> float_type;
  typedef const value_type& argument_type;

  static inline
  real_type real(argument_type z) { return std::real(z); }

  static inline
  real_type imag(argument_type z) { return std::imag(z); }

  static inline
  value_type conj(argument_type z) { return std::conj(z); }

  static inline
  real_type abs(argument_type z) {
    // the use of ceil() guarantees e.g. that abs(real(x)) <= abs(x),
    // and that abs(x)==0 if and only if x==0.
    return static_cast<value_type>(std::ceil(std::abs(static_cast<float_type>(x))));
  }

  static inline
  value_type sqrt(argument_type x) {
    return static_cast<value_type>(std::sqrt(static_cast<float_type>(x)));
  }

  enum { is_complex = true };

  /** Complexity on operations. */
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
  typedef std::complex<float> value_type;
  typedef float real_type;
  typedef value_type float_type;
  typedef const value_type& argument_type;

  static inline
  real_type real(argument_type z) { return std::real(z); }

  static inline
  real_type imag(argument_type z) { return std::imag(z); }

  static inline
  value_type conj(argument_type z) { return std::conj(z); }

  static inline
  value_type sqrt(argument_type x) {
    return std::sqrt(x);
  }

  static inline
  value_type abs(argument_type x) {
    return std::abs(x);
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
  typedef std::complex<double> value_type;
  typedef double real_type;
  typedef value_type float_type;
  typedef const value_type& argument_type;

  static inline
  real_type real(argument_type z) { return std::real(z); }

  static inline
  real_type imag(argument_type z) { return std::imag(z); }

  static inline
  value_type conj(argument_type z) { return std::conj(z); }

  static inline
  value_type sqrt(argument_type x) {
    return std::sqrt(x);
  }

  static inline
  value_type abs(argument_type x) {
    return std::abs(x);
  }

  enum { is_complex = true };

  /** Complexity on operations. */
  enum {
    ops_plus = 2,	/**< Complexity on plus/minus ops. */
    ops_muls = 6	/**< Complexity on multiplications. */
  };
};

#endif // defined(EIGEN_USE_COMPLEX)

} // namespace tvmet

#endif //  TVMET_NUMERIC_TRAITS_H
