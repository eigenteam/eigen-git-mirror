/* This file is part of Eigen, a C++ template library for linear algebra
 * Copyright (C) 2007 Benoit Jacob <jacob@math.jussieu.fr>
 *
 * Based on Tvmet source code, http://tvmet.sourceforge.net,
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
 * $Id: TraitsBase.h,v 1.11 2004/11/04 18:10:35 opetzold Exp $
 */

#ifndef TVMET_NUMERIC_TRAITS_BASE_H
#define TVMET_NUMERIC_TRAITS_BASE_H

#if defined(EIGEN_USE_COMPLEX)
#  include <complex>
#endif

#include <cmath>
#include <cstdlib>

namespace tvmet {

/**
 * \class TraitsBase TraitsBase.h "tvmet/TraitsBase.h"
 * \brief Traits for standard types.
 *
 * In this base class goes the basic stuff that has to be implemented specifically
 * for each type.
 */
template<typename T>
struct TraitsBase {
  typedef T real_type;
  typedef T value_type;
  typedef T float_type;
  typedef const T & argument_type;

  static real_type  real(argument_type x);
  static real_type  imag(argument_type x);
  static value_type conj(argument_type x);
  static real_type  abs(argument_type x);
  static value_type sqrt(argument_type x);
  static real_type  epsilon();
  static bool isComplex();
  static bool isFloat();
  
  static bool isLessThan_nonfuzzy(argument_type x, argument_type y);
};

/*
 * numeric traits for built-in types
 */

/**
 * \class TraitsBase<int> TraitsBase.h "tvmet/TraitsBase.h"
 * \brief Traits specialized for int.
 */
template<>
struct TraitsBase<int> {
  typedef int value_type;
  typedef value_type real_type;
  typedef double float_type;
  typedef value_type argument_type;

  static real_type real(argument_type x) { return x; }
  static real_type imag(argument_type x) { TVMET_UNUSED(x); return 0; }
  static value_type conj(argument_type x) { return x; }
  static value_type sqrt(argument_type x) {
    return static_cast<value_type>(std::sqrt(static_cast<float_type>(x)));
  }
  static real_type abs(argument_type x) {
    return std::abs(x);
  }
  static real_type epsilon() { return 0; }
  static bool isComplex() { return false; }
  static bool isFloat() { return false; }
  
  /** Complexity on operations. */
  enum {
    ops_plus = 1,	/**< Complexity on plus/minus ops. */
    ops_muls = 1	/**< Complexity on multiplications. */
  };

  static bool isLessThan_nonfuzzy(argument_type x, argument_type y) {
    return x <= y;
  }
};

/**
 * \class TraitsBase<float> TraitsBase.h "tvmet/TraitsBase.h"
 * \brief Traits specialized for float.
 */
template<>
struct TraitsBase<float> {
  typedef float value_type;
  typedef value_type real_type;
  typedef value_type float_type;
  typedef value_type argument_type;

  static real_type real(argument_type x) { return x; }
  static real_type imag(argument_type x) { TVMET_UNUSED(x); return 0; }
  static value_type conj(argument_type x) { return x; }
  static value_type sqrt(argument_type x) {
    return std::sqrt(x);
  }
  static real_type abs(argument_type x) {
    return std::abs(x);
  }
  static real_type epsilon() { return 1e-5f; }
  static bool isComplex() { return false; }
  static bool isFloat() { return true; }

  /** Complexity on operations. */
  enum {
    ops_plus = 1,	/**< Complexity on plus/minus ops. */
    ops_muls = 1	/**< Complexity on multiplications. */
  };

  static bool isLessThan_nonfuzzy(argument_type x, argument_type y) {
    return x <= y;
  }
};


/**
 * \class TraitsBase<double> TraitsBase.h "tvmet/TraitsBase.h"
 * \brief Traits specialized for double.
 */
template<>
struct TraitsBase<double> {
  typedef double value_type;
  typedef value_type real_type;
  typedef value_type float_type;
  typedef value_type argument_type;

  static real_type real(argument_type x) { return x; }
  static real_type imag(argument_type x) { TVMET_UNUSED(x); return 0; }
  static value_type conj(argument_type x) { return x; }
  static value_type sqrt(argument_type x) {
    return std::sqrt(x);
  }
  static real_type abs(argument_type x) {
    return std::abs(x);
  }
  static real_type epsilon() { return 1e-11; }
  static bool isComplex() { return false; }
  static bool isFloat() { return true; }

  /** Complexity on operations. */
  enum {
    ops_plus = 1,	/**< Complexity on plus/minus ops. */
    ops_muls = 1	/**< Complexity on multiplications. */
  };

  static bool isLessThan_nonfuzzy(argument_type x, argument_type y) {
    return x <= y;
  }

};


/*
 * numeric traits for complex types
 */
#if defined(EIGEN_USE_COMPLEX)

/**
 * \class TraitsBase< std::complex<int> > TraitsBase.h "tvmet/TraitsBase.h"
 * \brief Traits specialized for std::complex<int>.
 */
template<>
struct TraitsBase< std::complex<int> >
{
  typedef std::complex<int> value_type;
  typedef int real_type;
  typedef std::complex<float> float_type;
  typedef const value_type& argument_type;

  static real_type real(argument_type z) { return std::real(z); }
  static real_type imag(argument_type z) { return std::imag(z); }
  static value_type conj(argument_type z) { return std::conj(z); }
  static real_type abs(argument_type x) {
    // the use of ceil() guarantees e.g. that abs(real(x)) <= abs(x),
    // and that abs(x)==0 if and only if x==0.
    return static_cast<int>(std::ceil(std::abs(float_type(x.real(),x.imag()))));
  }
  static value_type sqrt(argument_type x) {
    float_type y = std::sqrt(float_type(x.real(), x.imag()));
    int r = static_cast<int>(y.real());
    int i = static_cast<int>(y.imag());
    return value_type(r,i);
  }
  static real_type epsilon() { return 0; }
  static bool isComplex() { return true; }
  static bool isFloat() { return false; }
  
  /** Complexity on operations. */
  enum {
    ops_plus = 2,	/**< Complexity on plus/minus ops. */
    ops_muls = 6	/**< Complexity on multiplications. */
  };

  static bool isLessThan_nonfuzzy(argument_type x, argument_type y) {
    TVMET_UNUSED(x);
    TVMET_UNUSED(y);
    return false;
  }

};


/**
 * \class TraitsBase< std::complex<float> > TraitsBase.h "tvmet/TraitsBase.h"
 * \brief Traits specialized for std::complex<float>.
 */
template<>
struct TraitsBase< std::complex<float> > {
  typedef std::complex<float> value_type;
  typedef float real_type;
  typedef value_type float_type;
  typedef const value_type& argument_type;

  static real_type real(argument_type z) { return std::real(z); }
  static real_type imag(argument_type z) { return std::imag(z); }
  static value_type conj(argument_type z) { return std::conj(z); }
  static value_type sqrt(argument_type x) {
    return std::sqrt(x);
  }
  static real_type abs(argument_type x) {
    return std::abs(x);
  }
  static real_type epsilon() { return 1e-5f; }
  static bool isComplex() { return true; }
  static bool isFloat() { return true; }

  /** Complexity on operations. */
  enum {
    ops_plus = 2,	/**< Complexity on plus/minus ops. */
    ops_muls = 6	/**< Complexity on multiplications. */
  };

  static bool isLessThan_nonfuzzy(argument_type x, argument_type y) {
    TVMET_UNUSED(x);
    TVMET_UNUSED(y);
    return false;
  }

};


/**
 * \class TraitsBase< std::complex<double> > TraitsBase.h "tvmet/TraitsBase.h"
 * \brief Traits specialized for std::complex<double>.
 */
template<>
struct TraitsBase< std::complex<double> > {
  typedef std::complex<double> value_type;
  typedef double real_type;
  typedef value_type float_type;
  typedef const value_type& argument_type;

  static real_type real(argument_type z) { return std::real(z); }
  static real_type imag(argument_type z) { return std::imag(z); }
  static value_type conj(argument_type z) { return std::conj(z); }
  static value_type sqrt(argument_type x) {
    return std::sqrt(x);
  }
  static real_type abs(argument_type x) {
    return std::abs(x);
  }
  static real_type epsilon() { return 1e-11; }
  static bool isComplex() { return true; }
  static bool isFloat() { return true; }

  /** Complexity on operations. */
  enum {
    ops_plus = 2,	/**< Complexity on plus/minus ops. */
    ops_muls = 6	/**< Complexity on multiplications. */
  };

  static bool isLessThan_nonfuzzy(argument_type x, argument_type y) {
    TVMET_UNUSED(x);
    TVMET_UNUSED(y);
    return false;
  }

};

#endif // defined(EIGEN_USE_COMPLEX)

#ifdef __GNUC__
# if __GNUC__>=4
#  define EIGEN_WITH_GCC_4_OR_LATER
# endif
#endif

/** Stores in x a random int between -RAND_MAX/2 and RAND_MAX/2 */
inline void pickRandom( int & x )
{
    x = rand() - RAND_MAX / 2;
}

/** Stores in x a random float between -1.0 and 1.0 */
inline void pickRandom( float & x )
{
    x = 2.0f * rand() / RAND_MAX - 1.0f;
}

/** Stores in x a random double between -1.0 and 1.0 */
inline void pickRandom( double & x )
{
    x = 2.0 * rand() / RAND_MAX - 1.0;
}

#ifdef EIGEN_USE_COMPLEX
/** Stores in the real and imaginary parts of x
  * random values between -1.0 and 1.0 */
template<typename T> void pickRandom( std::complex<T> & x )
{
#ifdef EIGEN_WITH_GCC_4_OR_LATER
  pickRandom( x.real() );
  pickRandom( x.imag() );
#else // workaround by David Faure for MacOS 10.3 and GCC 3.3, commit 630812
  T r = x.real();
  T i = x.imag();
  pickRandom( r );
  pickRandom( i );
  x = std::complex<T>(r,i);
#endif
}
#endif // EIGEN_USE_COMPLEX

} // namespace tvmet

#endif //  TVMET_NUMERIC_TRAITS_BASE_H
