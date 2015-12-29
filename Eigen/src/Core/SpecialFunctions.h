// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Eugene Brevdo <ebrevdo@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPECIAL_FUNCTIONS_H
#define EIGEN_SPECIAL_FUNCTIONS_H

namespace Eigen {
namespace internal {

namespace cephes {

/* polevl (modified for Eigen)
 *
 *      Evaluate polynomial
 *
 *
 *
 * SYNOPSIS:
 *
 * int N;
 * Scalar x, y, coef[N+1];
 *
 * y = polevl<decltype(x), N>( x, coef);
 *
 *
 *
 * DESCRIPTION:
 *
 * Evaluates polynomial of degree N:
 *
 *                     2          N
 * y  =  C  + C x + C x  +...+ C x
 *        0    1     2          N
 *
 * Coefficients are stored in reverse order:
 *
 * coef[0] = C  , ..., coef[N] = C  .
 *            N                   0
 *
 *  The function p1evl() assumes that coef[N] = 1.0 and is
 * omitted from the array.  Its calling arguments are
 * otherwise the same as polevl().
 *
 *
 * The Eigen implementation is templatized.  For best speed, store
 * coef as a const array (constexpr), e.g.
 *
 * const double coef[] = {1.0, 2.0, 3.0, ...};
 *
 */
template <typename Scalar, int N>
struct polevl {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  static Scalar run(const Scalar x, const Scalar coef[]) {
    EIGEN_STATIC_ASSERT(N > 0, YOU_MADE_A_PROGRAMMING_MISTAKE);

    return polevl<Scalar, N - 1>::run(x, coef) * x + coef[N];
  }
};

template <typename Scalar>
struct polevl<Scalar, 0> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  static Scalar run(const Scalar, const Scalar coef[]) {
    return coef[0];
  }
};

}  // end namespace cephes

/****************************************************************************
 * Implementation of lgamma                                                 *
 ****************************************************************************/

template <typename Scalar>
struct lgamma_impl {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Scalar) {
    EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                        THIS_TYPE_IS_NOT_SUPPORTED);
    return Scalar(0);
  }
};

template <typename Scalar>
struct lgamma_retval {
  typedef Scalar type;
};

#ifdef EIGEN_HAS_C99_MATH
template <>
struct lgamma_impl<float> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE float run(float x) { return ::lgammaf(x); }
};

template <>
struct lgamma_impl<double> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE double run(double x) { return ::lgamma(x); }
};
#endif

/****************************************************************************
 * Implementation of digamma (psi)                                          *
 ****************************************************************************/

template <typename Scalar>
struct digamma_impl {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Scalar) {
    EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                        THIS_TYPE_IS_NOT_SUPPORTED);
    return Scalar(0);
  }
};

template <typename Scalar>
struct digamma_retval {
  typedef Scalar type;
};

#ifdef EIGEN_HAS_C99_MATH
template <>
struct digamma_impl<float> {
  /*
   * Psi (digamma) function (modified for Eigen)
   *
   *
   * SYNOPSIS:
   *
   * float x, y, psif();
   *
   * y = psif( x );
   *
   *
   * DESCRIPTION:
   *
   *              d      -
   *   psi(x)  =  -- ln | (x)
   *              dx
   *
   * is the logarithmic derivative of the gamma function.
   * For integer x,
   *                   n-1
   *                    -
   * psi(n) = -EUL  +   >  1/k.
   *                    -
   *                   k=1
   *
   * If x is negative, it is transformed to a positive argument by the
   * reflection formula  psi(1-x) = psi(x) + pi cot(pi x).
   * For general positive x, the argument is made greater than 10
   * using the recurrence  psi(x+1) = psi(x) + 1/x.
   * Then the following asymptotic expansion is applied:
   *
   *                           inf.   B
   *                            -      2k
   * psi(x) = log(x) - 1/2x -   >   -------
   *                            -        2k
   *                           k=1   2k x
   *
   * where the B2k are Bernoulli numbers.
   *
   * ACCURACY:
   *    Absolute error,  relative when |psi| > 1 :
   * arithmetic   domain     # trials      peak         rms
   *    IEEE      -33,0        30000      8.2e-7      1.2e-7
   *    IEEE      0,33        100000      7.3e-7      7.7e-8
   *
   * ERROR MESSAGES:
   *     message         condition      value returned
   * psi singularity    x integer <=0      INFINITY
   */
  /*
    Cephes Math Library Release 2.2:  June, 1992
    Copyright 1984, 1987, 1992 by Stephen L. Moshier
    Direct inquiries to 30 Frost Street, Cambridge, MA 02140
  */
  EIGEN_DEVICE_FUNC
  static float run(float xx) {
    float p, q, nz, x, s, w, y, z;
    bool negative;

    // Some necessary constants
    const float M_PIF = 3.141592653589793238;
    const float MAXNUMF = std::numeric_limits<float>::infinity();

    const float A[] = {
      -4.16666666666666666667E-3,
      3.96825396825396825397E-3,
      -8.33333333333333333333E-3,
      8.33333333333333333333E-2
    };

    x = xx;
    nz = 0.0f;
    negative = 0;
    if (x <= 0.0f) {
      negative = 1;
      q = x;
      p = ::floor(q);
      if (p == q) {
        return (MAXNUMF);
      }
      nz = q - p;
      if (nz != 0.5f) {
        if (nz > 0.5f) {
          p += 1.0f;
          nz = q - p;
        }
        nz = M_PIF / ::tan(M_PIF * nz);
      } else {
        nz = 0.0f;
      }
      x = 1.0f - x;
    }

    /* use the recurrence psi(x+1) = psi(x) + 1/x. */
    s = x;
    w = 0.0f;
    while (s < 10.0f) {
      w += 1.0f / s;
      s += 1.0f;
    }

    if (s < 1.0e8f) {
      z = 1.0f / (s * s);
      y = z * cephes::polevl<float, 3>::run(z, A);
    } else
      y = 0.0f;

    y = ::log(s) - (0.5f / s) - y - w;

    return (negative) ? y - nz : y;
  }
};

template <>
struct digamma_impl<double> {
  EIGEN_DEVICE_FUNC
  static double run(double x) {
    /*
     *
     *     Psi (digamma) function (modified for Eigen)
     *
     *
     * SYNOPSIS:
     *
     * double x, y, psi();
     *
     * y = psi( x );
     *
     *
     * DESCRIPTION:
     *
     *              d      -
     *   psi(x)  =  -- ln | (x)
     *              dx
     *
     * is the logarithmic derivative of the gamma function.
     * For integer x,
     *                   n-1
     *                    -
     * psi(n) = -EUL  +   >  1/k.
     *                    -
     *                   k=1
     *
     * If x is negative, it is transformed to a positive argument by the
     * reflection formula  psi(1-x) = psi(x) + pi cot(pi x).
     * For general positive x, the argument is made greater than 10
     * using the recurrence  psi(x+1) = psi(x) + 1/x.
     * Then the following asymptotic expansion is applied:
     *
     *                           inf.   B
     *                            -      2k
     * psi(x) = log(x) - 1/2x -   >   -------
     *                            -        2k
     *                           k=1   2k x
     *
     * where the B2k are Bernoulli numbers.
     *
     * ACCURACY:
     *    Relative error (except absolute when |psi| < 1):
     * arithmetic   domain     # trials      peak         rms
     *    IEEE      0,30        30000       1.3e-15     1.4e-16
     *    IEEE      -30,0       40000       1.5e-15     2.2e-16
     *
     * ERROR MESSAGES:
     *     message         condition      value returned
     * psi singularity    x integer <=0      INFINITY
     */

    /*
     * Cephes Math Library Release 2.8:  June, 2000
     * Copyright 1984, 1987, 1992, 2000 by Stephen L. Moshier
     */
    double p, q, nz, s, w, y, z;
    bool negative;

    const double A[] = {
      8.33333333333333333333E-2,
      -2.10927960927960927961E-2,
      7.57575757575757575758E-3,
      -4.16666666666666666667E-3,
      3.96825396825396825397E-3,
      -8.33333333333333333333E-3,
      8.33333333333333333333E-2
    };

    const double MAXNUM = std::numeric_limits<double>::infinity();
    const double M_PI = 3.14159265358979323846;

    negative = 0;
    nz = 0.0;

    if (x <= 0.0) {
      negative = 1;
      q = x;
      p = ::floor(q);
      if (p == q) {
        return MAXNUM;
      }
      /* Remove the zeros of tan(M_PI x)
       * by subtracting the nearest integer from x
       */
      nz = q - p;
      if (nz != 0.5) {
        if (nz > 0.5) {
          p += 1.0;
          nz = q - p;
        }
        nz = M_PI / ::tan(M_PI * nz);
      }
      else {
        nz = 0.0;
      }
      x = 1.0 - x;
    }

    /* use the recurrence psi(x+1) = psi(x) + 1/x. */
    s = x;
    w = 0.0;
    while (s < 10.0) {
      w += 1.0 / s;
      s += 1.0;
    }

    if (s < 1.0e17) {
      z = 1.0 / (s * s);
      y = z * cephes::polevl<double, 6>::run(z, A);
    }
    else
      y = 0.0;

    y = ::log(s) - (0.5 / s) - y - w;

    return (negative) ? y - nz : y;
  }
};

#endif

/****************************************************************************
 * Implementation of erf                                                    *
 ****************************************************************************/

template <typename Scalar>
struct erf_impl {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Scalar) {
    EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                        THIS_TYPE_IS_NOT_SUPPORTED);
    return Scalar(0);
  }
};

template <typename Scalar>
struct erf_retval {
  typedef Scalar type;
};

#ifdef EIGEN_HAS_C99_MATH
template <>
struct erf_impl<float> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE float run(float x) { return ::erff(x); }
};

template <>
struct erf_impl<double> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE double run(double x) { return ::erf(x); }
};
#endif  // EIGEN_HAS_C99_MATH

/***************************************************************************
* Implementation of erfc                                                   *
****************************************************************************/

template <typename Scalar>
struct erfc_impl {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Scalar) {
    EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                        THIS_TYPE_IS_NOT_SUPPORTED);
    return Scalar(0);
  }
};

template <typename Scalar>
struct erfc_retval {
  typedef Scalar type;
};

#ifdef EIGEN_HAS_C99_MATH
template <>
struct erfc_impl<float> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE float run(const float x) { return ::erfcf(x); }
};

template <>
struct erfc_impl<double> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE double run(const double x) { return ::erfc(x); }
};
#endif  // EIGEN_HAS_C99_MATH

}  // end namespace internal

namespace numext {

template <typename Scalar>
EIGEN_DEVICE_FUNC inline EIGEN_MATHFUNC_RETVAL(lgamma, Scalar)
    lgamma(const Scalar& x) {
  return EIGEN_MATHFUNC_IMPL(lgamma, Scalar)::run(x);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline EIGEN_MATHFUNC_RETVAL(digamma, Scalar)
    digamma(const Scalar& x) {
  return EIGEN_MATHFUNC_IMPL(digamma, Scalar)::run(x);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline EIGEN_MATHFUNC_RETVAL(erf, Scalar)
    erf(const Scalar& x) {
  return EIGEN_MATHFUNC_IMPL(erf, Scalar)::run(x);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline EIGEN_MATHFUNC_RETVAL(erfc, Scalar)
    erfc(const Scalar& x) {
  return EIGEN_MATHFUNC_IMPL(erfc, Scalar)::run(x);
}

}  // end namespace numext

}  // end namespace Eigen

#endif  // EIGEN_SPECIAL_FUNCTIONS_H
