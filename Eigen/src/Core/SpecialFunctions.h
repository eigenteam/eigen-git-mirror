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

//  Parts of this code are based on the Cephes Math Library.
//
//  Cephes Math Library Release 2.8:  June, 2000
//  Copyright 1984, 1987, 1992, 2000 by Stephen L. Moshier
//
//  Permission has been kindly provided by the original author
//  to incorporate the Cephes software into the Eigen codebase:
//
//    From: Stephen Moshier
//    To: Eugene Brevdo
//    Subject: Re: Permission to wrap several cephes functions in Eigen
//
//    Hello Eugene,
//
//    Thank you for writing.
//
//    If your licensing is similar to BSD, the formal way that has been
//    handled is simply to add a statement to the effect that you are incorporating
//    the Cephes software by permission of the author.
//
//    Good luck with your project,
//    Steve

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
    EIGEN_STATIC_ASSERT((N > 0), YOU_MADE_A_PROGRAMMING_MISTAKE);

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
struct digamma_retval {
  typedef Scalar type;
};

#ifndef EIGEN_HAS_C99_MATH

template <typename Scalar>
struct digamma_impl {
  EIGEN_DEVICE_FUNC
  static Scalar run(Scalar x) {
    EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                        THIS_TYPE_IS_NOT_SUPPORTED);
    return Scalar(0);
  }
};

#else

/*
 *
 * Polynomial evaluation helper for the Psi (digamma) function.
 *
 * digamma_impl_maybe_poly::run(s) evaluates the asymptotic Psi expansion for
 * input Scalar s, assuming s is above 10.0.
 *
 * If s is above a certain threshold for the given Scalar type, zero
 * is returned.  Otherwise the polynomial is evaluated with enough
 * coefficients for results matching Scalar machine precision.
 *
 *
 */
template <typename Scalar>
struct digamma_impl_maybe_poly {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Scalar) {
    EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                        THIS_TYPE_IS_NOT_SUPPORTED);
    return Scalar(0);
  }
};


template <>
struct digamma_impl_maybe_poly<float> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE float run(const float s) {
    const float A[] = {
      -4.16666666666666666667E-3f,
      3.96825396825396825397E-3f,
      -8.33333333333333333333E-3f,
      8.33333333333333333333E-2f
    };

    float z;
    if (s < 1.0e8f) {
      z = 1.0f / (s * s);
      return z * cephes::polevl<float, 3>::run(z, A);
    } else return 0.0f;
  }
};

template <>
struct digamma_impl_maybe_poly<double> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE double run(const double s) {
    const double A[] = {
      8.33333333333333333333E-2,
      -2.10927960927960927961E-2,
      7.57575757575757575758E-3,
      -4.16666666666666666667E-3,
      3.96825396825396825397E-3,
      -8.33333333333333333333E-3,
      8.33333333333333333333E-2
    };

    double z;
    if (s < 1.0e17) {
      z = 1.0 / (s * s);
      return z * cephes::polevl<double, 6>::run(z, A);
    }
    else return 0.0;
  }
};

template <typename Scalar>
struct digamma_impl {
  EIGEN_DEVICE_FUNC
  static Scalar run(Scalar x) {
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
     * ACCURACY (float):
     *    Relative error (except absolute when |psi| < 1):
     * arithmetic   domain     # trials      peak         rms
     *    IEEE      0,30        30000       1.3e-15     1.4e-16
     *    IEEE      -30,0       40000       1.5e-15     2.2e-16
     *
     * ACCURACY (double):
     *    Absolute error,  relative when |psi| > 1 :
     * arithmetic   domain     # trials      peak         rms
     *    IEEE      -33,0        30000      8.2e-7      1.2e-7
     *    IEEE      0,33        100000      7.3e-7      7.7e-8
     *
     * ERROR MESSAGES:
     *     message         condition      value returned
     * psi singularity    x integer <=0      INFINITY
     */

    Scalar p, q, nz, s, w, y;
    bool negative;

    const Scalar maxnum = NumTraits<Scalar>::infinity();
    const Scalar m_pi = EIGEN_PI;

    negative = 0;
    nz = 0.0;

    const Scalar zero = 0.0;
    const Scalar one = 1.0;
    const Scalar half = 0.5;

    if (x <= zero) {
      negative = one;
      q = x;
      p = numext::floor(q);
      if (p == q) {
        return maxnum;
      }
      /* Remove the zeros of tan(m_pi x)
       * by subtracting the nearest integer from x
       */
      nz = q - p;
      if (nz != half) {
        if (nz > half) {
          p += one;
          nz = q - p;
        }
        nz = m_pi / numext::tan(m_pi * nz);
      }
      else {
        nz = zero;
      }
      x = one - x;
    }

    /* use the recurrence psi(x+1) = psi(x) + 1/x. */
    s = x;
    w = zero;
    while (s < Scalar(10)) {
      w += one / s;
      s += one;
    }

    y = digamma_impl_maybe_poly<Scalar>::run(s);

    y = numext::log(s) - (half / s) - y - w;

    return (negative) ? y - nz : y;
  }
};

#endif  // EIGEN_HAS_C99_MATH

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

/****************************************************************************
 * Implementation of igammac (complemented incomplete gamma integral)       *
 ****************************************************************************/

template <typename Scalar>
struct igammac_retval {
  typedef Scalar type;
};

#ifndef EIGEN_HAS_C99_MATH

template <typename Scalar>
struct igammac_impl {
  EIGEN_DEVICE_FUNC
  static Scalar run(Scalar a, Scalar x) {
    EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                        THIS_TYPE_IS_NOT_SUPPORTED);
    return Scalar(0);
  }
};

#else

template <typename Scalar> struct igamma_impl;  // predeclare igamma_impl

template <typename Scalar>
struct igamma_helper {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  static Scalar machep() { assert(false && "machep not supported for this type"); return 0.0; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  static Scalar big() { assert(false && "big not supported for this type"); return 0.0; }
};

template <>
struct igamma_helper<float> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  static float machep() {
    return NumTraits<float>::epsilon() / 2;  // 1.0 - machep == 1.0
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  static float big() {
    // use epsneg (1.0 - epsneg == 1.0)
    return 1.0 / (NumTraits<float>::epsilon() / 2);
  }
};

template <>
struct igamma_helper<double> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  static double machep() {
    return NumTraits<double>::epsilon() / 2;  // 1.0 - machep == 1.0
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  static double big() {
    return 1.0 / NumTraits<double>::epsilon();
  }
};

template <typename Scalar>
struct igammac_impl {
  EIGEN_DEVICE_FUNC
  static Scalar run(Scalar a, Scalar x) {
    /*							igamc()
     *
     *	Incomplete gamma integral (modified for Eigen)
     *
     *
     *
     * SYNOPSIS:
     *
     * double a, x, y, igamc();
     *
     * y = igamc( a, x );
     *
     * DESCRIPTION:
     *
     * The function is defined by
     *
     *
     *  igamc(a,x)   =   1 - igam(a,x)
     *
     *                            inf.
     *                              -
     *                     1       | |  -t  a-1
     *               =   -----     |   e   t   dt.
     *                    -      | |
     *                   | (a)    -
     *                             x
     *
     *
     * In this implementation both arguments must be positive.
     * The integral is evaluated by either a power series or
     * continued fraction expansion, depending on the relative
     * values of a and x.
     *
     * ACCURACY (float):
     *
     *                      Relative error:
     * arithmetic   domain     # trials      peak         rms
     *    IEEE      0,30        30000       7.8e-6      5.9e-7
     *
     *
     * ACCURACY (double):
     *
     * Tested at random a, x.
     *                a         x                      Relative error:
     * arithmetic   domain   domain     # trials      peak         rms
     *    IEEE     0.5,100   0,100      200000       1.9e-14     1.7e-15
     *    IEEE     0.01,0.5  0,100      200000       1.4e-13     1.6e-15
     *
     */
    /*
      Cephes Math Library Release 2.2: June, 1992
      Copyright 1985, 1987, 1992 by Stephen L. Moshier
      Direct inquiries to 30 Frost Street, Cambridge, MA 02140
    */
    const Scalar zero = 0;
    const Scalar one = 1;
    const Scalar two = 2;
    const Scalar machep = igamma_helper<Scalar>::machep();
    const Scalar maxlog = numext::log(NumTraits<Scalar>::highest());
    const Scalar big = igamma_helper<Scalar>::big();
    const Scalar biginv = 1 / big;
    const Scalar nan = NumTraits<Scalar>::quiet_NaN();
    const Scalar inf = NumTraits<Scalar>::infinity();

    Scalar ans, ax, c, yc, r, t, y, z;
    Scalar pk, pkm1, pkm2, qk, qkm1, qkm2;

    if ((x < zero) || ( a <= zero)) {
      // domain error
      return nan;
    }

    if ((x < one) || (x < a)) {
      return (one - igamma_impl<Scalar>::run(a, x));
    }

    if (x == inf) return zero;  // std::isinf crashes on CUDA

    /* Compute  x**a * exp(-x) / gamma(a)  */
    ax = a * numext::log(x) - x - lgamma_impl<Scalar>::run(a);
    if (ax < -maxlog) {  // underflow
      return zero;
    }
    ax = numext::exp(ax);

    // continued fraction
    y = one - a;
    z = x + y + one;
    c = zero;
    pkm2 = one;
    qkm2 = x;
    pkm1 = x + one;
    qkm1 = z * x;
    ans = pkm1 / qkm1;

    while (true) {
      c += one;
      y += one;
      z += two;
      yc = y * c;
      pk = pkm1 * z - pkm2 * yc;
      qk = qkm1 * z - qkm2 * yc;
      if (qk != zero) {
        r = pk / qk;
        t = numext::abs((ans - r) / r);
        ans = r;
      } else {
        t = one;
      }
      pkm2 = pkm1;
      pkm1 = pk;
      qkm2 = qkm1;
      qkm1 = qk;
      if (abs(pk) > big) {
        pkm2 *= biginv;
        pkm1 *= biginv;
        qkm2 *= biginv;
        qkm1 *= biginv;
      }
      if (t <= machep) break;
    }

    return (ans * ax);
  }
};

#endif  // EIGEN_HAS_C99_MATH

/****************************************************************************
 * Implementation of igamma (incomplete gamma integral)                     *
 ****************************************************************************/

template <typename Scalar>
struct igamma_retval {
  typedef Scalar type;
};

#ifndef EIGEN_HAS_C99_MATH

template <typename Scalar>
struct igamma_impl {
  EIGEN_DEVICE_FUNC
  static Scalar run(Scalar a, Scalar x) {
    EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                        THIS_TYPE_IS_NOT_SUPPORTED);
    return Scalar(0);
  }
};

#else

template <typename Scalar>
struct igamma_impl {
  EIGEN_DEVICE_FUNC
  static Scalar run(Scalar a, Scalar x) {
    /*							igam()
     *	Incomplete gamma integral
     *
     *
     *
     * SYNOPSIS:
     *
     * double a, x, y, igam();
     *
     * y = igam( a, x );
     *
     * DESCRIPTION:
     *
     * The function is defined by
     *
     *                           x
     *                            -
     *                   1       | |  -t  a-1
     *  igam(a,x)  =   -----     |   e   t   dt.
     *                  -      | |
     *                 | (a)    -
     *                           0
     *
     *
     * In this implementation both arguments must be positive.
     * The integral is evaluated by either a power series or
     * continued fraction expansion, depending on the relative
     * values of a and x.
     *
     * ACCURACY (double):
     *
     *                      Relative error:
     * arithmetic   domain     # trials      peak         rms
     *    IEEE      0,30       200000       3.6e-14     2.9e-15
     *    IEEE      0,100      300000       9.9e-14     1.5e-14
     *
     *
     * ACCURACY (float):
     *
     *                      Relative error:
     * arithmetic   domain     # trials      peak         rms
     *    IEEE      0,30        20000       7.8e-6      5.9e-7
     *
     */
    /*
      Cephes Math Library Release 2.2: June, 1992
      Copyright 1985, 1987, 1992 by Stephen L. Moshier
      Direct inquiries to 30 Frost Street, Cambridge, MA 02140
    */


    /* left tail of incomplete gamma function:
     *
     *          inf.      k
     *   a  -x   -       x
     *  x  e     >   ----------
     *           -     -
     *          k=0   | (a+k+1)
     *
     */
    const Scalar zero = 0;
    const Scalar one = 1;
    const Scalar machep = igamma_helper<Scalar>::machep();
    const Scalar maxlog = numext::log(NumTraits<Scalar>::highest());
    const Scalar nan = NumTraits<Scalar>::quiet_NaN();

    double ans, ax, c, r;

    if (x == zero) return zero;

    if ((x < zero) || ( a <= zero)) {  // domain error
      return nan;
    }

    if ((x > one) && (x > a)) {
      return (one - igammac_impl<Scalar>::run(a, x));
    }

    /* Compute  x**a * exp(-x) / gamma(a)  */
    ax = a * numext::log(x) - x - lgamma_impl<Scalar>::run(a);
    if (ax < -maxlog) {
      // underflow
      return zero;
    }
    ax = numext::exp(ax);

    /* power series */
    r = a;
    c = one;
    ans = one;

    while (true) {
      r += one;
      c *= x/r;
      ans += c;
      if (c/ans <= machep) break;
    }

    return (ans * ax / a);
  }
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

template <typename Scalar>
EIGEN_DEVICE_FUNC inline EIGEN_MATHFUNC_RETVAL(igamma, Scalar)
    igamma(const Scalar& a, const Scalar& x) {
  return EIGEN_MATHFUNC_IMPL(igamma, Scalar)::run(a, x);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline EIGEN_MATHFUNC_RETVAL(igammac, Scalar)
    igammac(const Scalar& a, const Scalar& x) {
  return EIGEN_MATHFUNC_IMPL(igammac, Scalar)::run(a, x);
}

}  // end namespace numext


}  // end namespace Eigen

#endif  // EIGEN_SPECIAL_FUNCTIONS_H
