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
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Scalar x, const Scalar coef[]) {
    EIGEN_STATIC_ASSERT((N > 0), YOU_MADE_A_PROGRAMMING_MISTAKE);

    return polevl<Scalar, N - 1>::run(x, coef) * x + coef[N];
  }
};

template <typename Scalar>
struct polevl<Scalar, 0> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Scalar, const Scalar coef[]) {
    return coef[0];
  }
};

}  // end namespace cephes

/****************************************************************************
 * Implementation of lgamma, requires C++11/C99                             *
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
 * Implementation of digamma (psi), based on Cephes                         *
 ****************************************************************************/

template <typename Scalar>
struct digamma_retval {
  typedef Scalar type;
};

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
    bool negative = false;

    const Scalar maxnum = NumTraits<Scalar>::infinity();
    const Scalar m_pi = Scalar(EIGEN_PI);

    const Scalar zero = Scalar(0);
    const Scalar one = Scalar(1);
    const Scalar half = Scalar(0.5);
    nz = zero;

    if (x <= zero) {
      negative = true;
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

/****************************************************************************
 * Implementation of erf, requires C++11/C99                                *
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
* Implementation of erfc, requires C++11/C99                               *
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

/**************************************************************************************************************
 * Implementation of igammac (complemented incomplete gamma integral), based on Cephes but requires C++11/C99 *
 **************************************************************************************************************/

template <typename Scalar>
struct igammac_retval {
  typedef Scalar type;
};

// NOTE: igamma_helper is also used to implement zeta
template <typename Scalar>
struct igamma_helper {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar machep() { assert(false && "machep not supported for this type"); return 0.0; }
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar big() { assert(false && "big not supported for this type"); return 0.0; }
};

template <>
struct igamma_helper<float> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE float machep() {
    return NumTraits<float>::epsilon() / 2;  // 1.0 - machep == 1.0
  }
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE float big() {
    // use epsneg (1.0 - epsneg == 1.0)
    return 1.0f / (NumTraits<float>::epsilon() / 2);
  }
};

template <>
struct igamma_helper<double> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE double machep() {
    return NumTraits<double>::epsilon() / 2;  // 1.0 - machep == 1.0
  }
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE double big() {
    return 1.0 / NumTraits<double>::epsilon();
  }
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
struct igammac_impl {
  EIGEN_DEVICE_FUNC
  static Scalar run(Scalar a, Scalar x) {
    /*  igamc()
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
    const Scalar nan = NumTraits<Scalar>::quiet_NaN();

    if ((x < zero) || (a <= zero)) {
      // domain error
      return nan;
    }

    if ((x < one) || (x < a)) {
      /* The checks above ensure that we meet the preconditions for
       * igamma_impl::Impl(), so call it, rather than igamma_impl::Run().
       * Calling Run() would also work, but in that case the compiler may not be
       * able to prove that igammac_impl::Run and igamma_impl::Run are not
       * mutually recursive.  This leads to worse code, particularly on
       * platforms like nvptx, where recursion is allowed only begrudgingly.
       */
      return (one - igamma_impl<Scalar>::Impl(a, x));
    }

    return Impl(a, x);
  }

 private:
  /* igamma_impl calls igammac_impl::Impl. */
  friend struct igamma_impl<Scalar>;

  /* Actually computes igamc(a, x).
   *
   * Preconditions:
   *   a > 0
   *   x >= 1
   *   x >= a
   */
  EIGEN_DEVICE_FUNC static Scalar Impl(Scalar a, Scalar x) {
    const Scalar zero = 0;
    const Scalar one = 1;
    const Scalar two = 2;
    const Scalar machep = igamma_helper<Scalar>::machep();
    const Scalar maxlog = numext::log(NumTraits<Scalar>::highest());
    const Scalar big = igamma_helper<Scalar>::big();
    const Scalar biginv = 1 / big;
    const Scalar inf = NumTraits<Scalar>::infinity();

    Scalar ans, ax, c, yc, r, t, y, z;
    Scalar pk, pkm1, pkm2, qk, qkm1, qkm2;

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
      if (numext::abs(pk) > big) {
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

/************************************************************************************************
 * Implementation of igamma (incomplete gamma integral), based on Cephes but requires C++11/C99 *
 ************************************************************************************************/

template <typename Scalar>
struct igamma_retval {
  typedef Scalar type;
};

#ifndef EIGEN_HAS_C99_MATH

template <typename Scalar>
struct igamma_impl {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(Scalar a, Scalar x) {
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
    /*	igam()
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
    const Scalar nan = NumTraits<Scalar>::quiet_NaN();

    if (x == zero) return zero;

    if ((x < zero) || (a <= zero)) {  // domain error
      return nan;
    }

    if ((x > one) && (x > a)) {
      /* The checks above ensure that we meet the preconditions for
       * igammac_impl::Impl(), so call it, rather than igammac_impl::Run().
       * Calling Run() would also work, but in that case the compiler may not be
       * able to prove that igammac_impl::Run and igamma_impl::Run are not
       * mutually recursive.  This leads to worse code, particularly on
       * platforms like nvptx, where recursion is allowed only begrudgingly.
       */
      return (one - igammac_impl<Scalar>::Impl(a, x));
    }

    return Impl(a, x);
  }

 private:
  /* igammac_impl calls igamma_impl::Impl. */
  friend struct igammac_impl<Scalar>;

  /* Actually computes igam(a, x).
   *
   * Preconditions:
   *   x > 0
   *   a > 0
   *   !(x > 1 && x > a)
   */
  EIGEN_DEVICE_FUNC static Scalar Impl(Scalar a, Scalar x) {
    const Scalar zero = 0;
    const Scalar one = 1;
    const Scalar machep = igamma_helper<Scalar>::machep();
    const Scalar maxlog = numext::log(NumTraits<Scalar>::highest());

    Scalar ans, ax, c, r;

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

/****************************************************************************
 * Implementation of Riemann zeta function of two arguments                 *
 ****************************************************************************/

template <typename Scalar>
struct zeta_retval {
    typedef Scalar type;
};

template <typename Scalar>
struct zeta_impl_series {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Scalar) {
    EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                        THIS_TYPE_IS_NOT_SUPPORTED);
    return Scalar(0);
  }
};

template <>
struct zeta_impl_series<float> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE bool run(float& a, float& b, float& s, const float x, const float machep) {
    int i = 0;  
    while(i < 9)
    {
        i += 1;
        a += 1.0f;
        b = numext::pow( a, -x );
        s += b;
        if( numext::abs(b/s) < machep )
            return true;
    }
    
    //Return whether we are done
    return false;
  }
};

template <>
struct zeta_impl_series<double> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE bool run(double& a, double& b, double& s, const double x, const double machep) {
    int i = 0;  
    while( (i < 9) || (a <= 9.0) )
    {
        i += 1;
        a += 1.0;
        b = numext::pow( a, -x );
        s += b;
        if( numext::abs(b/s) < machep )
            return true;
    }
    
    //Return whether we are done
    return false;
  }
};
    
template <typename Scalar>
struct zeta_impl {
    EIGEN_DEVICE_FUNC
    static Scalar run(Scalar x, Scalar q) {
        /*							zeta.c
         *
         *	Riemann zeta function of two arguments
         *
         *
         *
         * SYNOPSIS:
         *
         * double x, q, y, zeta();
         *
         * y = zeta( x, q );
         *
         *
         *
         * DESCRIPTION:
         *
         *
         *
         *                 inf.
         *                  -        -x
         *   zeta(x,q)  =   >   (k+q)
         *                  -
         *                 k=0
         *
         * where x > 1 and q is not a negative integer or zero.
         * The Euler-Maclaurin summation formula is used to obtain
         * the expansion
         *
         *                n
         *                -       -x
         * zeta(x,q)  =   >  (k+q)
         *                -
         *               k=1
         *
         *           1-x                 inf.  B   x(x+1)...(x+2j)
         *      (n+q)           1         -     2j
         *  +  ---------  -  -------  +   >    --------------------
         *        x-1              x      -                   x+2j+1
         *                   2(n+q)      j=1       (2j)! (n+q)
         *
         * where the B2j are Bernoulli numbers.  Note that (see zetac.c)
         * zeta(x,1) = zetac(x) + 1.
         *
         *
         *
         * ACCURACY:
         *
         * Relative error for single precision:
         * arithmetic   domain     # trials      peak         rms
         *    IEEE      0,25        10000       6.9e-7      1.0e-7
         *
         * Large arguments may produce underflow in powf(), in which
         * case the results are inaccurate.
         *
         * REFERENCE:
         *
         * Gradshteyn, I. S., and I. M. Ryzhik, Tables of Integrals,
         * Series, and Products, p. 1073; Academic Press, 1980.
         *
         */
        
        int i;
        Scalar p, r, a, b, k, s, t, w;
        
        const Scalar A[] = {
            Scalar(12.0),
            Scalar(-720.0),
            Scalar(30240.0),
            Scalar(-1209600.0),
            Scalar(47900160.0),
            Scalar(-1.8924375803183791606e9), /*1.307674368e12/691*/
            Scalar(7.47242496e10),
            Scalar(-2.950130727918164224e12), /*1.067062284288e16/3617*/
            Scalar(1.1646782814350067249e14), /*5.109094217170944e18/43867*/
            Scalar(-4.5979787224074726105e15), /*8.028576626982912e20/174611*/
            Scalar(1.8152105401943546773e17), /*1.5511210043330985984e23/854513*/
            Scalar(-7.1661652561756670113e18) /*1.6938241367317436694528e27/236364091*/
            };
            
        const Scalar maxnum = NumTraits<Scalar>::infinity();
        const Scalar zero = 0.0, half = 0.5, one = 1.0;
        const Scalar machep = igamma_helper<Scalar>::machep();
        const Scalar nan = NumTraits<Scalar>::quiet_NaN();
        
        if( x == one )
            return maxnum;
        
        if( x < one )
        {
            return nan;
        }
        
        if( q <= zero )
        {
            if(q == numext::floor(q))
            {
                return maxnum;
            }
            p = x;
            r = numext::floor(p);
            if (p != r)
                return nan;
        }
        
        /* Permit negative q but continue sum until n+q > +9 .
         * This case should be handled by a reflection formula.
         * If q<0 and x is an integer, there is a relation to
         * the polygamma function.
         */
        s = numext::pow( q, -x );
        a = q;
        b = zero;
        // Run the summation in a helper function that is specific to the floating precision
        if (zeta_impl_series<Scalar>::run(a, b, s, x, machep)) {
            return s;
        }
        
        w = a;
        s += b*w/(x-one);
        s -= half * b;
        a = one;
        k = zero;
        for( i=0; i<12; i++ )
        {
            a *= x + k;
            b /= w;
            t = a*b/A[i];
            s = s + t;
            t = numext::abs(t/s);
            if( t < machep )
                return s;
            k += one;
            a *= x + k;
            b /= w;
            k += one;
        }
        return s;
  }
};

/****************************************************************************
 * Implementation of polygamma function, requires C++11/C99                 *
 ****************************************************************************/

template <typename Scalar>
struct polygamma_retval {
    typedef Scalar type;
};
    
#ifndef EIGEN_HAS_C99_MATH
    
template <typename Scalar>
struct polygamma_impl {
    EIGEN_DEVICE_FUNC
    static EIGEN_STRONG_INLINE Scalar run(Scalar n, Scalar x) {
        EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                            THIS_TYPE_IS_NOT_SUPPORTED);
        return Scalar(0);
    }
};
    
#else
    
template <typename Scalar>
struct polygamma_impl {
    EIGEN_DEVICE_FUNC
    static Scalar run(Scalar n, Scalar x) {
        Scalar zero = 0.0, one = 1.0;
        Scalar nplus = n + one;
        const Scalar nan = NumTraits<Scalar>::quiet_NaN();
        
        // Check that n is an integer
        if (numext::floor(n) != n) {
            return nan;
        }
        // Just return the digamma function for n = 1
        else if (n == zero) {
            return digamma_impl<Scalar>::run(x);
        }
        // Use the same implementation as scipy
        else {
            Scalar factorial = numext::exp(lgamma_impl<Scalar>::run(nplus));
            return numext::pow(-one, nplus) * factorial * zeta_impl<Scalar>::run(nplus, x);
        }
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
EIGEN_DEVICE_FUNC inline EIGEN_MATHFUNC_RETVAL(zeta, Scalar)
zeta(const Scalar& x, const Scalar& q) {
    return EIGEN_MATHFUNC_IMPL(zeta, Scalar)::run(x, q);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline EIGEN_MATHFUNC_RETVAL(polygamma, Scalar)
polygamma(const Scalar& n, const Scalar& x) {
    return EIGEN_MATHFUNC_IMPL(polygamma, Scalar)::run(n, x);
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
