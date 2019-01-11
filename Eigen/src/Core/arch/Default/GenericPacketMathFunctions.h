// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2007 Julien Pommier
// Copyright (C) 2014 Pedro Gonnet (pedro.gonnet@gmail.com)
// Copyright (C) 2009-2018 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

/* The exp and log functions of this file initially come from
 * Julien Pommier's sse math library: http://gruntthepeon.free.fr/ssemath/
 */

namespace Eigen {
namespace internal {

template<typename Packet> EIGEN_STRONG_INLINE Packet
pfrexp_float(const Packet& a, Packet& exponent) {
  typedef typename unpacket_traits<Packet>::integer_packet PacketI;
  const Packet cst_126f = pset1<Packet>(126.0f);
  const Packet cst_half = pset1<Packet>(0.5f);
  const Packet cst_inv_mant_mask  = pset1frombits<Packet>(~0x7f800000u);
  exponent = psub(pcast<PacketI,Packet>(pshiftright<23>(preinterpret<PacketI>(a))), cst_126f);
  return por(pand(a, cst_inv_mant_mask), cst_half);
}

template<typename Packet> EIGEN_STRONG_INLINE Packet
pldexp_float(Packet a, Packet exponent)
{
  typedef typename unpacket_traits<Packet>::integer_packet PacketI;
  const Packet cst_127 = pset1<Packet>(127.f);
  // return a * 2^exponent
  PacketI ei = pcast<Packet,PacketI>(padd(exponent, cst_127));
  return pmul(a, preinterpret<Packet>(pshiftleft<23>(ei)));
}

// Natural logarithm
// Computes log(x) as log(2^e * m) = C*e + log(m), where the constant C =log(2)
// and m is in the range [sqrt(1/2),sqrt(2)). In this range, the logarithm can
// be easily approximated by a polynomial centered on m=1 for stability.
// TODO(gonnet): Further reduce the interval allowing for lower-degree
//               polynomial interpolants -> ... -> profit!
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
EIGEN_UNUSED
Packet plog_float(const Packet _x)
{
  Packet x = _x;

  const Packet cst_1              = pset1<Packet>(1.0f);
  const Packet cst_half           = pset1<Packet>(0.5f);
  // The smallest non denormalized float number.
  const Packet cst_min_norm_pos   = pset1frombits<Packet>( 0x00800000u);
  const Packet cst_minus_inf      = pset1frombits<Packet>( 0xff800000u);
  const Packet cst_pos_inf        = pset1frombits<Packet>( 0x7f800000u);

  // Polynomial coefficients.
  const Packet cst_cephes_SQRTHF = pset1<Packet>(0.707106781186547524f);
  const Packet cst_cephes_log_p0 = pset1<Packet>(7.0376836292E-2f);
  const Packet cst_cephes_log_p1 = pset1<Packet>(-1.1514610310E-1f);
  const Packet cst_cephes_log_p2 = pset1<Packet>(1.1676998740E-1f);
  const Packet cst_cephes_log_p3 = pset1<Packet>(-1.2420140846E-1f);
  const Packet cst_cephes_log_p4 = pset1<Packet>(+1.4249322787E-1f);
  const Packet cst_cephes_log_p5 = pset1<Packet>(-1.6668057665E-1f);
  const Packet cst_cephes_log_p6 = pset1<Packet>(+2.0000714765E-1f);
  const Packet cst_cephes_log_p7 = pset1<Packet>(-2.4999993993E-1f);
  const Packet cst_cephes_log_p8 = pset1<Packet>(+3.3333331174E-1f);
  const Packet cst_cephes_log_q1 = pset1<Packet>(-2.12194440e-4f);
  const Packet cst_cephes_log_q2 = pset1<Packet>(0.693359375f);

  // Truncate input values to the minimum positive normal.
  x = pmax(x, cst_min_norm_pos);

  Packet e;
  // extract significant in the range [0.5,1) and exponent
  x = pfrexp(x,e);

  // part2: Shift the inputs from the range [0.5,1) to [sqrt(1/2),sqrt(2))
  // and shift by -1. The values are then centered around 0, which improves
  // the stability of the polynomial evaluation.
  //   if( x < SQRTHF ) {
  //     e -= 1;
  //     x = x + x - 1.0;
  //   } else { x = x - 1.0; }
  Packet mask = pcmp_lt(x, cst_cephes_SQRTHF);
  Packet tmp = pand(x, mask);
  x = psub(x, cst_1);
  e = psub(e, pand(cst_1, mask));
  x = padd(x, tmp);

  Packet x2 = pmul(x, x);
  Packet x3 = pmul(x2, x);

  // Evaluate the polynomial approximant of degree 8 in three parts, probably
  // to improve instruction-level parallelism.
  Packet y, y1, y2;
  y  = pmadd(cst_cephes_log_p0, x, cst_cephes_log_p1);
  y1 = pmadd(cst_cephes_log_p3, x, cst_cephes_log_p4);
  y2 = pmadd(cst_cephes_log_p6, x, cst_cephes_log_p7);
  y  = pmadd(y, x, cst_cephes_log_p2);
  y1 = pmadd(y1, x, cst_cephes_log_p5);
  y2 = pmadd(y2, x, cst_cephes_log_p8);
  y  = pmadd(y, x3, y1);
  y  = pmadd(y, x3, y2);
  y  = pmul(y, x3);

  // Add the logarithm of the exponent back to the result of the interpolation.
  y1  = pmul(e, cst_cephes_log_q1);
  tmp = pmul(x2, cst_half);
  y   = padd(y, y1);
  x   = psub(x, tmp);
  y2  = pmul(e, cst_cephes_log_q2);
  x   = padd(x, y);
  x   = padd(x, y2);

  Packet invalid_mask = pcmp_lt_or_nan(_x, pzero(_x));
  Packet iszero_mask  = pcmp_eq(_x,pzero(_x));
  Packet pos_inf_mask = pcmp_eq(_x,cst_pos_inf);
  // Filter out invalid inputs, i.e.:
  //  - negative arg will be NAN
  //  - 0 will be -INF
  //  - +INF will be +INF
  return pselect(iszero_mask, cst_minus_inf,
                              por(pselect(pos_inf_mask,cst_pos_inf,x), invalid_mask));
}

// Exponential function. Works by writing "x = m*log(2) + r" where
// "m = floor(x/log(2)+1/2)" and "r" is the remainder. The result is then
// "exp(x) = 2^m*exp(r)" where exp(r) is in the range [-1,1).
template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
EIGEN_UNUSED
Packet pexp_float(const Packet _x)
{
  const Packet cst_1      = pset1<Packet>(1.0f);
  const Packet cst_half   = pset1<Packet>(0.5f);
  const Packet cst_exp_hi = pset1<Packet>( 88.3762626647950f);
  const Packet cst_exp_lo = pset1<Packet>(-88.3762626647949f);

  const Packet cst_cephes_LOG2EF = pset1<Packet>(1.44269504088896341f);
  const Packet cst_cephes_exp_p0 = pset1<Packet>(1.9875691500E-4f);
  const Packet cst_cephes_exp_p1 = pset1<Packet>(1.3981999507E-3f);
  const Packet cst_cephes_exp_p2 = pset1<Packet>(8.3334519073E-3f);
  const Packet cst_cephes_exp_p3 = pset1<Packet>(4.1665795894E-2f);
  const Packet cst_cephes_exp_p4 = pset1<Packet>(1.6666665459E-1f);
  const Packet cst_cephes_exp_p5 = pset1<Packet>(5.0000001201E-1f);

  // Clamp x.
  Packet x = pmax(pmin(_x, cst_exp_hi), cst_exp_lo);

  // Express exp(x) as exp(m*ln(2) + r), start by extracting
  // m = floor(x/ln(2) + 0.5).
  Packet m = pfloor(pmadd(x, cst_cephes_LOG2EF, cst_half));

  // Get r = x - m*ln(2). If no FMA instructions are available, m*ln(2) is
  // subtracted out in two parts, m*C1+m*C2 = m*ln(2), to avoid accumulating
  // truncation errors.
  Packet r;
#ifdef EIGEN_HAS_SINGLE_INSTRUCTION_MADD
  const Packet cst_nln2 = pset1<Packet>(-0.6931471805599453f);
  r = pmadd(m, cst_nln2, x);
#else
  const Packet cst_cephes_exp_C1 = pset1<Packet>(0.693359375f);
  const Packet cst_cephes_exp_C2 = pset1<Packet>(-2.12194440e-4f);
  r = psub(x, pmul(m, cst_cephes_exp_C1));
  r = psub(r, pmul(m, cst_cephes_exp_C2));
#endif

  Packet r2 = pmul(r, r);

  // TODO(gonnet): Split into odd/even polynomials and try to exploit
  //               instruction-level parallelism.
  Packet y = cst_cephes_exp_p0;
  y = pmadd(y, r, cst_cephes_exp_p1);
  y = pmadd(y, r, cst_cephes_exp_p2);
  y = pmadd(y, r, cst_cephes_exp_p3);
  y = pmadd(y, r, cst_cephes_exp_p4);
  y = pmadd(y, r, cst_cephes_exp_p5);
  y = pmadd(y, r2, r);
  y = padd(y, cst_1);

  // Return 2^m * exp(r).
  return pmax(pldexp(y,m), _x);
}

template <typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
EIGEN_UNUSED
Packet pexp_double(const Packet _x)
{
  Packet x = _x;

  const Packet cst_1 = pset1<Packet>(1.0);
  const Packet cst_2 = pset1<Packet>(2.0);
  const Packet cst_half = pset1<Packet>(0.5);

  const Packet cst_exp_hi = pset1<Packet>(709.437);
  const Packet cst_exp_lo = pset1<Packet>(-709.436139303);

  const Packet cst_cephes_LOG2EF = pset1<Packet>(1.4426950408889634073599);
  const Packet cst_cephes_exp_p0 = pset1<Packet>(1.26177193074810590878e-4);
  const Packet cst_cephes_exp_p1 = pset1<Packet>(3.02994407707441961300e-2);
  const Packet cst_cephes_exp_p2 = pset1<Packet>(9.99999999999999999910e-1);
  const Packet cst_cephes_exp_q0 = pset1<Packet>(3.00198505138664455042e-6);
  const Packet cst_cephes_exp_q1 = pset1<Packet>(2.52448340349684104192e-3);
  const Packet cst_cephes_exp_q2 = pset1<Packet>(2.27265548208155028766e-1);
  const Packet cst_cephes_exp_q3 = pset1<Packet>(2.00000000000000000009e0);
  const Packet cst_cephes_exp_C1 = pset1<Packet>(0.693145751953125);
  const Packet cst_cephes_exp_C2 = pset1<Packet>(1.42860682030941723212e-6);

  Packet tmp, fx;

  // clamp x
  x = pmax(pmin(x, cst_exp_hi), cst_exp_lo);
  // Express exp(x) as exp(g + n*log(2)).
  fx = pmadd(cst_cephes_LOG2EF, x, cst_half);

  // Get the integer modulus of log(2), i.e. the "n" described above.
  fx = pfloor(fx);

  // Get the remainder modulo log(2), i.e. the "g" described above. Subtract
  // n*log(2) out in two steps, i.e. n*C1 + n*C2, C1+C2=log2 to get the last
  // digits right.
  tmp = pmul(fx, cst_cephes_exp_C1);
  Packet z = pmul(fx, cst_cephes_exp_C2);
  x = psub(x, tmp);
  x = psub(x, z);

  Packet x2 = pmul(x, x);

  // Evaluate the numerator polynomial of the rational interpolant.
  Packet px = cst_cephes_exp_p0;
  px = pmadd(px, x2, cst_cephes_exp_p1);
  px = pmadd(px, x2, cst_cephes_exp_p2);
  px = pmul(px, x);

  // Evaluate the denominator polynomial of the rational interpolant.
  Packet qx = cst_cephes_exp_q0;
  qx = pmadd(qx, x2, cst_cephes_exp_q1);
  qx = pmadd(qx, x2, cst_cephes_exp_q2);
  qx = pmadd(qx, x2, cst_cephes_exp_q3);

  // I don't really get this bit, copied from the SSE2 routines, so...
  // TODO(gonnet): Figure out what is going on here, perhaps find a better
  // rational interpolant?
  x = pdiv(px, psub(qx, px));
  x = pmadd(cst_2, x, cst_1);

  // Construct the result 2^n * exp(g) = e * x. The max is used to catch
  // non-finite values in the input.
  return pmax(pldexp(x,fx), _x);
}

/* The code is the rewriting of the cephes sinf/cosf functions.
   Precision is excellent as long as x < 8192 (I did not bother to
   take into account the special handling they have for greater values
   -- it does not return garbage for arguments over 8192, though, but
   the extra precision is missing).

   Note that it is such that sinf((float)M_PI) = 8.74e-8, which is the
   surprising but correct result.
*/

template<bool ComputeSine,typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
EIGEN_UNUSED
Packet psincos_float(const Packet& _x)
{
  typedef typename unpacket_traits<Packet>::integer_packet PacketI;

  const Packet  cst_2oPI            = pset1<Packet>(0.636619746685028076171875f); // 2/PI
  const Packet  cst_rounding_magic  = pset1<Packet>(12582912); // 2^23 for rounding
  const PacketI csti_1              = pset1<PacketI>(1);
  const Packet  cst_sign_mask       = pset1frombits<Packet>(0x80000000u);

  Packet x = pabs(_x);

  // Scale x by 2/Pi to find x's octant.
  Packet y = pmul(x, cst_2oPI);

  // Rounding trick:
  Packet y_round = padd(y, cst_rounding_magic);
  PacketI y_int = preinterpret<PacketI>(y_round); // last 23 digits represent integer (if abs(x)<2^24)
  y = psub(y_round, cst_rounding_magic); // nearest integer to x*4/pi

  // Compute the sign to apply to the polynomial.
  // sin: sign = second_bit(y_int) xor signbit(_x)
  // cos: sign = second_bit(y_int+1)
  Packet sign_bit = ComputeSine ? pxor(_x, preinterpret<Packet>(pshiftleft<30>(y_int)))
                                : preinterpret<Packet>(pshiftleft<30>(padd(y_int,csti_1)));
  sign_bit = pand(sign_bit, cst_sign_mask); // clear all but left most bit

  // Get the polynomial selection mask from the second bit of y_int
  // We'll calculate both (sin and cos) polynomials and then select from the two.
  Packet poly_mask = preinterpret<Packet>(pcmp_eq(pand(y_int, csti_1), pzero(y_int)));

  // Reduce x by y octants to get: -Pi/4 <= x <= +Pi/4
  // using "Extended precision modular arithmetic"
  #if defined(EIGEN_HAS_SINGLE_INSTRUCTION_MADD)
  // This version requires true FMA for high accuracy
  // It provides a max error of 1ULP up to (with absolute_error < 5.9605e-08):
  const float huge_th = ComputeSine ? 117435.992f : 71476.0625f;
  x = pmadd(y, pset1<Packet>(-1.57079601287841796875f), x);
  x = pmadd(y, pset1<Packet>(-3.1391647326017846353352069854736328125e-07f), x);
  x = pmadd(y, pset1<Packet>(-5.390302529957764765544681040410068817436695098876953125e-15f), x);
  #else
  // Without true FMA, the previous set of coefficients maintain 1ULP accuracy
  // up to x<15.7 (for sin), but accuracy is immediately lost for x>15.7.
  // We thus use one more iteration to maintain 2ULPs up to reasonably large inputs.

  // The following set of coefficients maintain 1ULP up to 9.43 and 14.16 for sin and cos respectively.
  // and 2 ULP up to:
  const float huge_th = ComputeSine ? 25966.f : 18838.f;
  x = pmadd(y, pset1<Packet>(-1.5703125), x); // = 0xbfc90000
  x = pmadd(y, pset1<Packet>(-0.000483989715576171875), x); // = 0xb9fdc000
  x = pmadd(y, pset1<Packet>(1.62865035235881805419921875e-07), x); // = 0x342ee000
  x = pmadd(y, pset1<Packet>(5.5644315544167710640977020375430583953857421875e-11), x); // = 0x2e74b9ee

  // For the record, the following set of coefficients maintain 2ULP up
  // to a slightly larger range:
  // const float huge_th = ComputeSine ? 51981.f : 39086.125f;
  // but it slightly fails to maintain 1ULP for two values of sin below pi.
  // x = pmadd(y, pset1<Packet>(-3.140625/2.), x);
  // x = pmadd(y, pset1<Packet>(-0.00048351287841796875), x);
  // x = pmadd(y, pset1<Packet>(-3.13855707645416259765625e-07), x);
  // x = pmadd(y, pset1<Packet>(-6.0771006282767103812147979624569416046142578125e-11), x);

  // For the record, with only 3 iterations it is possible to maintain
  // 1 ULP up to 3PI (maybe more) and 2ULP up to 255.
  // The coefficients are: 0xbfc90f80, 0xb7354480, 0x2e74b9ee
  #endif

  // We use huge_vals as a temporary for abs(_x) to ensure huge_vals
  // is fully initialized for the last pselect(). (prevent compiler warning)
  Packet huge_vals = pabs(_x);
  Packet huge_mask = pcmp_le(pset1<Packet>(huge_th),huge_vals);
  
  if(predux_any(huge_mask))
  {
    const int PacketSize = unpacket_traits<Packet>::size;
    EIGEN_ALIGN_TO_BOUNDARY(sizeof(Packet)) float vals[PacketSize];
    pstoreu(vals, _x);
    for(int k=0; k<PacketSize;++k) {
      float val = vals[k];
      if(numext::abs(val)>=huge_th)  {
        vals[k] = ComputeSine ? std::sin(val) : std::cos(val);
      }
    }
    huge_vals = ploadu<Packet>(vals);
  }

  Packet x2 = pmul(x,x);

  // Evaluate the cos(x) polynomial. (-Pi/4 <= x <= Pi/4)
  Packet y1 =        pset1<Packet>(2.4372266125283204019069671630859375e-05f);
  y1 = pmadd(y1, x2, pset1<Packet>(-0.00138865201734006404876708984375f     ));
  y1 = pmadd(y1, x2, pset1<Packet>(0.041666619479656219482421875f           ));
  y1 = pmadd(y1, x2, pset1<Packet>(-0.5f));
  y1 = pmadd(y1, x2, pset1<Packet>(1.f));

  // Evaluate the sin(x) polynomial. (Pi/4 <= x <= Pi/4)
  // octave/matlab code to compute those coefficients:
  //    x = (0:0.0001:pi/4)';
  //    A = [x.^3 x.^5 x.^7];
  //    w = ((1.-(x/(pi/4)).^2).^5)*2000+1;         # weights trading relative accuracy
  //    c = (A'*diag(w)*A)\(A'*diag(w)*(sin(x)-x)); # weighted LS, linear coeff forced to 1
  //    printf('%.64f\n %.64f\n%.64f\n', c(3), c(2), c(1))
  //
  Packet y2 =        pset1<Packet>(-0.0001959234114083702898469196984621021329076029360294342041015625f);
  y2 = pmadd(y2, x2, pset1<Packet>( 0.0083326873655616851693794799871284340042620897293090820312500000f));
  y2 = pmadd(y2, x2, pset1<Packet>(-0.1666666203982298255503735617821803316473960876464843750000000000f));
  y2 = pmul(y2, x2);
  y2 = pmadd(y2, x, x);

  // Select the correct result from the two polynomials.
  y = ComputeSine ? pselect(poly_mask,y2,y1)
                  : pselect(poly_mask,y1,y2);

  // Update the sign and filter huge inputs
  return pselect(huge_mask, huge_vals, pxor(y, sign_bit));
}

template<typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
EIGEN_UNUSED
Packet psin_float(const Packet& x)
{
  return psincos_float<true>(x);
}

template<typename Packet>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
EIGEN_UNUSED
Packet pcos_float(const Packet& x)
{
  return psincos_float<false>(x);
}

} // end namespace internal
} // end namespace Eigen
