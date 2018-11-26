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

  Packet invalid_mask = pcmp_lt_or_nan(x, pzero(x));
  Packet iszero_mask  = pcmp_eq(x,pzero(x));

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

  // Filter out invalid inputs, i.e. negative arg will be NAN, 0 will be -INF.
  return pselect(iszero_mask, cst_minus_inf, por(x, invalid_mask));
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

} // end namespace internal
} // end namespace Eigen
