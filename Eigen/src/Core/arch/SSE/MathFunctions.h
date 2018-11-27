// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2007 Julien Pommier
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

/* The sin and cos and functions of this file come from
 * Julien Pommier's sse math library: http://gruntthepeon.free.fr/ssemath/
 */

#ifndef EIGEN_MATH_FUNCTIONS_SSE_H
#define EIGEN_MATH_FUNCTIONS_SSE_H

#include "../Default/GenericPacketMathFunctions.h"

namespace Eigen {

namespace internal {

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet4f plog<Packet4f>(const Packet4f& _x)
{
  return plog_float(_x);
}

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet4f pexp<Packet4f>(const Packet4f& _x)
{
  return pexp_float(_x);
}

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet2d pexp<Packet2d>(const Packet2d& x)
{
  return pexp_double(x);
}

/* evaluation of 4 sines at once, using SSE2 intrinsics.

   The code is the exact rewriting of the cephes sinf function.
   Precision is excellent as long as x < 8192 (I did not bother to
   take into account the special handling they have for greater values
   -- it does not return garbage for arguments over 8192, though, but
   the extra precision is missing).

   Note that it is such that sinf((float)M_PI) = 8.74e-8, which is the
   surprising but correct result.
*/

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet4f psin<Packet4f>(const Packet4f& _x)
{
  return psin_float(_x);
}

/* almost the same as psin */
template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet4f pcos<Packet4f>(const Packet4f& _x)
{
  Packet4f x = _x;
  _EIGEN_DECLARE_CONST_Packet4f(1 , 1.0f);
  _EIGEN_DECLARE_CONST_Packet4f(half, 0.5f);

  _EIGEN_DECLARE_CONST_Packet4i(1, 1);
  _EIGEN_DECLARE_CONST_Packet4i(not1, ~1);
  _EIGEN_DECLARE_CONST_Packet4i(2, 2);
  _EIGEN_DECLARE_CONST_Packet4i(4, 4);

  _EIGEN_DECLARE_CONST_Packet4f(minus_cephes_DP1,-0.78515625f);
  _EIGEN_DECLARE_CONST_Packet4f(minus_cephes_DP2, -2.4187564849853515625e-4f);
  _EIGEN_DECLARE_CONST_Packet4f(minus_cephes_DP3, -3.77489497744594108e-8f);
  _EIGEN_DECLARE_CONST_Packet4f(sincof_p0, -1.9515295891E-4f);
  _EIGEN_DECLARE_CONST_Packet4f(sincof_p1,  8.3321608736E-3f);
  _EIGEN_DECLARE_CONST_Packet4f(sincof_p2, -1.6666654611E-1f);
  _EIGEN_DECLARE_CONST_Packet4f(coscof_p0,  2.443315711809948E-005f);
  _EIGEN_DECLARE_CONST_Packet4f(coscof_p1, -1.388731625493765E-003f);
  _EIGEN_DECLARE_CONST_Packet4f(coscof_p2,  4.166664568298827E-002f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_FOPI, 1.27323954473516f); // 4 / M_PI

  Packet4f xmm1, xmm2, xmm3, y;
  Packet4i emm0, emm2;

  x = pabs(x);

  /* scale by 4/Pi */
  y = pmul(x, p4f_cephes_FOPI);

  /* get the integer part of y */
  emm2 = _mm_cvttps_epi32(y);
  /* j=(j+1) & (~1) (see the cephes sources) */
  emm2 = _mm_add_epi32(emm2, p4i_1);
  emm2 = _mm_and_si128(emm2, p4i_not1);
  y = _mm_cvtepi32_ps(emm2);

  emm2 = _mm_sub_epi32(emm2, p4i_2);

  /* get the swap sign flag */
  emm0 = _mm_andnot_si128(emm2, p4i_4);
  emm0 = _mm_slli_epi32(emm0, 29);
  /* get the polynom selection mask */
  emm2 = _mm_and_si128(emm2, p4i_2);
  emm2 = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());

  Packet4f sign_bit = _mm_castsi128_ps(emm0);
  Packet4f poly_mask = _mm_castsi128_ps(emm2);

  /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
  xmm1 = pmul(y, p4f_minus_cephes_DP1);
  xmm2 = pmul(y, p4f_minus_cephes_DP2);
  xmm3 = pmul(y, p4f_minus_cephes_DP3);
  x = padd(x, xmm1);
  x = padd(x, xmm2);
  x = padd(x, xmm3);

  /* Evaluate the first polynom  (0 <= x <= Pi/4) */
  y = p4f_coscof_p0;
  Packet4f z = pmul(x,x);

  y = pmadd(y,z,p4f_coscof_p1);
  y = pmadd(y,z,p4f_coscof_p2);
  y = pmul(y, z);
  y = pmul(y, z);
  Packet4f tmp = _mm_mul_ps(z, p4f_half);
  y = psub(y, tmp);
  y = padd(y, p4f_1);

  /* Evaluate the second polynom  (Pi/4 <= x <= 0) */
  Packet4f y2 = p4f_sincof_p0;
  y2 = pmadd(y2, z, p4f_sincof_p1);
  y2 = pmadd(y2, z, p4f_sincof_p2);
  y2 = pmul(y2, z);
  y2 = pmadd(y2, x, x);

  /* select the correct result from the two polynoms */
  y2 = _mm_and_ps(poly_mask, y2);
  y  = _mm_andnot_ps(poly_mask, y);
  y  = _mm_or_ps(y,y2);

  /* update the sign */
  return _mm_xor_ps(y, sign_bit);
}

#if EIGEN_FAST_MATH

// Functions for sqrt.
// The EIGEN_FAST_MATH version uses the _mm_rsqrt_ps approximation and one step
// of Newton's method, at a cost of 1-2 bits of precision as opposed to the
// exact solution. It does not handle +inf, or denormalized numbers correctly.
// The main advantage of this approach is not just speed, but also the fact that
// it can be inlined and pipelined with other computations, further reducing its
// effective latency. This is similar to Quake3's fast inverse square root.
// For detail see here: http://www.beyond3d.com/content/articles/8/
template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet4f psqrt<Packet4f>(const Packet4f& _x)
{
  Packet4f half = pmul(_x, pset1<Packet4f>(.5f));
  Packet4f denormal_mask = _mm_and_ps(
      _mm_cmpge_ps(_x, _mm_setzero_ps()),
      _mm_cmplt_ps(_x, pset1<Packet4f>((std::numeric_limits<float>::min)())));

  // Compute approximate reciprocal sqrt.
  Packet4f x = _mm_rsqrt_ps(_x);
  // Do a single step of Newton's iteration.
  x = pmul(x, psub(pset1<Packet4f>(1.5f), pmul(half, pmul(x,x))));
  // Flush results for denormals to zero.
  return _mm_andnot_ps(denormal_mask, pmul(_x,x));
}

#else

template<>EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet4f psqrt<Packet4f>(const Packet4f& x) { return _mm_sqrt_ps(x); }

#endif

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet2d psqrt<Packet2d>(const Packet2d& x) { return _mm_sqrt_pd(x); }

#if EIGEN_FAST_MATH

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet4f prsqrt<Packet4f>(const Packet4f& _x) {
  _EIGEN_DECLARE_CONST_Packet4f_FROM_INT(inf, 0x7f800000u);
  _EIGEN_DECLARE_CONST_Packet4f_FROM_INT(nan, 0x7fc00000u);
  _EIGEN_DECLARE_CONST_Packet4f(one_point_five, 1.5f);
  _EIGEN_DECLARE_CONST_Packet4f(minus_half, -0.5f);
  _EIGEN_DECLARE_CONST_Packet4f_FROM_INT(flt_min, 0x00800000u);

  Packet4f neg_half = pmul(_x, p4f_minus_half);

  // select only the inverse sqrt of positive normal inputs (denormals are
  // flushed to zero and cause infs as well).
  Packet4f le_zero_mask = _mm_cmple_ps(_x, p4f_flt_min);
  Packet4f x = _mm_andnot_ps(le_zero_mask, _mm_rsqrt_ps(_x));

  // Fill in NaNs and Infs for the negative/zero entries.
  Packet4f neg_mask = _mm_cmplt_ps(_x, _mm_setzero_ps());
  Packet4f zero_mask = _mm_andnot_ps(neg_mask, le_zero_mask);
  Packet4f infs_and_nans = _mm_or_ps(_mm_and_ps(neg_mask, p4f_nan),
                                     _mm_and_ps(zero_mask, p4f_inf));

  // Do a single step of Newton's iteration.
  x = pmul(x, pmadd(neg_half, pmul(x, x), p4f_one_point_five));

  // Insert NaNs and Infs in all the right places.
  return _mm_or_ps(x, infs_and_nans);
}

#else

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet4f prsqrt<Packet4f>(const Packet4f& x) {
  // Unfortunately we can't use the much faster mm_rqsrt_ps since it only provides an approximation.
  return _mm_div_ps(pset1<Packet4f>(1.0f), _mm_sqrt_ps(x));
}

#endif

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet2d prsqrt<Packet2d>(const Packet2d& x) {
  // Unfortunately we can't use the much faster mm_rqsrt_pd since it only provides an approximation.
  return _mm_div_pd(pset1<Packet2d>(1.0), _mm_sqrt_pd(x));
}

// Hyperbolic Tangent function.
template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED Packet4f
ptanh<Packet4f>(const Packet4f& x) {
  return internal::generic_fast_tanh_float(x);
}

} // end namespace internal

namespace numext {

template<>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
float sqrt(const float &x)
{
  return internal::pfirst(internal::Packet4f(_mm_sqrt_ss(_mm_set_ss(x))));
}

template<>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
double sqrt(const double &x)
{
#if EIGEN_COMP_GNUC_STRICT
  // This works around a GCC bug generating poor code for _mm_sqrt_pd
  // See https://bitbucket.org/eigen/eigen/commits/14f468dba4d350d7c19c9b93072e19f7b3df563b
  return internal::pfirst(internal::Packet2d(__builtin_ia32_sqrtsd(_mm_set_sd(x))));
#else
  return internal::pfirst(internal::Packet2d(_mm_sqrt_pd(_mm_set_sd(x))));
#endif
}

} // end namespace numex

} // end namespace Eigen

#endif // EIGEN_MATH_FUNCTIONS_SSE_H
