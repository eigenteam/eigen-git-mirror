// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2007 Julien Pommier
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2016 Konstantinos Margaritis <markos@freevec.org>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

/* The exp function of this file comes from
 * Julien Pommier's sse math library: http://gruntthepeon.free.fr/ssemath/
 */

#ifndef EIGEN_MATH_FUNCTIONS_ALTIVEC_H
#define EIGEN_MATH_FUNCTIONS_ALTIVEC_H

#include "../Default/GenericPacketMathFunctions.h"

namespace Eigen {

namespace internal {

#ifdef __VSX__
static _EIGEN_DECLARE_CONST_Packet2d(1 , 1.0);
static _EIGEN_DECLARE_CONST_Packet2d(2 , 2.0);
static _EIGEN_DECLARE_CONST_Packet2d(half, 0.5);

static _EIGEN_DECLARE_CONST_Packet2d(exp_hi,  709.437);
static _EIGEN_DECLARE_CONST_Packet2d(exp_lo, -709.436139303);

static _EIGEN_DECLARE_CONST_Packet2d(cephes_LOG2EF, 1.4426950408889634073599);

static _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_p0, 1.26177193074810590878e-4);
static _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_p1, 3.02994407707441961300e-2);
static _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_p2, 9.99999999999999999910e-1);

static _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_q0, 3.00198505138664455042e-6);
static _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_q1, 2.52448340349684104192e-3);
static _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_q2, 2.27265548208155028766e-1);
static _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_q3, 2.00000000000000000009e0);

static _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_C1, 0.693145751953125);
static _EIGEN_DECLARE_CONST_Packet2d(cephes_exp_C2, 1.42860682030941723212e-6);

#ifdef __POWER8_VECTOR__
static Packet2l p2l_1023 = { 1023, 1023 };
static Packet2ul p2ul_52 = { 52, 52 };
#endif

#endif

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

#ifndef EIGEN_COMP_CLANG
template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet4f prsqrt<Packet4f>(const Packet4f& x)
{
  return  vec_rsqrt(x);
}
#endif

#ifdef __VSX__
#ifndef EIGEN_COMP_CLANG
template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet2d prsqrt<Packet2d>(const Packet2d& x)
{
  return  vec_rsqrt(x);
}
#endif

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet4f psqrt<Packet4f>(const Packet4f& x)
{
  return  vec_sqrt(x);
}

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet2d psqrt<Packet2d>(const Packet2d& x)
{
  return  vec_sqrt(x);
}

// VSX support varies between different compilers and even different
// versions of the same compiler.  For gcc version >= 4.9.3, we can use
// vec_cts to efficiently convert Packet2d to Packet2l.  Otherwise, use
// a slow version that works with older compilers. 
// Update: apparently vec_cts/vec_ctf intrinsics for 64-bit doubles
// are buggy, https://gcc.gnu.org/bugzilla/show_bug.cgi?id=70963
static inline Packet2l ConvertToPacket2l(const Packet2d& x) {
#if EIGEN_GNUC_AT_LEAST(5, 4) || \
    (EIGEN_GNUC_AT(6, 1) && __GNUC_PATCHLEVEL__ >= 1)
  return vec_cts(x, 0);    // TODO: check clang version.
#else
  double tmp[2];
  memcpy(tmp, &x, sizeof(tmp));
  Packet2l l = { static_cast<long long>(tmp[0]),
                 static_cast<long long>(tmp[1]) };
  return l;
#endif
}

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet2d pexp<Packet2d>(const Packet2d& _x)
{
  Packet2d x = _x;

  Packet2d tmp, fx;
  Packet2l emm0;

  // clamp x
  x = pmax(pmin(x, p2d_exp_hi), p2d_exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
  fx = pmadd(x, p2d_cephes_LOG2EF, p2d_half);

  fx = pfloor(fx);

  tmp = pmul(fx, p2d_cephes_exp_C1);
  Packet2d z = pmul(fx, p2d_cephes_exp_C2);
  x = psub(x, tmp);
  x = psub(x, z);

  Packet2d x2 = pmul(x,x);

  Packet2d px = p2d_cephes_exp_p0;
  px = pmadd(px, x2, p2d_cephes_exp_p1);
  px = pmadd(px, x2, p2d_cephes_exp_p2);
  px = pmul (px, x);

  Packet2d qx = p2d_cephes_exp_q0;
  qx = pmadd(qx, x2, p2d_cephes_exp_q1);
  qx = pmadd(qx, x2, p2d_cephes_exp_q2);
  qx = pmadd(qx, x2, p2d_cephes_exp_q3);

  x = pdiv(px,psub(qx,px));
  x = pmadd(p2d_2,x,p2d_1);

  // build 2^n
  emm0 = ConvertToPacket2l(fx);

#ifdef __POWER8_VECTOR__ 
  emm0 = vec_add(emm0, p2l_1023);
  emm0 = vec_sl(emm0, p2ul_52);
#else
  // Code is a bit complex for POWER7.  There is actually a
  // vec_xxsldi intrinsic but it is not supported by some gcc versions.
  // So we shift (52-32) bits and do a word swap with zeros.
  _EIGEN_DECLARE_CONST_Packet4i(1023, 1023);
  _EIGEN_DECLARE_CONST_Packet4i(20, 20);    // 52 - 32

  Packet4i emm04i = reinterpret_cast<Packet4i>(emm0);
  emm04i = vec_add(emm04i, p4i_1023);
  emm04i = vec_sl(emm04i, reinterpret_cast<Packet4ui>(p4i_20));
  static const Packet16uc perm = {
    0x14, 0x15, 0x16, 0x17, 0x00, 0x01, 0x02, 0x03, 
    0x1c, 0x1d, 0x1e, 0x1f, 0x08, 0x09, 0x0a, 0x0b };
#ifdef  _BIG_ENDIAN
  emm0 = reinterpret_cast<Packet2l>(vec_perm(p4i_ZERO, emm04i, perm));
#else
  emm0 = reinterpret_cast<Packet2l>(vec_perm(emm04i, p4i_ZERO, perm));
#endif

#endif

  // Altivec's max & min operators just drop silent NaNs. Check NaNs in 
  // inputs and return them unmodified.
  Packet2ul isnumber_mask = reinterpret_cast<Packet2ul>(vec_cmpeq(_x, _x));
  return vec_sel(_x, pmax(pmul(x, reinterpret_cast<Packet2d>(emm0)), _x),
                 isnumber_mask);
}
#endif

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_MATH_FUNCTIONS_ALTIVEC_H
