// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Konstantinos Margaritis <markos@freevec.org>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PACKET_MATH_ZVECTOR_H
#define EIGEN_PACKET_MATH_ZVECTOR_H

namespace Eigen {

namespace internal {

#ifndef EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 4
#endif

#ifndef EIGEN_HAS_SINGLE_INSTRUCTION_MADD
#define EIGEN_HAS_SINGLE_INSTRUCTION_MADD
#endif

#ifndef EIGEN_HAS_SINGLE_INSTRUCTION_CJMADD
#define EIGEN_HAS_SINGLE_INSTRUCTION_CJMADD
#endif

// NOTE Altivec has 32 registers, but Eigen only accepts a value of 8 or 16
#ifndef EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS
#define EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS  32
#endif

typedef __vector float               Packet4f;
typedef __vector int                 Packet4i;
typedef __vector unsigned int        Packet4ui;
typedef __vector __bool int          Packet4bi;
typedef __vector short int           Packet8i;
typedef __vector unsigned char       Packet16uc;
typedef __vector double              Packet2d;
typedef __vector unsigned long long  Packet2ul;
typedef __vector long long           Packet2l;

typedef union {
  float f[4];
  double d[2];
  int i[4];
  Packet4f v4f;
  Packet4i v4i;
  Packet2d v2d;
} Packet;

// We don't want to write the same code all the time, but we need to reuse the constants
// and it doesn't really work to declare them global, so we define macros instead

#define _EIGEN_DECLARE_CONST_FAST_Packet4f(NAME,X) \
  Packet4f p4f_##NAME = reinterpret_cast<Packet4f>(vec_splat_s32(X))

#define _EIGEN_DECLARE_CONST_FAST_Packet4i(NAME,X) \
  Packet4i p4i_##NAME = reinterpret_cast<Packet4i>(vec_splat_s32(X))

#define _EIGEN_DECLARE_CONST_FAST_Packet2d(NAME,X) \
  Packet2d p2d_##NAME = reinterpret_cast<Packet2d>(vec_splat_s64(X))

#define _EIGEN_DECLARE_CONST_FAST_Packet2l(NAME,X) \
  Packet2l p2l_##NAME = reinterpret_cast<Packet2l>(vec_splat_s64(X))

#define _EIGEN_DECLARE_CONST_Packet4f(NAME,X) \
  Packet4f p4f_##NAME = pset1<Packet4f>(X)

#define _EIGEN_DECLARE_CONST_Packet4i(NAME,X) \
  Packet4i p4i_##NAME = pset1<Packet4i>(X)

#define _EIGEN_DECLARE_CONST_Packet2d(NAME,X) \
  Packet2d p2d_##NAME = pset1<Packet2d>(X)

#define _EIGEN_DECLARE_CONST_Packet2l(NAME,X) \
  Packet2l p2l_##NAME = pset1<Packet2l>(X)

#define _EIGEN_DECLARE_CONST_Packet4f_FROM_INT(NAME,X) \
  const Packet4f p4f_##NAME = reinterpret_cast<Packet4f>(pset1<Packet4i>(X))

// These constants are endian-agnostic
static _EIGEN_DECLARE_CONST_FAST_Packet4f(ZERO, 0); //{ 0.0, 0.0, 0.0, 0.0}
static _EIGEN_DECLARE_CONST_FAST_Packet4i(ZERO, 0); //{ 0, 0, 0, 0,}
static _EIGEN_DECLARE_CONST_FAST_Packet4i(ONE, 1); //{ 1, 1, 1, 1}

static _EIGEN_DECLARE_CONST_FAST_Packet2d(ZERO, 0);
static _EIGEN_DECLARE_CONST_FAST_Packet2l(ZERO, 0);

static Packet4f p4f_ONE = { 1.0, 1.0, 1.0, 1.0 }; 
static Packet2d p2d_ONE = { 1.0, 1.0 }; 
static Packet2d p2d_ZERO_ = { -0.0, -0.0 };

/*
static Packet4f p4f_ONE = vec_ctf(p4i_ONE, 0); //{ 1.0, 1.0, 1.0, 1.0}
static _EIGEN_DECLARE_CONST_FAST_Packet4i(MINUS16,-16); //{ -16, -16, -16, -16}
static _EIGEN_DECLARE_CONST_FAST_Packet4i(MINUS1,-1); //{ -1, -1, -1, -1}
static Packet4f p4f_ZERO_ = (Packet4f) vec_sl((Packet4ui)p4i_MINUS1, (Packet4ui)p4i_MINUS1); //{ 0x80000000, 0x80000000, 0x80000000, 0x80000000}
*/
static Packet4f p4f_COUNTDOWN = { 0.0, 1.0, 2.0, 3.0 };
static Packet4i p4i_COUNTDOWN = { 0, 1, 2, 3 };
static Packet2d p2d_COUNTDOWN = reinterpret_cast<Packet2d>(vec_sld(reinterpret_cast<Packet16uc>(p2d_ZERO), reinterpret_cast<Packet16uc>(p2d_ONE), 8));

static Packet16uc p16uc_PSET64_HI = { 0,1,2,3, 4,5,6,7, 0,1,2,3, 4,5,6,7 };
static Packet16uc p16uc_DUPLICATE32_HI = { 0,1,2,3, 0,1,2,3, 4,5,6,7, 4,5,6,7 };

// Mask alignment
#define _EIGEN_MASK_ALIGNMENT	0xfffffffffffffff0

#define _EIGEN_ALIGNED_PTR(x)	((ptrdiff_t)(x) & _EIGEN_MASK_ALIGNMENT)

// Handle endianness properly while loading constants
// Define global static constants:
/*
static Packet16uc p16uc_FORWARD = vec_lvsl(0, (float*)0);*/
static Packet16uc p16uc_REVERSE32 = { 12,13,14,15, 8,9,10,11, 4,5,6,7, 0,1,2,3 };
static Packet16uc p16uc_REVERSE64 = { 8,9,10,11, 12,13,14,15, 0,1,2,3, 4,5,6,7 };
/*
static Packet16uc p16uc_PSET32_WODD   = vec_sld((Packet16uc) vec_splat((Packet4ui)p16uc_FORWARD, 0), (Packet16uc) vec_splat((Packet4ui)p16uc_FORWARD, 2), 8);//{ 0,1,2,3, 0,1,2,3, 8,9,10,11, 8,9,10,11 };
static Packet16uc p16uc_PSET32_WEVEN  = vec_sld(p16uc_DUPLICATE32_HI, (Packet16uc) vec_splat((Packet4ui)p16uc_FORWARD, 3), 8);//{ 4,5,6,7, 4,5,6,7, 12,13,14,15, 12,13,14,15 };
static Packet16uc p16uc_HALF64_0_16 = vec_sld((Packet16uc)p4i_ZERO, vec_splat((Packet16uc) vec_abs(p4i_MINUS16), 3), 8);      //{ 0,0,0,0, 0,0,0,0, 16,16,16,16, 16,16,16,16};

static Packet16uc p16uc_PSET64_HI = (Packet16uc) vec_mergeh((Packet4ui)p16uc_PSET32_WODD, (Packet4ui)p16uc_PSET32_WEVEN);     //{ 0,1,2,3, 4,5,6,7, 0,1,2,3, 4,5,6,7 };
static Packet16uc p16uc_PSET64_LO = (Packet16uc) vec_mergel((Packet4ui)p16uc_PSET32_WODD, (Packet4ui)p16uc_PSET32_WEVEN);     //{ 8,9,10,11, 12,13,14,15, 8,9,10,11, 12,13,14,15 };
static Packet16uc p16uc_TRANSPOSE64_HI = vec_add(p16uc_PSET64_HI, p16uc_HALF64_0_16);                                         //{ 0,1,2,3, 4,5,6,7, 16,17,18,19, 20,21,22,23};
static Packet16uc p16uc_TRANSPOSE64_LO = vec_add(p16uc_PSET64_LO, p16uc_HALF64_0_16);                                         //{ 8,9,10,11, 12,13,14,15, 24,25,26,27, 28,29,30,31};*/
static Packet16uc p16uc_TRANSPOSE64_HI = { 0,1,2,3, 4,5,6,7, 16,17,18,19, 20,21,22,23};
static Packet16uc p16uc_TRANSPOSE64_LO = { 8,9,10,11, 12,13,14,15, 24,25,26,27, 28,29,30,31};

static Packet16uc p16uc_COMPLEX32_REV = vec_sld(p16uc_REVERSE32, p16uc_REVERSE32, 8);                                         //{ 4,5,6,7, 0,1,2,3, 12,13,14,15, 8,9,10,11 };

//static Packet16uc p16uc_COMPLEX32_REV2 = vec_sld(p16uc_FORWARD, p16uc_FORWARD, 8);                                            //{ 8,9,10,11, 12,13,14,15, 0,1,2,3, 4,5,6,7 };


#if EIGEN_HAS_BUILTIN(__builtin_prefetch) || EIGEN_COMP_GNUC
  #define EIGEN_ZVECTOR_PREFETCH(ADDR) __builtin_prefetch(ADDR);
#else
  #define EIGEN_ZVECTOR_PREFETCH(ADDR) asm( "   pfd [%[addr]]\n" :: [addr] "r" (ADDR) : "cc" );
#endif

template<> struct packet_traits<float>  : default_packet_traits
{
  typedef Packet4f type;
  typedef Packet4f half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 4,
    HasHalfPacket = 0,

    // FIXME check the Has*
    HasAdd  = 1,
    HasSub  = 1,
    HasMul  = 1,
    HasDiv  = 1,
    HasSin  = 0,
    HasCos  = 0,
    HasLog  = 1,
    HasExp  = 1,
    HasSqrt = 0
  };
};

template<> struct packet_traits<int>    : default_packet_traits
{
  typedef Packet4i type;
  typedef Packet4i half;
  enum {
    // FIXME check the Has*
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 4,
    HasHalfPacket = 0,

    // FIXME check the Has*
    HasAdd  = 1,
    HasSub  = 1,
    HasMul  = 1,
    HasDiv  = 1,
    HasSin  = 0,
    HasCos  = 0,
    HasLog  = 1,
    HasExp  = 1,
    HasSqrt = 0
  };
};

template<> struct packet_traits<double> : default_packet_traits
{
  typedef Packet2d type;
  typedef Packet2d half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size=2,
    HasHalfPacket = 1,

    // FIXME check the Has*
    HasAdd  = 1,
    HasSub  = 1,
    HasMul  = 1,
    HasDiv  = 1,
    HasSin  = 0,
    HasCos  = 0,
    HasLog  = 1,
    HasExp  = 1,
    HasSqrt = 1
  };
};

template<> struct unpacket_traits<Packet4f> { typedef float  type; enum {size=4, alignment=Aligned16}; typedef Packet4f half; };
template<> struct unpacket_traits<Packet4i> { typedef int    type; enum {size=4, alignment=Aligned16}; typedef Packet4i half; };
template<> struct unpacket_traits<Packet2d> { typedef double type; enum {size=2, alignment=Aligned16}; typedef Packet2d half; };

inline std::ostream & operator <<(std::ostream & s, const Packet16uc & v)
{
  union {
    Packet16uc   v;
    unsigned char n[16];
  } vt;
  vt.v = v;
  for (int i=0; i< 16; i++)
    s << (int)vt.n[i] << ", ";
  return s;
}

inline std::ostream & operator <<(std::ostream & s, const Packet4f & v)
{
  union {
    Packet4f   v;
    float n[4];
  } vt;
  vt.v = v;
  s << vt.n[0] << ", " << vt.n[1] << ", " << vt.n[2] << ", " << vt.n[3];
  return s;
}

inline std::ostream & operator <<(std::ostream & s, const Packet4i & v)
{
  union {
    Packet4i   v;
    int n[4];
  } vt;
  vt.v = v;
  s << vt.n[0] << ", " << vt.n[1] << ", " << vt.n[2] << ", " << vt.n[3];
  return s;
}

inline std::ostream & operator <<(std::ostream & s, const Packet4ui & v)
{
  union {
    Packet4ui   v;
    unsigned int n[4];
  } vt;
  vt.v = v;
  s << vt.n[0] << ", " << vt.n[1] << ", " << vt.n[2] << ", " << vt.n[3];
  return s;
}

inline std::ostream & operator <<(std::ostream & s, const Packet2d & v)
{
  union {
    Packet2d   v;
    double n[2];
  } vt;
  vt.v = v;
  s << vt.n[0] << ", " << vt.n[1];
  return s;
}

template<int Offset>
struct palign_impl<Offset,Packet4f>
{
  static EIGEN_STRONG_INLINE void run(Packet4f& first, const Packet4f& second)
  {
    switch (Offset % 4) {
    case 1:
      first = reinterpret_cast<Packet4f>(vec_sld(reinterpret_cast<Packet4i>(first), reinterpret_cast<Packet4i>(second), 4)); break;
    case 2:
      first = reinterpret_cast<Packet4f>(vec_sld(reinterpret_cast<Packet4i>(first), reinterpret_cast<Packet4i>(second), 8)); break;
    case 3:
      first = reinterpret_cast<Packet4f>(vec_sld(reinterpret_cast<Packet4i>(first), reinterpret_cast<Packet4i>(second), 12)); break;
    }
  }
};

template<int Offset>
struct palign_impl<Offset,Packet4i>
{
  static EIGEN_STRONG_INLINE void run(Packet4i& first, const Packet4i& second)
  {
    switch (Offset % 4) {
    case 1:
      first = vec_sld(first, second, 4); break;
    case 2:
      first = vec_sld(first, second, 8); break;
    case 3:
      first = vec_sld(first, second, 12); break;
    }
  }
};

template<int Offset>
struct palign_impl<Offset,Packet2d>
{
  static EIGEN_STRONG_INLINE void run(Packet2d& first, const Packet2d& second)
  {
    if (Offset == 1)
      first = reinterpret_cast<Packet2d>(vec_sld(reinterpret_cast<Packet4i>(first), reinterpret_cast<Packet4i>(second), 8));
  }
};

// Need to define them first or we get specialization after instantiation errors
template<> EIGEN_STRONG_INLINE Packet4f pload<Packet4f>(const float* from)
{
  // FIXME: No intrinsic yet
  EIGEN_DEBUG_ALIGNED_LOAD
  Packet *vfrom;
  vfrom = (Packet *) from;
  return vfrom->v4f;
}

template<> EIGEN_STRONG_INLINE Packet4i pload<Packet4i>(const int*     from)
{
  // FIXME: No intrinsic yet
  EIGEN_DEBUG_ALIGNED_LOAD
  Packet *vfrom;
  vfrom = (Packet *) from;
  return vfrom->v4i;
}

template<> EIGEN_STRONG_INLINE Packet2d pload<Packet2d>(const double* from)
{
  EIGEN_DEBUG_ALIGNED_LOAD
  Packet *vfrom;
  vfrom = (Packet *) from;
  return vfrom->v2d;
}

template<> EIGEN_STRONG_INLINE void pstore<float>(float*   to, const Packet4f& from)
{
  // FIXME: No intrinsic yet
  EIGEN_DEBUG_ALIGNED_STORE
  Packet *vto;
  vto = (Packet *) to;
  vto->v4f = from;
}

template<> EIGEN_STRONG_INLINE void pstore<int>(int*       to, const Packet4i& from)
{
  // FIXME: No intrinsic yet
  EIGEN_DEBUG_ALIGNED_STORE
  Packet *vto;
  vto = (Packet *) to;
  vto->v4i = from;
}

template<> EIGEN_STRONG_INLINE void pstore<double>(double*   to, const Packet2d& from)
{
  // FIXME: No intrinsic yet
  EIGEN_DEBUG_ALIGNED_STORE
  Packet *vto;
  vto = (Packet *) to;
  vto->v2d = from;
}

template<> EIGEN_STRONG_INLINE Packet4f pset1<Packet4f>(const float&  from)
{
  // FIXME: Check if proper intrinsic exists
  Packet res;
  res.f[0] = from;
  res.v4f = reinterpret_cast<Packet4f>(vec_splats(res.i[0]));
  return res.v4f;
}

template<> EIGEN_STRONG_INLINE Packet4i pset1<Packet4i>(const int&    from)
{
  return vec_splats(from);
}

template<> EIGEN_STRONG_INLINE Packet2d pset1<Packet2d>(const double&  from) {
  Packet2d res;
  res = vec_splats(from);
  return res;
}

template<> EIGEN_STRONG_INLINE void
pbroadcast4<Packet4f>(const float *a,
                      Packet4f& a0, Packet4f& a1, Packet4f& a2, Packet4f& a3)
{
  a3 = pload<Packet4f>(a);
  a0 = reinterpret_cast<Packet4f>(vec_splat(reinterpret_cast<Packet4i>(a3), 0));
  a1 = reinterpret_cast<Packet4f>(vec_splat(reinterpret_cast<Packet4i>(a3), 1));
  a2 = reinterpret_cast<Packet4f>(vec_splat(reinterpret_cast<Packet4i>(a3), 2));
  a3 = reinterpret_cast<Packet4f>(vec_splat(reinterpret_cast<Packet4i>(a3), 3));
}

template<> EIGEN_STRONG_INLINE void
pbroadcast4<Packet4i>(const int *a,
                      Packet4i& a0, Packet4i& a1, Packet4i& a2, Packet4i& a3)
{
  a3 = pload<Packet4i>(a);
  a0 = vec_splat(a3, 0);
  a1 = vec_splat(a3, 1);
  a2 = vec_splat(a3, 2);
  a3 = vec_splat(a3, 3);
}

template<> EIGEN_STRONG_INLINE void
pbroadcast4<Packet2d>(const double *a,
                      Packet2d& a0, Packet2d& a1, Packet2d& a2, Packet2d& a3)
{
  a1 = pload<Packet2d>(a);
  a0 = vec_splat(a1, 0);
  a1 = vec_splat(a1, 1);
  a3 = pload<Packet2d>(a+2);
  a2 = vec_splat(a3, 0);
  a3 = vec_splat(a3, 1);
}

template<> EIGEN_DEVICE_FUNC inline Packet4f pgather<float, Packet4f>(const float* from, Index stride)
{
  float EIGEN_ALIGN16 af[4];
  af[0] = from[0*stride];
  af[1] = from[1*stride];
  af[2] = from[2*stride];
  af[3] = from[3*stride];
 return pload<Packet4f>(af);
}

template<> EIGEN_DEVICE_FUNC inline Packet4i pgather<int, Packet4i>(const int* from, Index stride)
{
  int EIGEN_ALIGN16 ai[4];
  ai[0] = from[0*stride];
  ai[1] = from[1*stride];
  ai[2] = from[2*stride];
  ai[3] = from[3*stride];
 return pload<Packet4i>(ai);
}

template<> EIGEN_DEVICE_FUNC inline Packet2d pgather<double, Packet2d>(const double* from, Index stride)
{
  double EIGEN_ALIGN16 af[2];
  af[0] = from[0*stride];
  af[1] = from[1*stride];
 return pload<Packet2d>(af);
}

template<> EIGEN_DEVICE_FUNC inline void pscatter<float, Packet4f>(float* to, const Packet4f& from, Index stride)
{
  float EIGEN_ALIGN16 af[4];
  pstore<float>(af, from);
  to[0*stride] = af[0];
  to[1*stride] = af[1];
  to[2*stride] = af[2];
  to[3*stride] = af[3];
}

template<> EIGEN_DEVICE_FUNC inline void pscatter<int, Packet4i>(int* to, const Packet4i& from, Index stride)
{
  int EIGEN_ALIGN16 ai[4];
  pstore<int>((int *)ai, from);
  to[0*stride] = ai[0];
  to[1*stride] = ai[1];
  to[2*stride] = ai[2];
  to[3*stride] = ai[3];
}

template<> EIGEN_DEVICE_FUNC inline void pscatter<double, Packet2d>(double* to, const Packet2d& from, Index stride)
{
  double EIGEN_ALIGN16 af[2];
  pstore<double>(af, from);
  to[0*stride] = af[0];
  to[1*stride] = af[1];
}

/*
template<> EIGEN_STRONG_INLINE Packet4f psub<Packet4f>(const Packet4f& a, const Packet4f& b) { return vec_sub(a,b); }
template<> EIGEN_STRONG_INLINE Packet4i psub<Packet4i>(const Packet4i& a, const Packet4i& b) { return vec_sub(a,b); }
template<> EIGEN_STRONG_INLINE Packet2d psub<Packet2d>(const Packet2d& a, const Packet2d& b) { return vec_sub(a,b); }

template<> EIGEN_STRONG_INLINE Packet4f pnegate(const Packet4f& a) { return psub<Packet4f>(p4f_ZERO, a); }
template<> EIGEN_STRONG_INLINE Packet4i pnegate(const Packet4i& a) { return psub<Packet4i>(p4i_ZERO, a); }
template<> EIGEN_STRONG_INLINE Packet2d pnegate(const Packet2d& a) { return psub<Packet2d>(p2d_ZERO, a); }
*/
template<> EIGEN_STRONG_INLINE Packet4f pconj(const Packet4f& a) { return a; }
template<> EIGEN_STRONG_INLINE Packet4i pconj(const Packet4i& a) { return a; }
template<> EIGEN_STRONG_INLINE Packet2d pconj(const Packet2d& a) { return a; }

template<> EIGEN_STRONG_INLINE Packet4f pmadd(const Packet4f& a, const Packet4f& b, const Packet4f& c)
{
  return a;
}

template<> EIGEN_STRONG_INLINE Packet4i pmadd(const Packet4i& a, const Packet4i& b, const Packet4i& c)
{
  return reinterpret_cast<Packet4i>(__builtin_s390_vmalf(reinterpret_cast<Packet4ui>(a), reinterpret_cast<Packet4ui>(b), reinterpret_cast<Packet4ui>(c)));
}

template<> EIGEN_STRONG_INLINE Packet2d pmadd(const Packet2d& a, const Packet2d& b, const Packet2d& c)
{
  return vec_madd(a, b, c);
}

template<> EIGEN_STRONG_INLINE Packet4f pmul<Packet4f>(const Packet4f& a, const Packet4f& b)
{
  return pmadd<Packet4f>(a,b,p4f_ZERO);
}

template<> EIGEN_STRONG_INLINE Packet4i pmul<Packet4i>(const Packet4i& a, const Packet4i& b)
{
  return pmadd<Packet4i>(a,b,p4i_ZERO);
}

template<> EIGEN_STRONG_INLINE Packet2d pmul<Packet2d>(const Packet2d& a, const Packet2d& b)
{
  return pmadd<Packet2d>(a,b,p2d_ZERO);
}

/*template<> EIGEN_STRONG_INLINE Packet4f pdiv<Packet4f>(const Packet4f& a, const Packet4f& b)
{
#ifndef __VSX__  // VSX actually provides a div instruction
  Packet4f t, y_0, y_1;

  // Altivec does not offer a divide instruction, we have to do a reciprocal approximation
  y_0 = vec_re(b);

  // Do one Newton-Raphson iteration to get the needed accuracy
  t   = vec_nmsub(y_0, b, p4f_ONE);
  y_1 = vec_madd(y_0, t, y_0);

  return vec_madd(a, y_1, p4f_ZERO);
#else
  return vec_div(a, b);
#endif
}

template<> EIGEN_STRONG_INLINE Packet4i pdiv<Packet4i>(const Packet4i& a, const Packet4i& b)
{ eigen_assert(false && "packet integer division are not supported by AltiVec");
  return pset1<Packet4i>(0);
}
template<> EIGEN_STRONG_INLINE Packet2d pdiv<Packet2d>(const Packet2d& a, const Packet2d& b) { return vec_div(a,b); }
*/

template<> EIGEN_STRONG_INLINE Packet4f padd<Packet4f>(const Packet4f& a, const Packet4f& b) { return pmadd<Packet4f>(a, p4f_ONE, b); }
template<> EIGEN_STRONG_INLINE Packet4i padd<Packet4i>(const Packet4i& a, const Packet4i& b) { return pmadd<Packet4i>(a, p4i_ONE, b); }
template<> EIGEN_STRONG_INLINE Packet2d padd<Packet2d>(const Packet2d& a, const Packet2d& b) { return pmadd<Packet2d>(a, p2d_ONE, b); }

template<> EIGEN_STRONG_INLINE Packet4f plset<Packet4f>(const float& a) { return padd<Packet4f>(pset1<Packet4f>(a), p4f_COUNTDOWN); }
template<> EIGEN_STRONG_INLINE Packet4i plset<Packet4i>(const int& a)   { return padd<Packet4i>(pset1<Packet4i>(a), p4i_COUNTDOWN); }
template<> EIGEN_STRONG_INLINE Packet2d plset<Packet2d>(const double& a) { return padd<Packet2d>(pset1<Packet2d>(a), p2d_COUNTDOWN); }


template<> EIGEN_STRONG_INLINE Packet4f pmin<Packet4f>(const Packet4f& a, const Packet4f& b) { return a; /*vec_min(a, b);*/ }
template<> EIGEN_STRONG_INLINE Packet4i pmin<Packet4i>(const Packet4i& a, const Packet4i& b) { return vec_min(a, b); }
template<> EIGEN_STRONG_INLINE Packet2d pmin<Packet2d>(const Packet2d& a, const Packet2d& b) { return vec_min(a, b); }

template<> EIGEN_STRONG_INLINE Packet4f pmax<Packet4f>(const Packet4f& a, const Packet4f& b) { return a; /*vec_max(a, b);*/ }
template<> EIGEN_STRONG_INLINE Packet4i pmax<Packet4i>(const Packet4i& a, const Packet4i& b) { return vec_max(a, b); }
template<> EIGEN_STRONG_INLINE Packet2d pmax<Packet2d>(const Packet2d& a, const Packet2d& b) { return vec_max(a, b); }


template<> EIGEN_STRONG_INLINE Packet4f pand<Packet4f>(const Packet4f& a, const Packet4f& b)
{
  return reinterpret_cast<Packet4f>(vec_and(reinterpret_cast<Packet4i>(a), reinterpret_cast<Packet4i>(b)));
}

template<> EIGEN_STRONG_INLINE Packet4i pand<Packet4i>(const Packet4i& a, const Packet4i& b)
{
  return vec_and(a, b);
}

template<> EIGEN_STRONG_INLINE Packet2d pand<Packet2d>(const Packet2d& a, const Packet2d& b)
{
  return vec_and(a, b);
}

template<> EIGEN_STRONG_INLINE Packet4f por<Packet4f>(const Packet4f& a, const Packet4f& b)
{
  return reinterpret_cast<Packet4f>(vec_or(reinterpret_cast<Packet4i>(a), reinterpret_cast<Packet4i>(b)));
}

template<> EIGEN_STRONG_INLINE Packet4i por<Packet4i>(const Packet4i& a, const Packet4i& b)
{
  return vec_or(a, b);
}

template<> EIGEN_STRONG_INLINE Packet2d por<Packet2d>(const Packet2d& a, const Packet2d& b)
{
  return vec_or(a, b);
}

template<> EIGEN_STRONG_INLINE Packet4f pxor<Packet4f>(const Packet4f& a, const Packet4f& b)
{
  return reinterpret_cast<Packet4f>(vec_xor(reinterpret_cast<Packet4i>(a), reinterpret_cast<Packet4i>(b)));
}

template<> EIGEN_STRONG_INLINE Packet4i pxor<Packet4i>(const Packet4i& a, const Packet4i& b)
{
  return vec_xor(a, b);
}

template<> EIGEN_STRONG_INLINE Packet2d pandnot<Packet2d>(const Packet2d& a, const Packet2d& b)
{
  return vec_and(a, vec_nor(b, b));
}

template<> EIGEN_STRONG_INLINE Packet2d pxor<Packet2d>(const Packet2d& a, const Packet2d& b)
{
  return vec_xor(a, b);
}

template<> EIGEN_STRONG_INLINE Packet4f pandnot<Packet4f>(const Packet4f& a, const Packet4f& b)
{
  return pand<Packet4f>(a, reinterpret_cast<Packet4f>(vec_nor(reinterpret_cast<Packet4i>(b), reinterpret_cast<Packet4i>(b))));
}

template<> EIGEN_STRONG_INLINE Packet4i pandnot<Packet4i>(const Packet4i& a, const Packet4i& b)
{
  return pand<Packet4i>(a, vec_nor(b, b));
}

template<> EIGEN_STRONG_INLINE Packet4f ploadu<Packet4f>(const float* from)
{
  EIGEN_DEBUG_UNALIGNED_LOAD
  Packet *vfrom;
  vfrom = (Packet *) from;
  return vfrom->v4f;
}

template<> EIGEN_STRONG_INLINE Packet4i ploadu<Packet4i>(const int*     from)
{
  EIGEN_DEBUG_UNALIGNED_LOAD
  Packet *vfrom;
  vfrom = (Packet *) from;
  return vfrom->v4i;
}

template<> EIGEN_STRONG_INLINE Packet2d ploadu<Packet2d>(const double* from)
{
  EIGEN_DEBUG_UNALIGNED_LOAD
  Packet *vfrom;
  vfrom = (Packet *) from;
  return vfrom->v2d;
}

template<> EIGEN_STRONG_INLINE Packet4f ploaddup<Packet4f>(const float*   from)
{
  Packet4f p;
  if((ptrdiff_t(from) % 16) == 0)  p = pload<Packet4f>(from);
  else                             p = ploadu<Packet4f>(from);
  return (Packet4f) vec_perm((Packet16uc)(p), (Packet16uc)(p), p16uc_DUPLICATE32_HI);
}

template<> EIGEN_STRONG_INLINE Packet4i ploaddup<Packet4i>(const int*     from)
{
  Packet4i p;
  if((ptrdiff_t(from) % 16) == 0)  p = pload<Packet4i>(from);
  else                             p = ploadu<Packet4i>(from);
  return vec_perm(p, p, p16uc_DUPLICATE32_HI);
}

template<> EIGEN_STRONG_INLINE Packet2d ploaddup<Packet2d>(const double*   from)
{
  Packet2d p;
  if((ptrdiff_t(from) % 16) == 0)  p = pload<Packet2d>(from);
  else                             p = ploadu<Packet2d>(from);
  return vec_perm(p, p, p16uc_PSET64_HI);
}

template<> EIGEN_STRONG_INLINE void pstoreu<float>(float*   to, const Packet4f& from)
{
  EIGEN_DEBUG_UNALIGNED_STORE
  Packet *vto;
  vto = (Packet *) to;
  vto->v4f = from;
}

template<> EIGEN_STRONG_INLINE void pstoreu<int>(int*       to, const Packet4i& from)
{
  EIGEN_DEBUG_UNALIGNED_STORE
  Packet *vto;
  vto = (Packet *) to;
  vto->v4i = from;
}

template<> EIGEN_STRONG_INLINE void pstoreu<double>(double*  to, const Packet2d& from)
{
  EIGEN_DEBUG_UNALIGNED_STORE
  Packet *vto;
  vto = (Packet *) to;
  vto->v2d = from;
}

template<> EIGEN_STRONG_INLINE void prefetch<float>(const float* addr)
{
  EIGEN_ZVECTOR_PREFETCH(addr);
}

template<> EIGEN_STRONG_INLINE void prefetch<int>(const int*     addr)
{
  EIGEN_ZVECTOR_PREFETCH(addr);
}

template<> EIGEN_STRONG_INLINE void prefetch<double>(const double* addr)
{
  EIGEN_ZVECTOR_PREFETCH(addr);
}

template<> EIGEN_STRONG_INLINE float  pfirst<Packet4f>(const Packet4f& a) { float EIGEN_ALIGN16 x[4]; pstore(x, a); return x[0]; }
template<> EIGEN_STRONG_INLINE int    pfirst<Packet4i>(const Packet4i& a) { int   EIGEN_ALIGN16 x[4]; pstore(x, a); return x[0]; }
template<> EIGEN_STRONG_INLINE double pfirst<Packet2d>(const Packet2d& a) { double EIGEN_ALIGN16 x[2]; pstore(x, a); return x[0]; }

template<> EIGEN_STRONG_INLINE Packet4f preverse(const Packet4f& a)
{
  return reinterpret_cast<Packet4f>(vec_perm(reinterpret_cast<Packet16uc>(a), reinterpret_cast<Packet16uc>(a), p16uc_REVERSE32));
}

template<> EIGEN_STRONG_INLINE Packet4i preverse(const Packet4i& a)
{
  return reinterpret_cast<Packet4i>(vec_perm(reinterpret_cast<Packet16uc>(a), reinterpret_cast<Packet16uc>(a), p16uc_REVERSE32));
}

template<> EIGEN_STRONG_INLINE Packet2d preverse(const Packet2d& a)
{
  return reinterpret_cast<Packet2d>(vec_perm(reinterpret_cast<Packet16uc>(a), reinterpret_cast<Packet16uc>(a), p16uc_REVERSE64));
}

template<> EIGEN_STRONG_INLINE Packet4f pabs(const Packet4f& a) { return a; /*vec_abs(a);*/ }
template<> EIGEN_STRONG_INLINE Packet4i pabs(const Packet4i& a) { return vec_abs(a); }
template<> EIGEN_STRONG_INLINE Packet2d pabs(const Packet2d& a) { return vec_abs(a); }

template<> EIGEN_STRONG_INLINE float predux<Packet4f>(const Packet4f& a)
{
  Packet4f b, sum;
  b   = reinterpret_cast<Packet4f>(vec_sld(reinterpret_cast<Packet4i>(a), reinterpret_cast<Packet4i>(a), 8));
  sum = padd<Packet4f>(a, b);
  b   = reinterpret_cast<Packet4f>(vec_sld(reinterpret_cast<Packet4i>(sum), reinterpret_cast<Packet4i>(sum), 4));
  sum = padd<Packet4f>(sum, b);
  return pfirst(sum);
}

template<> EIGEN_STRONG_INLINE int predux<Packet4i>(const Packet4i& a)
{
  Packet4i b, sum;
  b   = vec_sld(a, a, 8);
  sum = padd<Packet4i>(a, b);
  b   = vec_sld(sum, sum, 4);
  sum = padd<Packet4i>(sum, b);
  return pfirst(sum);
}

template<> EIGEN_STRONG_INLINE double predux<Packet2d>(const Packet2d& a)
{
  Packet2d b, sum;
  b   = reinterpret_cast<Packet2d>(vec_sld(reinterpret_cast<Packet4i>(a), reinterpret_cast<Packet4i>(a), 8));
  sum = padd<Packet2d>(a, b);
  return pfirst(sum);
}


template<> EIGEN_STRONG_INLINE Packet4f preduxp<Packet4f>(const Packet4f* vecs)
{
  Packet4f v[4], sum[4];

  // It's easier and faster to transpose then add as columns
  // Check: http://www.freevec.org/function/matrix_4x4_transpose_floats for explanation
  // Do the transpose, first set of moves
  v[0] = reinterpret_cast<Packet4f>(vec_mergeh(reinterpret_cast<Packet4i>(vecs[0]), reinterpret_cast<Packet4i>(vecs[2])));
  v[1] = reinterpret_cast<Packet4f>(vec_mergeh(reinterpret_cast<Packet4i>(vecs[0]), reinterpret_cast<Packet4i>(vecs[2])));
  v[2] = reinterpret_cast<Packet4f>(vec_mergeh(reinterpret_cast<Packet4i>(vecs[1]), reinterpret_cast<Packet4i>(vecs[3])));
  v[3] = reinterpret_cast<Packet4f>(vec_mergeh(reinterpret_cast<Packet4i>(vecs[1]), reinterpret_cast<Packet4i>(vecs[3])));
  // Get the resulting vectors
  sum[0] = reinterpret_cast<Packet4f>(vec_mergeh(reinterpret_cast<Packet4i>(v[0]), reinterpret_cast<Packet4i>(v[2])));
  sum[1] = reinterpret_cast<Packet4f>(vec_mergeh(reinterpret_cast<Packet4i>(v[0]), reinterpret_cast<Packet4i>(v[2])));
  sum[2] = reinterpret_cast<Packet4f>(vec_mergeh(reinterpret_cast<Packet4i>(v[1]), reinterpret_cast<Packet4i>(v[3])));
  sum[3] = reinterpret_cast<Packet4f>(vec_mergeh(reinterpret_cast<Packet4i>(v[1]), reinterpret_cast<Packet4i>(v[3])));

  // Now do the summation:
  // Lines 0+1
  sum[0] = padd<Packet4f>(sum[0], sum[1]);
  // Lines 2+3
  sum[1] = padd<Packet4f>(sum[2], sum[3]);
  // Add the results
  sum[0] = padd<Packet4f>(sum[0], sum[1]);

  return sum[0];
}

template<> EIGEN_STRONG_INLINE Packet4i preduxp<Packet4i>(const Packet4i* vecs)
{
  Packet4i v[4], sum[4];

  // It's easier and faster to transpose then add as columns
  // Check: http://www.freevec.org/function/matrix_4x4_transpose_floats for explanation
  // Do the transpose, first set of moves
  v[0] = vec_mergeh(vecs[0], vecs[2]);
  v[1] = vec_mergel(vecs[0], vecs[2]);
  v[2] = vec_mergeh(vecs[1], vecs[3]);
  v[3] = vec_mergel(vecs[1], vecs[3]);
  // Get the resulting vectors
  sum[0] = vec_mergeh(v[0], v[2]);
  sum[1] = vec_mergel(v[0], v[2]);
  sum[2] = vec_mergeh(v[1], v[3]);
  sum[3] = vec_mergel(v[1], v[3]);

  // Now do the summation:
  // Lines 0+1
  sum[0] = padd<Packet4i>(sum[0], sum[1]);
  // Lines 2+3
  sum[1] = padd<Packet4i>(sum[2], sum[3]);
  // Add the results
  sum[0] = padd<Packet4i>(sum[0], sum[1]);

  return sum[0];
}

template<> EIGEN_STRONG_INLINE Packet2d preduxp<Packet2d>(const Packet2d* vecs)
{
  Packet2d v[2], sum;
  v[0] = padd<Packet2d>(vecs[0], reinterpret_cast<Packet2d>(vec_sld(reinterpret_cast<Packet4ui>(vecs[0]), reinterpret_cast<Packet4ui>(vecs[0]), 8)));
  v[1] = padd<Packet2d>(vecs[1], reinterpret_cast<Packet2d>(vec_sld(reinterpret_cast<Packet4ui>(vecs[1]), reinterpret_cast<Packet4ui>(vecs[1]), 8)));
 
  sum = reinterpret_cast<Packet2d>(vec_sld(reinterpret_cast<Packet4ui>(v[0]), reinterpret_cast<Packet4ui>(v[1]), 8));

  return sum;
}

// Other reduction functions:
// mul
template<> EIGEN_STRONG_INLINE float predux_mul<Packet4f>(const Packet4f& a)
{
  Packet4f prod;
  prod = pmul(a, reinterpret_cast<Packet4f>(vec_sld(reinterpret_cast<Packet4i>(a), reinterpret_cast<Packet4i>(a), 8)));
  return pfirst(pmul(prod, reinterpret_cast<Packet4f>(vec_sld(reinterpret_cast<Packet4i>(prod), reinterpret_cast<Packet4i>(prod), 4))));
}

template<> EIGEN_STRONG_INLINE int predux_mul<Packet4i>(const Packet4i& a)
{
  EIGEN_ALIGN16 int aux[4];
  pstore(aux, a);
  return aux[0] * aux[1] * aux[2] * aux[3];
}

template<> EIGEN_STRONG_INLINE double predux_mul<Packet2d>(const Packet2d& a)
{
  return pfirst(pmul(a, reinterpret_cast<Packet2d>(vec_sld(reinterpret_cast<Packet4i>(a), reinterpret_cast<Packet4i>(a), 8))));
}

// min
template<> EIGEN_STRONG_INLINE float predux_min<Packet4f>(const Packet4f& a)
{
  Packet4f b, res;
  b = pmin<Packet4f>(a, reinterpret_cast<Packet4f>(vec_sld(reinterpret_cast<Packet4i>(a), reinterpret_cast<Packet4i>(a), 8)));
  res = pmin<Packet4f>(b, reinterpret_cast<Packet4f>(vec_sld(reinterpret_cast<Packet4i>(b), reinterpret_cast<Packet4i>(b), 4)));
  return pfirst(res);
}

template<> EIGEN_STRONG_INLINE int predux_min<Packet4i>(const Packet4i& a)
{
  Packet4i b, res;
  b = pmin<Packet4i>(a, vec_sld(a, a, 8));
  res = pmin<Packet4i>(b, vec_sld(b, b, 4));
  return pfirst(res);
}

template<> EIGEN_STRONG_INLINE double predux_min<Packet2d>(const Packet2d& a)
{
  return pfirst(pmin<Packet2d>(a, reinterpret_cast<Packet2d>(vec_sld(reinterpret_cast<Packet4i>(a), reinterpret_cast<Packet4i>(a), 8))));
}

// max
template<> EIGEN_STRONG_INLINE float predux_max<Packet4f>(const Packet4f& a)
{
  Packet4f b, res;
  b = pmax<Packet4f>(a, reinterpret_cast<Packet4f>(vec_sld(reinterpret_cast<Packet4i>(a), reinterpret_cast<Packet4i>(a), 8)));
  res = pmax<Packet4f>(b, reinterpret_cast<Packet4f>(vec_sld(reinterpret_cast<Packet4i>(b), reinterpret_cast<Packet4i>(b), 4)));
  return pfirst(res);
}

template<> EIGEN_STRONG_INLINE int predux_max<Packet4i>(const Packet4i& a)
{
  Packet4i b, res;
  b = pmax<Packet4i>(a, vec_sld(a, a, 8));
  res = pmax<Packet4i>(b, vec_sld(b, b, 4));
  return pfirst(res);
}

// max
template<> EIGEN_STRONG_INLINE double predux_max<Packet2d>(const Packet2d& a)
{
  return pfirst(pmax<Packet2d>(a, reinterpret_cast<Packet2d>(vec_sld(reinterpret_cast<Packet4i>(a), reinterpret_cast<Packet4i>(a), 8))));
}

EIGEN_DEVICE_FUNC inline void
ptranspose(PacketBlock<Packet4f,4>& kernel) {
  Packet4f t0, t1, t2, t3;
  t0 = reinterpret_cast<Packet4f>(vec_mergeh(reinterpret_cast<Packet4i>(kernel.packet[0]), reinterpret_cast<Packet4i>(kernel.packet[2])));
  t1 = reinterpret_cast<Packet4f>(vec_mergel(reinterpret_cast<Packet4i>(kernel.packet[0]), reinterpret_cast<Packet4i>(kernel.packet[2])));
  t2 = reinterpret_cast<Packet4f>(vec_mergeh(reinterpret_cast<Packet4i>(kernel.packet[1]), reinterpret_cast<Packet4i>(kernel.packet[3])));
  t3 = reinterpret_cast<Packet4f>(vec_mergel(reinterpret_cast<Packet4i>(kernel.packet[1]), reinterpret_cast<Packet4i>(kernel.packet[3])));
  kernel.packet[0] = reinterpret_cast<Packet4f>(vec_mergeh(reinterpret_cast<Packet4i>(t0), reinterpret_cast<Packet4i>(t2)));
  kernel.packet[1] = reinterpret_cast<Packet4f>(vec_mergel(reinterpret_cast<Packet4i>(t0), reinterpret_cast<Packet4i>(t2)));
  kernel.packet[2] = reinterpret_cast<Packet4f>(vec_mergeh(reinterpret_cast<Packet4i>(t1), reinterpret_cast<Packet4i>(t3)));
  kernel.packet[3] = reinterpret_cast<Packet4f>(vec_mergel(reinterpret_cast<Packet4i>(t1), reinterpret_cast<Packet4i>(t3)));
}

EIGEN_DEVICE_FUNC inline void
ptranspose(PacketBlock<Packet4i,4>& kernel) {
  Packet4i t0, t1, t2, t3;
  t0 = vec_mergeh(kernel.packet[0], kernel.packet[2]);
  t1 = vec_mergel(kernel.packet[0], kernel.packet[2]);
  t2 = vec_mergeh(kernel.packet[1], kernel.packet[3]);
  t3 = vec_mergel(kernel.packet[1], kernel.packet[3]);
  kernel.packet[0] = vec_mergeh(t0, t2);
  kernel.packet[1] = vec_mergel(t0, t2);
  kernel.packet[2] = vec_mergeh(t1, t3);
  kernel.packet[3] = vec_mergel(t1, t3);
}

EIGEN_DEVICE_FUNC inline void
ptranspose(PacketBlock<Packet2d,2>& kernel) {
  Packet2d t0, t1;
  t0 = vec_perm(kernel.packet[0], kernel.packet[1], p16uc_TRANSPOSE64_HI);
  t1 = vec_perm(kernel.packet[0], kernel.packet[1], p16uc_TRANSPOSE64_LO);
  kernel.packet[0] = t0;
  kernel.packet[1] = t1;
}

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_PACKET_MATH_ZVECTOR_H
