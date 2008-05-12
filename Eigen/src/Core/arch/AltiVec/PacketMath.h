// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Konstantinos Margaritis <markos@codex.gr>
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
//
// Eigen is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// Alternatively, you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of
// the License, or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License or the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License and a copy of the GNU General Public License along with
// Eigen. If not, see <http://www.gnu.org/licenses/>.

#ifndef EIGEN_PACKET_MATH_ALTIVEC_H
#define EIGEN_PACKET_MATH_ALTIVEC_H

#ifndef EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 4
#endif

static const vector int   v0i   = vec_splat_u32(0);
static const vector int   v16i_ = vec_splat_u32(-16);
static const vector float v0f   = (vector float) v0i;

template<> struct ei_packet_traits<float>  { typedef vector float type; enum {size=4}; };
template<> struct ei_packet_traits<int>    { typedef vector int type; enum {size=4}; };

inline vector float  ei_padd(const vector float   a, const vector float   b) { return vec_add(a,b); }
inline vector int    ei_padd(const vector int     a, const vector int     b) { return vec_add(a,b); }

inline vector float  ei_psub(const vector float   a, const vector float   b) { return vec_sub(a,b); }
inline vector int    ei_psub(const vector int     a, const vector int     b) { return vec_sub(a,b); }

inline vector float  ei_pmul(const vector float   a, const vector float   b) { return vec_madd(a,b, v0f); }
inline vector int    ei_pmul(const vector int     a, const vector int     b)
{
  // Taken from http://developer.apple.com/hardwaredrivers/ve/algorithms.html#Multiply32

  //Set up constants
  vector int bswap, lowProduct, highProduct;

  //Do real work
  bswap = vec_rl( (vector unsigned int)b, (vector unsigned int)v16i_ );
  lowProduct = vec_mulo( (vector short)a,(vector short)b );
  highProduct = vec_msum((vector short)a,(vector short)bswap, v0i);
  highProduct = vec_sl( (vector unsigned int)highProduct, (vector unsigned int)v16i_ );
  return vec_add( lowProduct, highProduct );
}

inline vector float ei_pmadd(const vector float   a, const vector float   b, const vector float c) { return vec_madd(a, b, c); }

inline vector float  ei_pmin(const vector float   a, const vector float   b) { return vec_min(a,b); }
inline vector int    ei_pmin(const vector int     a, const vector int     b) { return vec_min(a,b); }

inline vector float  ei_pmax(const vector float   a, const vector float   b) { return vec_max(a,b); }
inline vector int    ei_pmax(const vector int     a, const vector int     b) { return vec_max(a,b); }

inline vector float  ei_pload(const float*   from) { return vec_ld(0, from); }
inline vector int    ei_pload(const int*     from) { return vec_ld(0, from); }

inline vector float  ei_pset1(const float&  from)
{
  static float __attribute__(aligned(16)) af[4];
  af[0] = from;
  vector float vc = vec_ld(0, af);
  vc = vec_splat(vc, 0);
  return vc;
}

inline vector int    ei_pset1(const int&    from)
{
  static int __attribute__(aligned(16)) ai[4];
  ai[0] = from;
  vector int vc = vec_ld(0, ai);
  vc = vec_splat(vc, 0);
  return vc;
}

inline void ei_pstore(float*   to, const vector float   from) { vec_st(from, 0, to); }
inline void ei_pstore(int*     to, const vector int     from) { vec_st(from, 0, to); }

inline float  ei_pfirst(const vector float  a)
{
  static float __attribute__(aligned(16)) af[4];
  vec_st(a, 0, af);
  return af[0];
}

inline int    ei_pfirst(const vector int    a)
{
  static int __attribute__(aligned(16)) ai[4];
  vec_st(a, 0, ai);
  return ai[0];
}

#endif // EIGEN_PACKET_MATH_ALTIVEC_H
